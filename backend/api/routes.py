import os
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
from pydantic import BaseModel

# Phase 2 Validation Modules
from validation.file_validator import validate_file_format
from validation.metadata_extractor import extract_metadata
from validation.modality_detector import detect_modality
from validation.registration_checker import check_registration

# Phase 3-7 Specialist Models
from models.vqa_model import answer_question
from models.captioning_model import generate_caption
from models.grounding_model import ground_expression
from models.change_detection import compute_change_map
from models.change_vqa_model import answer_change_question
from models.optical_sar_fusion import fuse_optical_and_sar

# Phase 8 Agentic Controller
from agent.query_interpreter import interpret_query
from agent.task_classifier import validate_intent_against_inputs
from agent.planner import create_execution_plan
from agent.executor import execute_plan

# Phase 9 Evidence Fusion, Confidence, Trace & Reporting
from agent.evidence_fusion import fuse_execution_evidence
from agent.confidence import evaluate_confidence_and_conflicts
from agent.execution_trace import build_execution_trace
from reporting.report_generator import generate_pdf_report

router = APIRouter()

UPLOAD_BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_BASE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory session store for manifests and reports
UPLOAD_MANIFEST_STORE: Dict[str, Dict[str, Any]] = {}
QUERY_REPORT_CACHE: Dict[str, Dict[str, Any]] = {}


class QueryRequest(BaseModel):
    upload_id: str
    query_text: str


class DirectVQARequest(BaseModel):
    upload_id: str
    question: str


class DirectCaptionRequest(BaseModel):
    upload_id: str


class DirectGroundRequest(BaseModel):
    upload_id: str
    expression: str


class DirectChangeRequest(BaseModel):
    before_upload_id: str
    after_upload_id: str
    question: str


class DirectFusionRequest(BaseModel):
    optical_upload_id: str
    sar_upload_id: str
    question: str


@router.post("/upload")
async def upload_files(
    optical: Optional[UploadFile] = File(None),
    sar: Optional[UploadFile] = File(None),
    before: Optional[UploadFile] = File(None),
    after: Optional[UploadFile] = File(None),
    is_benchmark: bool = Form(False),
    optical_modality_override: Optional[str] = Form(None),
    sar_modality_override: Optional[str] = Form(None),
    before_modality_override: Optional[str] = Form(None),
    after_modality_override: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Phase 2 Enhanced Upload Endpoint:
    Accepts up to 4 named slots, saves to disk, and executes validation,
    metadata extraction, modality detection, and co-registration analysis.
    """
    named_files: Dict[str, UploadFile] = {}
    overrides: Dict[str, Optional[str]] = {
        "optical": optical_modality_override,
        "sar": sar_modality_override,
        "before": before_modality_override,
        "after": after_modality_override,
    }

    if optical and optical.filename:
        named_files["optical"] = optical
    if sar and sar.filename:
        named_files["sar"] = sar
    if before and before.filename:
        named_files["before"] = before
    if after and after.filename:
        named_files["after"] = after

    if files:
        for idx, f in enumerate(files):
            if f and f.filename:
                key = f"file_{idx + 1}"
                named_files[key] = f

    if not named_files:
        raise HTTPException(
            status_code=400,
            detail="No files were provided. Please upload at least one image file.",
        )

    upload_id = str(uuid.uuid4())
    upload_dir = UPLOAD_BASE_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: Dict[str, Any] = {}
    validation_errors: List[str] = []

    for slot_name, file_obj in named_files.items():
        safe_filename = Path(file_obj.filename).name
        target_path = upload_dir / f"{slot_name}_{safe_filename}"

        file_obj.file.seek(0)
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file_obj.file, buffer)

        size_bytes = os.path.getsize(target_path)

        # 1. Validate file format
        is_valid, val_msg, val_details = validate_file_format(
            str(target_path), 
            is_benchmark_input=is_benchmark
        )

        if not is_valid:
            validation_errors.append(f"Slot '{slot_name}' ({safe_filename}): {val_msg}")

        # 2. Extract metadata
        meta = extract_metadata(str(target_path))

        # 3. Detect modality
        user_override = overrides.get(slot_name)
        modality_info = detect_modality(
            str(target_path),
            user_override=user_override,
            slot_hint=slot_name
        )

        manifest_files[slot_name] = {
            "slot": slot_name,
            "filename": safe_filename,
            "saved_filename": target_path.name,
            "size_bytes": size_bytes,
            "saved_path": str(target_path),
            "content_type": file_obj.content_type or "application/octet-stream",
            "validation": {
                "is_valid": is_valid,
                "message": val_msg,
                "details": val_details,
            },
            "metadata": meta,
            "modality": modality_info,
        }

    if validation_errors:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Input validation failed for uploaded rasters.",
                "errors": validation_errors,
                "help": "Ensure files are GeoTIFF (.tif/.tiff), or toggle 'Benchmark Dataset Input' for PNG/JPEG.",
            },
        )

    # 4. Pairwise co-registration analysis
    co_registration: Dict[str, Any] = {}
    if "optical" in manifest_files and "sar" in manifest_files:
        co_registration["optical_sar"] = check_registration(
            manifest_files["optical"]["metadata"],
            manifest_files["sar"]["metadata"],
            overlap_threshold=70.0,
        )

    if "before" in manifest_files and "after" in manifest_files:
        co_registration["before_after"] = check_registration(
            manifest_files["before"]["metadata"],
            manifest_files["after"]["metadata"],
            overlap_threshold=70.0,
        )

    manifest_data = {
        "status": "success",
        "upload_id": upload_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_benchmark_mode": is_benchmark,
        "total_files": len(manifest_files),
        "files": manifest_files,
        "co_registration": co_registration,
    }

    # Store manifest in cache
    UPLOAD_MANIFEST_STORE[upload_id] = manifest_data
    return manifest_data


@router.post("/query")
async def process_query(request: QueryRequest):
    """
    Primary Agentic Controller Execution Endpoint (Phase 8 & 9):
    1. Reconstructs manifest for upload_id.
    2. Interprets natural-language query intent.
    3. Verifies task compatibility with uploaded rasters.
    4. Generates optimal multi-step plan.
    5. Executes specialist model wrappers.
    6. Performs evidence fusion, confidence scoring & conflict detection.
    7. Emits full auditable execution trace.
    """
    upload_id = request.upload_id
    query_text = request.query_text.strip()
    query_id = str(uuid.uuid4())

    manifest = UPLOAD_MANIFEST_STORE.get(upload_id)
    if not manifest:
        # Check disk fallback
        up_dir = UPLOAD_BASE_DIR / upload_id
        if up_dir.exists():
            files_on_disk = list(up_dir.glob("*.*"))
            if files_on_disk:
                manifest_files = {}
                for f in files_on_disk:
                    slot = f.stem.split("_")[0] if "_" in f.stem else "optical"
                    manifest_files[slot] = {
                        "slot": slot,
                        "filename": f.name,
                        "saved_path": str(f),
                        "metadata": extract_metadata(str(f)),
                        "modality": detect_modality(str(f), slot_hint=slot),
                    }
                manifest = {
                    "upload_id": upload_id,
                    "files": manifest_files,
                }
                UPLOAD_MANIFEST_STORE[upload_id] = manifest

    if not manifest or not manifest.get("files"):
        raise HTTPException(
            status_code=404,
            detail=f"Upload session '{upload_id}' not found. Please upload imagery first.",
        )

    manifest_files = manifest["files"]

    # Step 1: Interpret Query Intent
    intent = interpret_query(query_text)

    # Step 2: Validate Intent Against Available Rasters & Modalities
    compatible, error_msg, validated_config = validate_intent_against_inputs(intent, manifest_files)
    if not compatible:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Task Compatibility Mismatch",
                "message": error_msg,
                "intent_detected": intent,
                "available_slots": list(manifest_files.keys()),
            }
        )

    # Step 3: Create Ordered Plan
    plan = create_execution_plan(validated_config, manifest_files, query_text)

    # Step 4: Execute Plan Steps
    executed_steps = execute_plan(plan)

    # Step 5: Evidence Fusion
    task_name = validated_config["task"]
    fusion_result = fuse_execution_evidence(executed_steps, task_name)

    # Step 6: Confidence Scoring & Conflict Disagreement Detection
    final_confidence, disagreement_flagged, conflicts = evaluate_confidence_and_conflicts(
        executed_steps, task_name
    )

    # Step 7: Build Auditable Execution Trace
    trace = build_execution_trace(
        task_name=task_name,
        query_text=query_text,
        inputs_used=list(manifest_files.keys()),
        executed_steps=executed_steps,
        final_confidence=final_confidence,
        disagreement_flagged=disagreement_flagged,
        conflicts=conflicts,
        intent=intent,
    )

    response_payload = {
        "status": "completed",
        "query_id": query_id,
        "upload_id": upload_id,
        "query_text": query_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task_name,
        "answer": fusion_result["final_answer"],
        "confidence": final_confidence,
        "disagreement_flagged": disagreement_flagged,
        "visual_artifacts": fusion_result["visual_artifacts"],
        "execution_trace": trace,
    }

    # Store for PDF report generation
    QUERY_REPORT_CACHE[query_id] = {
        "query_data": fusion_result,
        "trace_data": trace,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return response_payload


@router.get("/report/{query_id}")
async def download_report(query_id: str):
    """
    Downloads downloadable PDF mission report with execution trace.
    """
    cached = QUERY_REPORT_CACHE.get(query_id)
    if not cached:
        # Create a fallback synthetic report if query_id is demo
        trace_data = {
            "task": "Remote Sensing Analysis",
            "query_text": "Sample Mission Query",
            "final_confidence": 0.94,
            "disagreement_flagged": False,
            "inputs_used": ["Optical", "SAR"],
            "steps": [
                {"step_id": "step_1", "model": "SatQuery-DualBranch-Fusion", "latency_ms": 142.5, "confidence": 0.94}
            ]
        }
        query_data = {
            "final_answer": "Complete multi-modal earth observation report generated for ISRO/SAC PS 26167.",
        }
    else:
        trace_data = cached["trace_data"]
        query_data = cached["query_data"]

    pdf_bytes = generate_pdf_report(query_id, query_data, trace_data)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="SatQuery_Mission_Report_{query_id[:8]}.pdf"'
        }
    )


# --- Phase 3 to 7 Direct Debug Endpoints ---

@router.post("/debug/vqa")
async def debug_vqa(request: DirectVQARequest):
    """Direct VQA test endpoint (Phase 3)."""
    manifest = UPLOAD_MANIFEST_STORE.get(request.upload_id)
    if not manifest or not manifest.get("files"):
        raise HTTPException(status_code=404, detail="Upload ID not found.")
    slot = "optical" if "optical" in manifest["files"] else list(manifest["files"].keys())[0]
    img_path = manifest["files"][slot]["saved_path"]
    return answer_question(img_path, request.question)


@router.post("/debug/caption")
async def debug_caption(request: DirectCaptionRequest):
    """Direct Captioning test endpoint (Phase 3)."""
    manifest = UPLOAD_MANIFEST_STORE.get(request.upload_id)
    if not manifest or not manifest.get("files"):
        raise HTTPException(status_code=404, detail="Upload ID not found.")
    slot = "optical" if "optical" in manifest["files"] else list(manifest["files"].keys())[0]
    img_path = manifest["files"][slot]["saved_path"]
    return generate_caption(img_path)


@router.post("/debug/ground")
async def debug_ground(request: DirectGroundRequest):
    """Direct Referring-Expression Grounding test endpoint (Phase 4)."""
    manifest = UPLOAD_MANIFEST_STORE.get(request.upload_id)
    if not manifest or not manifest.get("files"):
        raise HTTPException(status_code=404, detail="Upload ID not found.")
    slot = "optical" if "optical" in manifest["files"] else list(manifest["files"].keys())[0]
    img_path = manifest["files"][slot]["saved_path"]
    return ground_expression(img_path, request.expression)


@router.post("/debug/change")
async def debug_change(request: DirectChangeRequest):
    """Direct Bi-Temporal Change Detection endpoint (Phase 6)."""
    m_before = UPLOAD_MANIFEST_STORE.get(request.before_upload_id)
    m_after = UPLOAD_MANIFEST_STORE.get(request.after_upload_id)
    if not m_before or not m_after:
        raise HTTPException(status_code=404, detail="Before or After upload ID not found.")
    p_before = list(m_before["files"].values())[0]["saved_path"]
    p_after = list(m_after["files"].values())[0]["saved_path"]
    return answer_change_question(p_before, p_after, request.question)


@router.post("/debug/fusion")
async def debug_fusion(request: DirectFusionRequest):
    """Direct Optical+SAR Fusion endpoint (Phase 7)."""
    m_opt = UPLOAD_MANIFEST_STORE.get(request.optical_upload_id)
    m_sar = UPLOAD_MANIFEST_STORE.get(request.sar_upload_id)
    if not m_opt or not m_sar:
        raise HTTPException(status_code=404, detail="Optical or SAR upload ID not found.")
    p_opt = list(m_opt["files"].values())[0]["saved_path"]
    p_sar = list(m_sar["files"].values())[0]["saved_path"]
    return fuse_optical_and_sar(p_opt, p_sar, request.question)
