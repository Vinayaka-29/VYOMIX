import os
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Base upload directory relative to backend folder
UPLOAD_BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_BASE_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    upload_id: str
    query_text: str


class QueryResponse(BaseModel):
    status: str
    task: str
    upload_id: str
    query_text: str
    timestamp: str
    message: str


@router.post("/upload")
async def upload_files(
    optical: Optional[UploadFile] = File(None),
    sar: Optional[UploadFile] = File(None),
    before: Optional[UploadFile] = File(None),
    after: Optional[UploadFile] = File(None),
    # Also support generic multi-file upload fallback
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Accepts multipart file upload for up to 4 named slots (optical, sar, before, after)
    or general file list. Saves to backend/data/uploads/<upload_id>/ and returns manifest.
    """
    named_files: Dict[str, UploadFile] = {}
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

    for slot_name, file_obj in named_files.items():
        # Sanitize filename
        safe_filename = Path(file_obj.filename).name
        # Keep original extension, optionally prefix slot name if needed
        target_path = upload_dir / f"{slot_name}_{safe_filename}"

        # Write to disk
        file_obj.file.seek(0)
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file_obj.file, buffer)

        size_bytes = os.path.getsize(target_path)

        manifest_files[slot_name] = {
            "slot": slot_name,
            "filename": safe_filename,
            "saved_filename": target_path.name,
            "size_bytes": size_bytes,
            "saved_path": str(target_path),
            "content_type": file_obj.content_type or "application/octet-stream",
        }

    return {
        "status": "success",
        "upload_id": upload_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_files": len(manifest_files),
        "files": manifest_files,
    }


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Phase 1 query endpoint stub.
    Accepts upload_id and query_text, returns an acknowledgment stub response.
    """
    upload_dir = UPLOAD_BASE_DIR / request.upload_id
    if not upload_dir.exists():
        # Informative notice if directory doesn't exist yet
        pass

    return QueryResponse(
        status="received",
        task="not_yet_implemented",
        upload_id=request.upload_id,
        query_text=request.query_text,
        timestamp=datetime.now(timezone.utc).isoformat(),
        message="Phase 1 stub: upload received and query registered. AI routing will activate in Phase 8.",
    )
