"""
DAG Execution Planner for SatQuery AI Central Brain
SIH Problem Statement 26167 | Team Vyomix

Generates deterministic, ordered execution plans with dependency management
and parameter bindings by querying registered specialist capabilities.
"""
from typing import Dict, Any, List, Optional
from agent.model_registry import SPECIALIST_REGISTRY, get_specialist
from agent.schemas import ExecutionPlan, ExecutionStep, ExecutionStatus


def create_execution_plan(
    validated_config: Dict[str, Any], 
    manifest_files: Dict[str, Any],
    query_text: str,
    geospatial_report: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Constructs an ordered DAG plan of specialist invocations.
    Returns:
      List[Dict[str, Any]] containing execution steps, dependencies, and parameter bindings.
    """
    task = validated_config.get("task", "single_image_vqa")
    plan: List[Dict[str, Any]] = []

    # --- PIPELINE 1: Bi-Temporal Change Detection & Change-VQA ---
    # Chained 2-step DAG: Differencing Mask Engine -> Change-VQA Specialist
    if task in ("change_vqa", "change_analysis"):
        b_slot = validated_config["before_slot"]
        a_slot = validated_config["after_slot"]
        before_path = manifest_files[b_slot]["saved_path"]
        after_path = manifest_files[a_slot]["saved_path"]

        diff_meta = get_specialist("differencing_engine") or SPECIALIST_REGISTRY["differencing_engine"]
        plan.append({
            "step_id": "step_1_cv_differencing",
            "model_id": "differencing_engine",
            "model_name": diff_meta.name if hasattr(diff_meta, "name") else diff_meta["name"],
            "model_version": diff_meta.version if hasattr(diff_meta, "version") else diff_meta["version"],
            "description": "Compute pixel-level change differencing, thresholding mask, and sector statistics.",
            "depends_on": [],
            "inputs": {
                "before_path": before_path,
                "after_path": after_path,
            },
        })

        cvqa_meta = get_specialist("change_vqa_specialist") or SPECIALIST_REGISTRY["change_vqa_specialist"]
        plan.append({
            "step_id": "step_2_temporal_reasoning",
            "model_id": "change_vqa_specialist",
            "model_name": cvqa_meta.name if hasattr(cvqa_meta, "name") else cvqa_meta["name"],
            "model_version": cvqa_meta.version if hasattr(cvqa_meta, "version") else cvqa_meta["version"],
            "description": "Synthesize temporal rasters with generated change mask to answer natural-language inquiry.",
            "depends_on": ["step_1_cv_differencing"],
            "inputs": {
                "before_path": before_path,
                "after_path": after_path,
                "question": query_text,
                "change_map_result_from_step": "step_1_cv_differencing",
            },
        })

    # --- PIPELINE 2: Optical + SAR Cross-Modal Fusion ---
    elif task in ("optical_sar_fusion", "optical_sar"):
        opt_slot = validated_config["optical_slot"]
        sar_slot = validated_config["sar_slot"]
        opt_path = manifest_files[opt_slot]["saved_path"]
        sar_path = manifest_files[sar_slot]["saved_path"]

        fus_meta = get_specialist("optical_sar_fusion_specialist") or SPECIALIST_REGISTRY["optical_sar_fusion_specialist"]
        plan.append({
            "step_id": "step_1_cross_modal_fusion",
            "model_id": "optical_sar_fusion_specialist",
            "model_name": fus_meta.name if hasattr(fus_meta, "name") else fus_meta["name"],
            "model_version": fus_meta.version if hasattr(fus_meta, "version") else fus_meta["version"],
            "description": "Execute dual-branch analysis (Optical spectral + SAR backscatter) and fuse evidence.",
            "depends_on": [],
            "inputs": {
                "optical_path": opt_path,
                "sar_path": sar_path,
                "query": query_text,
            },
        })

    # --- PIPELINE 3: Referring-Expression Grounding ---
    elif task == "grounding":
        slot = validated_config["primary_slot"]
        img_path = manifest_files[slot]["saved_path"]
        target = validated_config.get("target_entity", "region_of_interest")

        gnd_meta = get_specialist("grounding_specialist") or SPECIALIST_REGISTRY["grounding_specialist"]
        plan.append({
            "step_id": "step_1_referring_grounding",
            "model_id": "grounding_specialist",
            "model_name": gnd_meta.name if hasattr(gnd_meta, "name") else gnd_meta["name"],
            "model_version": gnd_meta.version if hasattr(gnd_meta, "version") else gnd_meta["version"],
            "description": f"Delineate referring expression bounding box for target entity: '{target}'.",
            "depends_on": [],
            "inputs": {
                "image_path": img_path,
                "expression": target,
            },
        })

    # --- PIPELINE 4: Dense Scene Captioning ---
    elif task == "captioning":
        slot = validated_config["primary_slot"]
        img_path = manifest_files[slot]["saved_path"]

        cap_meta = get_specialist("captioning_specialist") or SPECIALIST_REGISTRY["captioning_specialist"]
        plan.append({
            "step_id": "step_1_dense_captioning",
            "model_id": "captioning_specialist",
            "model_name": cap_meta.name if hasattr(cap_meta, "name") else cap_meta["name"],
            "model_version": cap_meta.version if hasattr(cap_meta, "version") else cap_meta["version"],
            "description": "Generate dense land-cover informed scene caption describing topography and infrastructure.",
            "depends_on": [],
            "inputs": {
                "image_path": img_path,
            },
        })

    # --- PIPELINE 5: Single-Image VQA ---
    else:
        slot = validated_config.get("primary_slot", list(manifest_files.keys())[0])
        img_path = manifest_files[slot]["saved_path"]

        vqa_meta = get_specialist("vqa_specialist") or SPECIALIST_REGISTRY["vqa_specialist"]
        plan.append({
            "step_id": "step_1_single_vqa",
            "model_id": "vqa_specialist",
            "model_name": vqa_meta.name if hasattr(vqa_meta, "name") else vqa_meta["name"],
            "model_version": vqa_meta.version if hasattr(vqa_meta, "version") else vqa_meta["version"],
            "description": "Execute visual question answering with remote sensing domain LoRA adapter.",
            "depends_on": [],
            "inputs": {
                "image_path": img_path,
                "question": query_text,
            },
        })

    return plan
