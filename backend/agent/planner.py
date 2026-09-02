"""
DAG Execution Planner for SatQuery AI Central Brain (Phase 8)
Generates deterministic, ordered execution plans with dependency management
and parameter bindings for all five specialist pipelines.
"""
from typing import Dict, Any, List, Optional
from agent.model_registry import SPECIALIST_REGISTRY


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
    task = validated_config["task"]
    plan: List[Dict[str, Any]] = []

    # --- PIPELINE 1: Bi-Temporal Change Detection & Change-VQA ---
    # Chained 2-step DAG: Differencing Mask Engine -> Change-VQA Specialist
    if task == "change_vqa":
        b_slot = validated_config["before_slot"]
        a_slot = validated_config["after_slot"]
        before_path = manifest_files[b_slot]["saved_path"]
        after_path = manifest_files[a_slot]["saved_path"]

        plan.append({
            "step_id": "step_1_cv_differencing",
            "model_id": "differencing_engine",
            "model_name": SPECIALIST_REGISTRY["differencing_engine"]["name"],
            "model_version": SPECIALIST_REGISTRY["differencing_engine"]["version"],
            "description": "Compute pixel-level change differencing, thresholding mask, and sector statistics.",
            "depends_on": [],
            "inputs": {
                "before_path": before_path,
                "after_path": after_path,
            },
        })

        plan.append({
            "step_id": "step_2_temporal_reasoning",
            "model_id": "change_vqa_specialist",
            "model_name": SPECIALIST_REGISTRY["change_vqa_specialist"]["name"],
            "model_version": SPECIALIST_REGISTRY["change_vqa_specialist"]["version"],
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
    # Single-step dual-branch synthesis engine
    elif task == "optical_sar_fusion":
        opt_slot = validated_config["optical_slot"]
        sar_slot = validated_config["sar_slot"]
        opt_path = manifest_files[opt_slot]["saved_path"]
        sar_path = manifest_files[sar_slot]["saved_path"]

        plan.append({
            "step_id": "step_1_cross_modal_fusion",
            "model_id": "optical_sar_fusion_specialist",
            "model_name": SPECIALIST_REGISTRY["optical_sar_fusion_specialist"]["name"],
            "model_version": SPECIALIST_REGISTRY["optical_sar_fusion_specialist"]["version"],
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

        plan.append({
            "step_id": "step_1_referring_grounding",
            "model_id": "grounding_specialist",
            "model_name": SPECIALIST_REGISTRY["grounding_specialist"]["name"],
            "model_version": SPECIALIST_REGISTRY["grounding_specialist"]["version"],
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

        plan.append({
            "step_id": "step_1_dense_captioning",
            "model_id": "captioning_specialist",
            "model_name": SPECIALIST_REGISTRY["captioning_specialist"]["name"],
            "model_version": SPECIALIST_REGISTRY["captioning_specialist"]["version"],
            "description": "Generate dense land cover, structural topography, and environmental survey.",
            "depends_on": [],
            "inputs": {
                "image_path": img_path,
            },
        })

    # --- PIPELINE 5: Single-Image VQA ---
    else:
        slot = validated_config.get("primary_slot", list(manifest_files.keys())[0])
        img_path = manifest_files[slot]["saved_path"]

        plan.append({
            "step_id": "step_1_single_vqa",
            "model_id": "vqa_specialist",
            "model_name": SPECIALIST_REGISTRY["vqa_specialist"]["name"],
            "model_version": SPECIALIST_REGISTRY["vqa_specialist"]["version"],
            "description": "Execute Visual Question Answering on single satellite tile with LoRA-adapted backbone.",
            "depends_on": [],
            "inputs": {
                "image_path": img_path,
                "question": query_text,
            },
        })

    return plan
