"""
Execution Planner for SatQuery AI Agentic Controller (Phase 8)
Translates validated intent and registered capabilities into an ordered DAG of execution steps.
"""
from typing import Dict, Any, List


def create_execution_plan(
    validated_config: Dict[str, Any], 
    manifest_files: Dict[str, Any],
    query_text: str
) -> List[Dict[str, Any]]:
    """
    Constructs an ordered sequence of specialist model calls.
    Returns:
      List[Dict[str, Any]] where each item describes a step to be executed.
    """
    task = validated_config["task"]
    plan: List[Dict[str, Any]] = []

    # 1. Bi-Temporal Change Route: [change_detection, change_vqa_model]
    if task == "change_vqa":
        b_slot = validated_config["before_slot"]
        a_slot = validated_config["after_slot"]
        before_path = manifest_files[b_slot]["saved_path"]
        after_path = manifest_files[a_slot]["saved_path"]

        plan.append({
            "step_id": "step_1_differencing",
            "model_id": "change_detection",
            "description": "Compute pixel-level change differencing, thresholding mask, and sector statistics.",
            "inputs": {
                "before_path": before_path,
                "after_path": after_path,
            },
        })

        plan.append({
            "step_id": "step_2_change_vqa",
            "model_id": "change_vqa_model",
            "description": "Synthesize temporal rasters with generated change map to answer natural-language inquiry.",
            "inputs": {
                "before_path": before_path,
                "after_path": after_path,
                "question": query_text,
                "change_map_result_from_step": "step_1_differencing",
            },
        })

    # 2. Optical + SAR Cross-Modal Fusion Route: [optical_sar_fusion]
    elif task == "optical_sar_fusion":
        opt_slot = validated_config["optical_slot"]
        sar_slot = validated_config["sar_slot"]
        opt_path = manifest_files[opt_slot]["saved_path"]
        sar_path = manifest_files[sar_slot]["saved_path"]

        plan.append({
            "step_id": "step_1_dual_branch_fusion",
            "model_id": "optical_sar_fusion",
            "description": "Execute dual-branch analysis (Optical spectral + SAR backscatter) and fuse evidence.",
            "inputs": {
                "optical_path": opt_path,
                "sar_path": sar_path,
                "query": query_text,
            },
        })

    # 3. Grounding Route: [grounding_model]
    elif task == "grounding":
        slot = validated_config["primary_slot"]
        img_path = manifest_files[slot]["saved_path"]
        target = validated_config.get("target_entity", "region")

        plan.append({
            "step_id": "step_1_grounding",
            "model_id": "grounding_model",
            "description": f"Delineate referring expression bounding box for '{target}'.",
            "inputs": {
                "image_path": img_path,
                "expression": target,
            },
        })

    # 4. Dense Captioning Route: [captioning_model]
    elif task == "captioning":
        slot = validated_config["primary_slot"]
        img_path = manifest_files[slot]["saved_path"]

        plan.append({
            "step_id": "step_1_captioning",
            "model_id": "captioning_model",
            "description": "Generate dense land cover and environmental scene description.",
            "inputs": {
                "image_path": img_path,
            },
        })

    # 5. Default: Single-Image VQA: [vqa_model]
    else:
        slot = validated_config.get("primary_slot") or list(manifest_files.keys())[0]
        img_path = manifest_files[slot]["saved_path"]

        plan.append({
            "step_id": "step_1_vqa",
            "model_id": "vqa_model",
            "description": "Execute Visual Question Answering on single satellite tile.",
            "inputs": {
                "image_path": img_path,
                "question": query_text,
            },
        })

    return plan
