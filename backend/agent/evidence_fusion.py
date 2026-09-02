"""
Evidence Fusion Module for SatQuery AI (Phase 9)
Unifies multi-step and multi-modal outputs into a single canonical result schema,
preserving all intermediate visual overlays, bounding boxes, and branch evidence.
"""
from typing import Dict, Any, List


def fuse_execution_evidence(
    executed_steps: List[Dict[str, Any]], 
    task_name: str
) -> Dict[str, Any]:
    """
    Fuses outputs from individual execution steps into one coherent final response.
    """
    if not executed_steps:
        return {
            "answer": "No execution steps were performed.",
            "primary_output": None,
            "evidence": {},
            "visual_artifacts": {},
        }

    # Extract primary answer & visual artifacts
    primary_answer = ""
    visual_artifacts = {}
    evidence_bundle = {}

    for step in executed_steps:
        s_id = step["step_id"]
        out = step["output"]
        evidence_bundle[s_id] = {
            "model": step["model_called"],
            "confidence": step["confidence"],
            "data": out,
        }

        if isinstance(out, dict):
            # Check for answers
            if "answer" in out and not primary_answer:
                primary_answer = out["answer"]
            elif "caption" in out and not primary_answer:
                primary_answer = out["caption"]

            # Visual overlays & boxes
            if "bbox" in out and out["bbox"] is not None:
                visual_artifacts["bounding_box"] = out["bbox"]
                visual_artifacts["normalized_bbox"] = out.get("normalized_bbox")
                visual_artifacts["grounding_found"] = out.get("found", True)
                if not primary_answer:
                    primary_answer = out.get("message", "Target region successfully grounded.")

            if "overlay_path" in out:
                visual_artifacts["change_overlay_path"] = out["overlay_path"]
                visual_artifacts["change_mask_path"] = out.get("mask_path")

            if "evidence" in out:
                # Optical + SAR dual branch evidence
                visual_artifacts["cross_modal_evidence"] = out["evidence"]

    if not primary_answer:
        last_step = executed_steps[-1]["output"]
        primary_answer = str(last_step.get("answer") or last_step.get("caption") or "Analysis complete.")

    return {
        "final_answer": primary_answer,
        "task": task_name,
        "total_steps": len(executed_steps),
        "visual_artifacts": visual_artifacts,
        "step_evidence": evidence_bundle,
    }
