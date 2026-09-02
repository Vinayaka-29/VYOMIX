"""
Evidence Fusion Engine for SatQuery AI Central Brain (Phase 10)
Unifies multi-step and multi-modal specialist outputs into a single canonical result schema,
preserving visual grounding overlays, change masks, and branch-specific sensor evidence.
"""
from typing import Dict, Any, List


def fuse_execution_evidence(
    executed_steps: List[Dict[str, Any]], 
    task_name: str,
    raw_query: str = ""
) -> Dict[str, Any]:
    """
    Fuses outputs from individual execution steps into one coherent final response.
    Returns:
      {
        "final_answer": str,
        "task": str,
        "total_steps": int,
        "visual_artifacts": dict,
        "step_evidence": dict,
        "supporting_facts": list
      }
    """
    if not executed_steps:
        return {
            "final_answer": "No execution steps were performed.",
            "task": task_name,
            "total_steps": 0,
            "visual_artifacts": {},
            "step_evidence": {},
            "supporting_facts": [],
        }

    primary_answer = ""
    visual_artifacts: Dict[str, Any] = {}
    evidence_bundle: Dict[str, Any] = {}
    supporting_facts: List[str] = []

    for step in executed_steps:
        s_id = step["step_id"]
        out = step["output"]
        model_name = step["model_called"]
        conf = step["confidence"]

        evidence_bundle[s_id] = {
            "model": model_name,
            "confidence": conf,
            "status": step.get("status", "SUCCESS"),
            "data": out,
        }

        if isinstance(out, dict):
            # 1. Textual Answers
            if "answer" in out and not primary_answer:
                primary_answer = out["answer"]
                supporting_facts.append(f"Grounded by {model_name}: {out['answer'][:120]}...")
            elif "caption" in out and not primary_answer:
                primary_answer = out["caption"]
                supporting_facts.append(f"Scene described by {model_name}.")

            # 2. Visual Grounding Overlays
            if "bbox" in out or "found" in out:
                visual_artifacts["bounding_box"] = out.get("bbox")
                visual_artifacts["normalized_bbox"] = out.get("normalized_bbox")
                visual_artifacts["grounding_found"] = out.get("found", False)
                if not primary_answer:
                    primary_answer = out.get("message", "Target entity successfully localized in satellite imagery.")
                if out.get("bbox"):
                    supporting_facts.append(f"Entity bounded at coordinates {out['bbox']}.")
                else:
                    supporting_facts.append("Entity not identified within visible bounds.")

            # 3. Bi-Temporal Change Visualizations
            if "overlay_path" in out:
                visual_artifacts["change_overlay_path"] = out["overlay_path"]
                visual_artifacts["change_mask_path"] = out.get("mask_path")
                pct = out.get("percentage_changed", 0.0)
                supporting_facts.append(f"Pixel differencing detected {pct}% surface variance.")

            if "change_metrics" in out:
                cm = out["change_metrics"]
                if "overlay_path" in cm and cm["overlay_path"]:
                    visual_artifacts["change_overlay_path"] = cm["overlay_path"]
                    visual_artifacts["change_mask_path"] = cm.get("mask_path")

            # 4. Optical + SAR Cross-Modal Evidence
            if "evidence" in out:
                visual_artifacts["cross_modal_evidence"] = out["evidence"]
                visual_artifacts["complementary_gains"] = out.get("complementary_gains", [])
                supporting_facts.append("Optical spectral reflectance corroborated by microwave radar backscatter.")

    if not primary_answer:
        last = executed_steps[-1]["output"]
        primary_answer = str(last.get("answer") or last.get("caption") or "Analysis completed successfully.")

    return {
        "final_answer": primary_answer,
        "task": task_name,
        "total_steps": len(executed_steps),
        "visual_artifacts": visual_artifacts,
        "step_evidence": evidence_bundle,
        "supporting_facts": supporting_facts,
    }
