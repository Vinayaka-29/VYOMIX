"""
Confidence Scoring & Conflict Disagreement Detector for SatQuery AI (Phase 9)
Computes honest, calibrated aggregate confidence scores across multi-step specialist outputs,
explicitly detecting and flagging internal model discrepancies.
"""
from typing import Dict, Any, List, Tuple


def evaluate_confidence_and_conflicts(
    executed_steps: List[Dict[str, Any]], 
    task_name: str
) -> Tuple[float, bool, List[Dict[str, str]]]:
    """
    Computes overall confidence and checks for disagreement between steps.
    Returns:
      (final_confidence: float, disagreement_flagged: bool, conflict_details: list)
    """
    if not executed_steps:
        return 0.5, False, []

    step_confs = [s.get("confidence", 0.85) for s in executed_steps]
    disagreements: List[Dict[str, str]] = []

    # Check for specific task conflicts
    if task_name == "change_vqa":
        # Check if differencing step contradicts Change-VQA reasoning
        diff_step = next((s for s in executed_steps if "differencing" in s["step_id"]), None)
        vqa_step = next((s for s in executed_steps if "change_vqa" in s["step_id"]), None)

        if diff_step and vqa_step:
            diff_out = diff_step["output"]
            vqa_out = vqa_step["output"]
            pct = diff_out.get("percentage_changed", 0.0)
            vqa_text = vqa_out.get("answer", "").lower()

            # Disagreement Case A: Differencing detected negligible change (<1.5%) but VQA claims significant increase
            if pct < 1.5 and ("substantial expansion" in vqa_text or "significant growth" in vqa_text):
                disagreements.append({
                    "type": "METRIC_REASONING_CONFLICT",
                    "description": (
                        f"Statistical pixel differencing detected minimal variation ({pct}%), "
                        f"whereas the VLM reasoning inferred high expansion. "
                        f"Flagged for human operator review."
                    ),
                    "step_a": "Computer-Vision Differencing Engine",
                    "step_b": "Bi-Temporal Change-VQA Specialist",
                })

            # Disagreement Case B: Differencing detected high change (>15%) but VQA reports no change
            elif pct > 15.0 and ("no significant" in vqa_text or "stable" in vqa_text):
                disagreements.append({
                    "type": "METRIC_REASONING_CONFLICT",
                    "description": (
                        f"Differencing identified high spatial variance ({pct}%), "
                        f"but the VLM reported scene stability."
                    ),
                    "step_a": "Computer-Vision Differencing Engine",
                    "step_b": "Bi-Temporal Change-VQA Specialist",
                })

    # Conflict in Grounding: low confidence or entity not found
    if task_name == "grounding":
        ground_step = executed_steps[0]
        if not ground_step["output"].get("found", True):
            disagreements.append({
                "type": "OBJECT_NOT_FOUND",
                "description": "Referring target could not be localized within visible raster boundaries.",
                "step_a": "Referring-Expression Grounding Engine",
                "step_b": "Scene Imagery",
            })

    # Base aggregate confidence: geometric mean or weighted average
    avg_conf = sum(step_confs) / len(step_confs)

    if disagreements:
        # Penalize confidence when disagreement is flagged
        disagreement_flagged = True
        final_confidence = round(max(0.35, avg_conf - 0.25), 2)
    else:
        disagreement_flagged = False
        final_confidence = round(avg_conf, 2)

    return final_confidence, disagreement_flagged, disagreements
