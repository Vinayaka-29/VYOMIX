"""
Confidence Scoring & Conflict Disagreement Engine for SatQuery AI (Phase 11)
Computes multi-factor aggregate confidence scores and detects discrepancies
between specialist outputs, ensuring auditable, trustworthy intelligence.
"""
from typing import Dict, Any, List, Tuple, Optional


def evaluate_confidence_and_conflicts(
    executed_steps: List[Dict[str, Any]], 
    task_name: str,
    geospatial_report: Optional[Dict[str, Any]] = None
) -> Tuple[float, bool, List[Dict[str, str]]]:
    """
    Evaluates multi-factor confidence and surfaces internal specialist conflicts.
    Returns:
      (final_confidence: float, disagreement_flagged: bool, conflict_details: list)
    """
    if not executed_steps:
        return 0.5, False, []

    step_confs = [
        s.get("confidence") for s in executed_steps
        if isinstance(s.get("confidence"), (int, float))
    ]
    if not step_confs:
        return 0.0, False, [{
            "conflict_id": "CONFIDENCE_UNAVAILABLE",
            "type": "UNCALIBRATED_MODEL_OUTPUT",
            "description": "The specialist completed without a calibrated confidence score.",
            "specialist_a": task_name,
            "specialist_b": "confidence_engine",
        }]
    disagreements: List[Dict[str, str]] = []

    # --- Conflict Check 1: Multi-Temporal Differencing vs. VLM Reasoning ---
    if task_name == "change_vqa":
        diff_step = next((s for s in executed_steps if "differencing" in s["step_id"]), None)
        vqa_step = next((s for s in executed_steps if "reasoning" in s["step_id"] or "change_vqa" in s["step_id"]), None)

        if diff_step and vqa_step:
            diff_out = diff_step["output"]
            vqa_out = vqa_step["output"]
            pct = diff_out.get("percentage_changed", 0.0)
            vqa_text = str(vqa_out.get("answer", "")).lower()

            # Case A: Differencing detected negligible change (<1.5%) but VQA asserted substantial growth
            if pct < 1.5 and any(term in vqa_text for term in ["substantial expansion", "significant growth", "massive increase"]):
                disagreements.append({
                    "conflict_id": "CONFLICT_TEMP_01",
                    "type": "METRIC_REASONING_DISCREPANCY",
                    "description": (
                        f"Statistical pixel differencing detected negligible variation ({pct}%), "
                        f"whereas the VLM reasoning inferred substantial surface expansion. "
                        f"Flagged for human operator review."
                    ),
                    "specialist_a": "Computer-Vision Differencing Engine",
                    "specialist_b": "Bi-Temporal Change-VQA Specialist",
                })

            # Case B: Differencing detected high change (>20%) but VQA reported scene stability
            elif pct > 20.0 and any(term in vqa_text for term in ["no significant", "stable profile", "negligible"]):
                disagreements.append({
                    "conflict_id": "CONFLICT_TEMP_02",
                    "type": "HIGH_VARIANCE_IGNORED",
                    "description": (
                        f"Differencing identified high spatial variance ({pct}%), "
                        f"but the VLM reported land cover stability."
                    ),
                    "specialist_a": "Computer-Vision Differencing Engine",
                    "specialist_b": "Bi-Temporal Change-VQA Specialist",
                })

    # --- Conflict Check 2: Grounding Entity Missing ---
    if task_name == "grounding":
        ground_step = executed_steps[0]
        if not ground_step["output"].get("found", True):
            disagreements.append({
                "conflict_id": "CONFLICT_GND_01",
                "type": "TARGET_ENTITY_ABSENT",
                "description": "The requested entity could not be localized within visible raster bounds.",
                "specialist_a": "Referring-Expression Grounding Engine",
                "specialist_b": "Input Satellite Imagery",
            })

    # --- Aggregate Confidence Calculation ---
    # Harmonic or weighted mean of step confidences
    base_confidence = sum(step_confs) / len(step_confs)

    # Apply penalty if geospatial warnings were flagged
    if geospatial_report and geospatial_report.get("warnings"):
        base_confidence -= 0.05 * len(geospatial_report["warnings"])

    # Disagreement penalty
    if disagreements:
        disagreement_flagged = True
        # Apply strict 0.25 confidence penalty for internal conflict
        final_confidence = round(max(0.35, base_confidence - 0.25), 2)
    else:
        disagreement_flagged = False
        final_confidence = round(min(0.98, max(0.40, base_confidence)), 2)

    return final_confidence, disagreement_flagged, disagreements
