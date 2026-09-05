"""
Confidence Scoring & Conflict Disagreement Engine for SatQuery AI Central Brain
SIH Problem Statement 26167 | Team Vyomix

Implements an explainable, deterministic mathematical confidence formulation grounded
in empirical signals:
- Model token probabilities / specialist confidence
- Physical input quality (georeferencing, spatial resolution consistency)
- Cross-specialist evidence agreement
- Disagreement penalties for metric vs. reasoning discrepancies

Mathematical Formulation:
  If no calibrated model confidences exist:
    Confidence = None (Source: "unavailable")
  Else:
    Confidence = w_m * C_model + w_i * Q_input + w_a * A_agreement - P_conflict - P_geo
    Bounded to [0.0, 1.0].
"""
from typing import Dict, Any, List, Tuple, Optional
from agent.schemas import ConfidenceResult, Conflict


def compute_defensible_confidence(
    executed_steps: List[Dict[str, Any]],
    task_name: str,
    conflicts: Optional[List[Dict[str, Any]]] = None,
    geospatial_report: Optional[Dict[str, Any]] = None
) -> ConfidenceResult:
    """
    Computes a mathematically grounded, defensible confidence score.
    Returns ConfidenceResult with full factor attribution.
    """
    if not executed_steps:
        return ConfidenceResult(
            value=None,
            source="unavailable",
            method="no_steps_executed",
            components={},
        )

    # 1. Extract genuine model confidences
    model_confs = []
    for s in executed_steps:
        conf = s.get("confidence")
        if isinstance(conf, (int, float)) and conf > 0.0:
            model_confs.append(float(conf))

    if not model_confs:
        return ConfidenceResult(
            value=None,
            source="unavailable",
            method="uncalibrated_specialists",
            components={"reason": "No participating specialists returned a calibrated confidence score."},
        )

    # Base model component (arithmetic mean of specialist confidences)
    c_model = sum(model_confs) / len(model_confs)

    # 2. Input Quality Factor (Q_input)
    # Starts at 1.0, penalizes uncalibrated or un-georeferenced imagery
    q_input = 1.0
    if geospatial_report:
        if not geospatial_report.get("is_georeferenced", True):
            q_input -= 0.15
        if geospatial_report.get("spatial_alignment_status") == "MARGINAL_OVERLAP":
            q_input -= 0.20

    # 3. Evidence Agreement Factor (A_agreement)
    a_agreement = 1.0
    has_conflicts = bool(conflicts and len(conflicts) > 0)
    if has_conflicts:
        a_agreement = 0.50

    # 4. Conflict and Geospatial Penalties
    p_conflict = 0.25 if has_conflicts else 0.0
    p_geo = 0.0
    if geospatial_report and geospatial_report.get("warnings"):
        p_geo = min(0.15, 0.05 * len(geospatial_report["warnings"]))

    # 5. Weighted Linear Combination:
    # Weights: w_m = 0.70, w_i = 0.15, w_a = 0.15
    raw_val = (0.70 * c_model) + (0.15 * q_input) + (0.15 * a_agreement) - p_conflict - p_geo
    final_val = round(min(0.98, max(0.10, raw_val)), 2)

    components = {
        "model_confidence": round(c_model, 3),
        "input_quality": round(q_input, 2),
        "evidence_agreement": round(a_agreement, 2),
        "conflict_penalty": p_conflict,
        "geospatial_penalty": p_geo,
    }

    return ConfidenceResult(
        value=final_val,
        source="weighted_evidence_aggregation",
        method="linear_weighted_factor_model",
        components=components,
    )


def evaluate_confidence_and_conflicts(
    executed_steps: List[Dict[str, Any]], 
    task_name: str,
    geospatial_report: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[float], bool, List[Dict[str, str]]]:
    """
    Backwards-compatible API returning:
      (final_confidence: Optional[float], disagreement_flagged: bool, conflict_details: list)
    """
    if not executed_steps:
        return None, False, []

    conflicts: List[Dict[str, str]] = []

    # Check 1: Multi-temporal differencing vs VLM reasoning
    if task_name in ("change_vqa", "change_analysis"):
        diff_step = next((s for s in executed_steps if "differencing" in s.get("step_id", "")), None)
        vqa_step = next((s for s in executed_steps if "reasoning" in s.get("step_id", "") or "change_vqa" in s.get("step_id", "")), None)

        if diff_step and vqa_step:
            diff_out = diff_step.get("output", {})
            vqa_out = vqa_step.get("output", {})
            pct = diff_out.get("percentage_changed", 0.0)
            vqa_text = str(vqa_out.get("answer", "")).lower()

            if pct < 1.5 and any(term in vqa_text for term in ["substantial expansion", "significant growth", "massive increase"]):
                conflicts.append({
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
            elif pct > 20.0 and any(term in vqa_text for term in ["no significant", "stable profile", "negligible"]):
                conflicts.append({
                    "conflict_id": "CONFLICT_TEMP_02",
                    "type": "HIGH_VARIANCE_IGNORED",
                    "description": (
                        f"Differencing identified high spatial variance ({pct}%), "
                        f"but the VLM reported land cover stability."
                    ),
                    "specialist_a": "Computer-Vision Differencing Engine",
                    "specialist_b": "Bi-Temporal Change-VQA Specialist",
                })

    # Check 2: Grounding target absent
    if task_name == "grounding" and executed_steps:
        ground_step = executed_steps[0]
        out = ground_step.get("output", {})
        if isinstance(out, dict) and not out.get("found", True):
            conflicts.append({
                "conflict_id": "CONFLICT_GND_01",
                "type": "TARGET_ENTITY_ABSENT",
                "description": "The requested entity could not be localized within visible raster bounds.",
                "specialist_a": "Referring-Expression Grounding Engine",
                "specialist_b": "Input Satellite Imagery",
            })

    # Compute defensible confidence
    conf_res = compute_defensible_confidence(
        executed_steps=executed_steps,
        task_name=task_name,
        conflicts=conflicts,
        geospatial_report=geospatial_report
    )

    disagreement_flagged = len(conflicts) > 0
    return conf_res.value, disagreement_flagged, conflicts
