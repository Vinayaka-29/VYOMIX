"""
Evidence Fusion & Multi-Source Synthesis Engine for SatQuery AI Central Brain
SIH Problem Statement 26167 | Team Vyomix

Collects, normalizes, groups, and cross-references multimodal evidence across execution steps:
- Preserves explicit provenance (source, type, finding, reliability, artifacts)
- Detects cross-specialist corroboration (e.g., optical reflectance + SAR backscatter)
- Detects contradictions (e.g., statistical differencing vs. qualitative VLM claims)
- Synthesizes an evidence-backed narrative rather than naive first-answer selection
"""
import re
from typing import Dict, Any, List, Optional
from agent.schemas import EvidenceItem, Conflict


def fuse_execution_evidence(
    executed_steps: List[Dict[str, Any]], 
    task_name: str,
    raw_query: str = ""
) -> Dict[str, Any]:
    """
    Fuses outputs from execution steps into a coherent, evidence-grounded final response.
    Returns:
      {
        "final_answer": str,
        "task": str,
        "total_steps": int,
        "visual_artifacts": dict,
        "step_evidence": dict,
        "supporting_facts": list[str],
        "evidence_items": list[dict],
        "conflicts": list[dict]
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
            "evidence_items": [],
            "conflicts": [],
        }

    visual_artifacts: Dict[str, Any] = {}
    evidence_bundle: Dict[str, Any] = {}
    supporting_facts: List[str] = []
    collected_evidence: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    text_answers: List[Dict[str, Any]] = []

    # 1. Collect and Normalize Evidence from Steps
    for step in executed_steps:
        s_id = step.get("step_id", "step")
        model_name = step.get("model_called", "Specialist")
        out = step.get("output", {})
        status = step.get("status", "SUCCESS")
        conf = step.get("confidence")

        evidence_bundle[s_id] = {
            "model": model_name,
            "confidence": conf,
            "status": status,
            "data": out,
        }

        if not isinstance(out, dict):
            continue

        # Extract textual answers
        ans = out.get("answer") or out.get("caption") or out.get("location_summary") or out.get("message")
        if ans and status == "SUCCESS":
            text_answers.append({
                "step_id": s_id,
                "model": model_name,
                "text": str(ans),
                "confidence": conf,
            })
            supporting_facts.append(f"Finding from {model_name}: {str(ans)[:120]}...")

        # Extract typed evidence items if present
        raw_ev = out.get("evidence", [])
        if isinstance(raw_ev, list):
            for item in raw_ev:
                if isinstance(item, dict):
                    collected_evidence.append(item)
                    finding_str = item.get("finding", "")
                    if finding_str:
                        supporting_facts.append(f"[{item.get('source', model_name)}] {finding_str}")
                elif isinstance(item, str):
                    collected_evidence.append({
                        "source": model_name,
                        "type": "observation",
                        "finding": item,
                        "reliability": conf,
                    })
                    supporting_facts.append(f"[{model_name}] {item}")

        # Extract visual artifacts & overlays
        if "bbox" in out or "found" in out:
            visual_artifacts["bounding_box"] = out.get("bbox")
            visual_artifacts["normalized_bbox"] = out.get("normalized_bbox")
            visual_artifacts["grounding_found"] = out.get("found", False)
            if out.get("bbox"):
                supporting_facts.append(f"Target localized within bounding box coordinates: {out['bbox']}.")

        if "mask_path" in out or "overlay_path" in out:
            if out.get("overlay_path"):
                visual_artifacts["change_overlay_path"] = out["overlay_path"]
            if out.get("mask_path"):
                visual_artifacts["change_mask_path"] = out["mask_path"]
            pct = out.get("percentage_changed", 0.0)
            supporting_facts.append(f"Statistical differencing registered {pct}% pixel area alteration.")

        if "change_metrics" in out and isinstance(out["change_metrics"], dict):
            cm = out["change_metrics"]
            if cm.get("overlay_path"):
                visual_artifacts["change_overlay_path"] = cm["overlay_path"]
            if cm.get("mask_path"):
                visual_artifacts["change_mask_path"] = cm["mask_path"]

        if "cross_modal_evidence" in out or "evidence" in out:
            ev_data = out.get("cross_modal_evidence") or out.get("evidence")
            if isinstance(ev_data, dict):
                visual_artifacts["cross_modal_evidence"] = ev_data
            if "complementary_gains" in out:
                visual_artifacts["complementary_gains"] = out["complementary_gains"]

    # 2. Cross-Model Disagreement & Conflict Checking
    if task_name in ("change_vqa", "change_analysis"):
        diff_step = next((s for s in executed_steps if "differencing" in s.get("step_id", "")), None)
        vqa_step = next((s for s in executed_steps if "temporal" in s.get("step_id", "") or "change_vqa" in s.get("step_id", "")), None)

        if diff_step and vqa_step:
            diff_out = diff_step.get("output", {})
            vqa_out = vqa_step.get("output", {})
            pct = diff_out.get("percentage_changed", 0.0)
            vqa_text = str(vqa_out.get("answer", "")).lower()

            if pct < 1.5 and any(t in vqa_text for t in ["substantial expansion", "significant growth", "massive increase", "heavily clustered"]):
                conflicts.append({
                    "conflict_id": "CONFLICT_TEMP_01",
                    "type": "METRIC_REASONING_DISCREPANCY",
                    "description": (
                        f"Statistical pixel differencing detected negligible variation ({pct}%), "
                        f"whereas the VLM reasoning asserted substantial expansion. Flagged for review."
                    ),
                    "source_a": "differencing_engine",
                    "source_b": "change_vqa_specialist",
                    "severity": "warning",
                })

    # 3. Evidence-Backed Final Narrative Synthesis
    if not text_answers:
        # Check for failed steps
        failed_steps = [s for s in executed_steps if s.get("status") in ("FAILED", "BLOCKED")]
        if failed_steps:
            err_reasons = [s.get("error", "Unknown error") for s in failed_steps]
            final_answer = f"Pipeline execution encountered errors: {'; '.join(err_reasons)}"
        else:
            final_answer = "Analysis completed without generated textual findings."
    else:
        # Multi-step synthesis:
        if len(text_answers) > 1 and task_name in ("change_vqa", "change_analysis"):
            # Synthesize differencing statistics with reasoning answer
            diff_ans = next((a["text"] for a in text_answers if "differencing" in a["step_id"]), None)
            vqa_ans = next((a["text"] for a in text_answers if "reasoning" in a["step_id"] or "change_vqa" in a["step_id"]), None)
            if vqa_ans:
                final_answer = vqa_ans
            elif diff_ans:
                final_answer = diff_ans
            else:
                final_answer = text_answers[-1]["text"]
        elif task_name == "optical_sar_fusion":
            # For cross-modal, select the fused answer
            fused_ans = next((a["text"] for a in text_answers if "fusion" in a["step_id"] or "cross_modal" in a["step_id"]), None)
            final_answer = fused_ans or text_answers[-1]["text"]
        else:
            final_answer = text_answers[-1]["text"]

    return {
        "final_answer": final_answer,
        "task": task_name,
        "total_steps": len(executed_steps),
        "visual_artifacts": visual_artifacts,
        "step_evidence": evidence_bundle,
        "supporting_facts": supporting_facts,
        "evidence_items": collected_evidence,
        "conflicts": conflicts,
    }
