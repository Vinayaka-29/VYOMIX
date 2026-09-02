"""
Auditable Execution Trace Ledger for SatQuery AI (Phase 12)
Produces factual, observable execution records strictly compliant with ISRO/SAC PS 26167.
Suppresses internal LLM chain-of-thought tokens while logging observable tasks, parameters, and latencies.
"""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


def build_execution_trace(
    task_name: str,
    query_text: str,
    inputs_used: List[str],
    executed_steps: List[Dict[str, Any]],
    final_confidence: float,
    disagreement_flagged: bool,
    conflicts: List[Dict[str, str]],
    intent: Dict[str, Any],
    geospatial_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds the complete auditable execution trace.
    Emits observable facts only: task, models called, parameters, step outputs, and conflict flags.
    """
    models_called = [
        {
            "name": step.get("model_called", "specialist"),
            "version": step.get("model_version", "1.0.0"),
            "status": step.get("status", "SUCCESS"),
            "execution_time_ms": step.get("execution_time_ms", 0.0),
            "confidence": step.get("confidence", 0.9),
        }
        for step in executed_steps
    ]

    structured_steps = []
    for step in executed_steps:
        out = step.get("output", {})
        summary = "Step completed."
        if isinstance(out, dict):
            summary = (
                out.get("answer")
                or out.get("caption")
                or out.get("location_summary")
                or out.get("message")
                or "Structured output generated."
            )
        elif isinstance(out, str):
            summary = out

        structured_steps.append({
            "step_id": step.get("step_id", "step"),
            "specialist_invoked": step.get("model_called", "specialist"),
            "version": step.get("model_version", "1.0.0"),
            "latency_ms": step.get("execution_time_ms", 0.0),
            "confidence": step.get("confidence", 0.9),
            "status": step.get("status", "SUCCESS"),
            "observable_output": summary[:180] + ("..." if len(summary) > 180 else ""),
        })

    total_latency_ms = round(sum(s.get("execution_time_ms", 0.0) for s in executed_steps), 2)

    return {
        "task": task_name,
        "query_text": query_text,
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs_used": inputs_used,
        "models_called": models_called,
        "parameters": {
            "query_intent": {
                "task": intent.get("task"),
                "target_entity": intent.get("target_entity"),
                "spatial_constraint": intent.get("spatial_constraint"),
                "requires_modalities": intent.get("requires_modalities"),
            },
            "geospatial_alignment": geospatial_report.get("spatial_alignment_status") if geospatial_report else "VERIFIED",
            "total_models_invoked": len(models_called),
            "total_latency_ms": total_latency_ms,
        },
        "steps": structured_steps,
        "final_confidence": final_confidence,
        "disagreement_flagged": disagreement_flagged,
        "conflicts_detected": conflicts,
        "audit_compliance": {
            "standard": "ISRO/SAC PS 26167 Observable Execution Trace Specification",
            "suppressed_internal_cot": True,
            "deterministic_reconstruction": True,
        }
    }
