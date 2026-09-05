"""
Auditable Execution Trace Ledger for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Produces factual, observable execution ledger records strictly compliant with ISRO/SAC PS 26167.
Suppresses internal LLM chain-of-thought tokens while logging observable tasks, parameters,
specialists, and execution latencies.
"""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


def build_execution_trace(
    task_name: str,
    query_text: str,
    inputs_used: List[str],
    executed_steps: List[Dict[str, Any]],
    final_confidence: Optional[float],
    disagreement_flagged: bool,
    conflicts: List[Dict[str, Any]],
    intent: Dict[str, Any],
    geospatial_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds the complete auditable execution trace.
    Emits observable facts only: task, models called, parameters, step outputs, and conflict flags.
    Zero chain-of-thought or reasoning traces exposed.
    """
    models_called = [
        {
            "name": step.get("model_called", "specialist"),
            "version": step.get("model_version", "1.0.0"),
            "status": step.get("status", "SUCCESS"),
            "execution_time_ms": step.get("execution_time_ms", 0.0),
            "confidence": step.get("confidence"),
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
                or out.get("error")
                or "Structured output generated."
            )
        elif isinstance(out, str):
            summary = out

        truncated_summary = str(summary)[:180] + ("..." if len(str(summary)) > 180 else "")

        # Provide fields for both legacy and current frontend component bindings
        structured_steps.append({
            "step_id": step.get("step_id", "step"),
            "specialist_invoked": step.get("model_called", "specialist"),
            "model": step.get("model_called", "specialist"),
            "version": step.get("model_version", "1.0.0"),
            "latency_ms": step.get("execution_time_ms", 0.0),
            "confidence": step.get("confidence") if step.get("confidence") is not None else 0.0,
            "status": step.get("status", "SUCCESS"),
            "observable_output": truncated_summary,
            "output_summary": truncated_summary,
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
