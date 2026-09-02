"""
Auditable Execution Trace Ledger for SatQuery AI (Phase 9)
Constructs deterministic, factual, observable execution traces of all agentic decisions,
specialist model invocations, parameters, and confidence scores as mandated by PS 26167.
"""
from typing import Dict, Any, List


def build_execution_trace(
    task_name: str,
    query_text: str,
    inputs_used: List[str],
    executed_steps: List[Dict[str, Any]],
    final_confidence: float,
    disagreement_flagged: bool,
    conflicts: List[Dict[str, str]],
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Builds the complete auditable execution trace.
    Exposes only observable execution facts (task, models called, parameters, outputs)
    without internal chain-of-thought tokens.
    """
    models_called = [
        {
            "name": step["model_called"],
            "version": step["model_version"],
            "execution_time_ms": step["execution_time_ms"],
            "confidence": step["confidence"],
        }
        for step in executed_steps
    ]

    structured_steps = [
        {
            "step_id": step["step_id"],
            "model": step["model_called"],
            "latency_ms": step["execution_time_ms"],
            "confidence": step["confidence"],
            "output_summary": (
                step["output"].get("answer")
                or step["output"].get("caption")
                or step["output"].get("location_summary")
                or step["output"].get("message")
                or "Step executed successfully."
            ),
        }
        for step in executed_steps
    ]

    total_latency_ms = round(sum(s["execution_time_ms"] for s in executed_steps), 2)

    return {
        "task": task_name,
        "query_text": query_text,
        "inputs_used": inputs_used,
        "models_called": models_called,
        "parameters": {
            "query_intent": intent,
            "total_models_invoked": len(models_called),
            "total_latency_ms": total_latency_ms,
        },
        "steps": structured_steps,
        "final_confidence": final_confidence,
        "disagreement_flagged": disagreement_flagged,
        "conflicts_detected": conflicts,
        "audit_compliance": "PS 26167 Observable Execution Specification (Strict)",
    }
