"""
Step Execution Engine for SatQuery AI Central Brain
SIH Problem Statement 26167 | Team Vyomix

Executes DAG plan steps using standardized SpecialistAdapters, resolves
intermediate DAG dependencies, isolates specialist failures, enforces BLOCKED cascades
upon upstream failures, and records authentic execution latencies with zero fabricated confidences.
"""
import time
import logging
from typing import Dict, Any, List, Optional

from agent.schemas import (
    SpecialistRequest,
    SpecialistResult,
    ExecutionStatus,
)
from agent.adapters import get_adapter

logger = logging.getLogger("satquery.brain.executor")


def execute_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Executes ordered plan steps via specialist adapters, passing intermediate artifacts
    along DAG edges and propagating BLOCKED states when dependencies fail.
    Returns:
      List[Dict[str, Any]] containing execution telemetry, status, and outputs per step.
    """
    executed_steps: List[Dict[str, Any]] = []
    step_cache: Dict[str, Any] = {}
    step_statuses: Dict[str, str] = {}

    for step in plan:
        step_id = step["step_id"]
        model_id = step["model_id"]
        inputs = dict(step.get("inputs", {}))
        depends_on = step.get("depends_on", [])

        # 1. Verify and check dependencies
        blocked_by: Optional[str] = None
        for dep in depends_on:
            dep_status = step_statuses.get(dep, "PENDING")
            if dep not in step_cache or dep_status in ("FAILED", "BLOCKED"):
                blocked_by = dep
                break

        if blocked_by:
            logger.warning(f"[Executor] Step '{step_id}' is BLOCKED because upstream dependency '{blocked_by}' failed or is missing.")
            status = ExecutionStatus.BLOCKED.value
            step_statuses[step_id] = status
            error_msg = f"Dependency failure: Step '{step_id}' was BLOCKED due to upstream failure in '{blocked_by}'."
            
            output_dict = {
                "error": error_msg,
                "answer": None,
                "confidence": None,
                "status": "blocked",
            }
            step_cache[step_id] = output_dict

            executed_steps.append({
                "step_id": step_id,
                "model_id": model_id,
                "model_called": step.get("model_name", model_id),
                "model_version": step.get("model_version", "1.0.0"),
                "status": status,
                "error": error_msg,
                "execution_time_ms": 0.0,
                "confidence": None,
                "output": output_dict,
            })
            continue

        # 2. Resolve intermediate data bindings (e.g. change_map_result_from_step)
        for k, v in list(inputs.items()):
            if isinstance(v, str) and v in step_cache:
                inputs[k] = step_cache[v]
            elif k.endswith("_from_step") and v in step_cache:
                resolved_key = k.replace("_from_step", "")
                inputs[resolved_key] = step_cache[v]

        # 3. Retrieve adapter
        adapter = get_adapter(model_id)
        start_time = time.time()
        output_data: Dict[str, Any] = {}
        status = ExecutionStatus.SUCCESS.value
        error_msg: Optional[str] = None
        step_conf: Optional[float] = None

        if adapter is not None:
            spec_req = SpecialistRequest(
                specialist_id=model_id,
                task=step.get("task", model_id),
                inputs=inputs,
                parameters=step.get("parameters", {}),
            )
            spec_res: SpecialistResult = adapter.execute(spec_req)
            elapsed_ms = spec_res.latency_ms or round((time.time() - start_time) * 1000, 2)

            if spec_res.status == "failed":
                status = ExecutionStatus.FAILED.value
                error_msg = "; ".join(spec_res.errors) if spec_res.errors else "Specialist execution failed."
                output_data = {
                    "error": error_msg,
                    "answer": None,
                    "confidence": None,
                }
            else:
                status = ExecutionStatus.SUCCESS.value
                output_data = spec_res.to_legacy_dict()
                
                # Merge top-level artifacts & metrics for downstream compatibility
                if spec_res.artifacts:
                    output_data.update(spec_res.artifacts)
                if spec_res.metrics:
                    output_data.update(spec_res.metrics)
                if spec_res.metadata:
                    output_data["details"] = spec_res.metadata

                if spec_res.confidence and isinstance(spec_res.confidence, dict):
                    step_conf = spec_res.confidence.get("value")
                elif isinstance(spec_res.confidence, (int, float)):
                    step_conf = float(spec_res.confidence)
        else:
            # Fallback for unregistered specialist
            error_msg = f"No adapter registered for specialist model: {model_id}"
            logger.error(f"[Executor Error]: {error_msg}")
            status = ExecutionStatus.FAILED.value
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            output_data = {
                "error": error_msg,
                "answer": None,
                "confidence": None,
            }

        step_statuses[step_id] = status
        step_cache[step_id] = output_data

        executed_steps.append({
            "step_id": step_id,
            "model_id": model_id,
            "model_called": step.get("model_name", model_id),
            "model_version": step.get("model_version", "1.0.0"),
            "status": status,
            "error": error_msg,
            "execution_time_ms": elapsed_ms,
            "confidence": step_conf,
            "output": output_data,
        })

    return executed_steps
