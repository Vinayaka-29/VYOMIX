"""
Step Execution Engine for SatQuery AI Central Brain (Phase 9)
Executes DAG plan steps, resolves intermediate dependencies between specialists,
measures execution latencies, and isolates specialist failures.
"""
import time
import logging
from typing import Dict, Any, List

# Specialist function wrappers
from models.vqa_model import answer_question
from models.captioning_model import generate_caption
from models.grounding_model import ground_expression
from models.change_detection import compute_change_map
from models.change_vqa_model import answer_change_question
from models.optical_sar_fusion import fuse_optical_and_sar

logger = logging.getLogger("satquery.brain.executor")


def execute_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Executes ordered plan steps, passing intermediate artifacts along DAG edges.
    Returns:
      List[Dict[str, Any]] containing execution telemetry, status, and outputs per step.
    """
    executed_steps: List[Dict[str, Any]] = []
    step_cache: Dict[str, Any] = {}

    for step in plan:
        step_id = step["step_id"]
        model_id = step["model_id"]
        inputs = step["inputs"]
        depends_on = step.get("depends_on", [])

        # Verify dependencies
        for dep in depends_on:
            if dep not in step_cache:
                raise RuntimeError(f"Unresolved DAG dependency: Step '{step_id}' requires '{dep}', but it has not executed.")

        start_time = time.time()
        output = None
        status = "SUCCESS"
        error_msg = None

        try:
            if model_id in ("vqa_specialist", "vqa_model"):
                output = answer_question(inputs["image_path"], inputs["question"])

            elif model_id in ("captioning_specialist", "captioning_model"):
                output = generate_caption(inputs["image_path"])

            elif model_id in ("grounding_specialist", "grounding_model"):
                output = ground_expression(inputs["image_path"], inputs["expression"])

            elif model_id in ("differencing_engine", "change_detection"):
                output = compute_change_map(inputs["before_path"], inputs["after_path"])

            elif model_id in ("change_vqa_specialist", "change_vqa_model"):
                # Resolve chained change map dependency
                dep_key = inputs.get("change_map_result_from_step")
                change_res = step_cache.get(dep_key) if dep_key else None
                output = answer_change_question(
                    inputs["before_path"],
                    inputs["after_path"],
                    inputs["question"],
                    change_map_result=change_res,
                )

            elif model_id in ("optical_sar_fusion_specialist", "optical_sar_fusion"):
                output = fuse_optical_and_sar(
                    inputs["optical_path"],
                    inputs["sar_path"],
                    inputs["query"],
                )

            else:
                raise ValueError(f"Unknown specialist model: {model_id}")

        except Exception as e:
            logger.error(f"[Executor Error] Step {step_id} failed: {e}")
            status = "FAILED"
            error_msg = str(e)
            output = {
                "error": error_msg,
                "answer": f"Specialist {model_id} encountered an error during execution: {error_msg}",
                "confidence": 0.3,
            }

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        step_cache[step_id] = output

        # Extract confidence score
        step_conf = 0.90
        if isinstance(output, dict):
            step_conf = output.get("confidence", 0.90)

        executed_steps.append({
            "step_id": step_id,
            "model_id": model_id,
            "model_called": step.get("model_name", model_id),
            "model_version": step.get("model_version", "1.0.0"),
            "status": status,
            "error": error_msg,
            "execution_time_ms": elapsed_ms,
            "confidence": step_conf,
            "output": output,
        })

    return executed_steps
