"""
Execution Engine for SatQuery AI Agentic Controller (Phase 8)
Dispatches and invokes specialist model wrappers according to the execution plan,
collecting intermediate outputs and step metadata.
"""
import time
from typing import Dict, Any, List

# Specialist model imports from Phases 3-7
from models.vqa_model import answer_question
from models.captioning_model import generate_caption
from models.grounding_model import ground_expression
from models.change_detection import compute_change_map
from models.change_vqa_model import answer_change_question
from models.optical_sar_fusion import fuse_optical_and_sar
from agent.model_registry import MODEL_REGISTRY


def execute_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Executes an ordered list of plan steps, passing intermediate results where required.
    Returns:
      List[Dict[str, Any]] containing execution telemetry and outputs per step.
    """
    executed_steps: List[Dict[str, Any]] = []
    step_results_cache: Dict[str, Any] = {}

    for step in plan:
        step_id = step["step_id"]
        model_id = step["model_id"]
        inputs = step["inputs"]
        reg_info = MODEL_REGISTRY.get(model_id, {})

        start_time = time.time()
        output = None

        if model_id == "vqa_model":
            output = answer_question(inputs["image_path"], inputs["question"])

        elif model_id == "captioning_model":
            output = generate_caption(inputs["image_path"])

        elif model_id == "grounding_model":
            output = ground_expression(inputs["image_path"], inputs["expression"])

        elif model_id == "change_detection":
            output = compute_change_map(inputs["before_path"], inputs["after_path"])

        elif model_id == "change_vqa_model":
            # Resolve change map from previous step if chained
            change_res = None
            if "change_map_result_from_step" in inputs:
                dep_step = inputs["change_map_result_from_step"]
                change_res = step_results_cache.get(dep_step)
            output = answer_change_question(
                inputs["before_path"],
                inputs["after_path"],
                inputs["question"],
                change_map_result=change_res,
            )

        elif model_id == "optical_sar_fusion":
            output = fuse_optical_and_sar(
                inputs["optical_path"],
                inputs["sar_path"],
                inputs["query"],
            )

        else:
            raise ValueError(f"Unknown model identifier: {model_id}")

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        step_results_cache[step_id] = output

        # Extract confidence if available
        step_conf = output.get("confidence", 0.90) if isinstance(output, dict) else 0.90

        executed_steps.append({
            "step_id": step_id,
            "model_called": reg_info.get("name", model_id),
            "model_version": reg_info.get("version", "1.0.0"),
            "execution_time_ms": elapsed_ms,
            "confidence": step_conf,
            "output": output,
        })

    return executed_steps
