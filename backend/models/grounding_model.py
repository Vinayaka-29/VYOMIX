"""
Referring-Expression Grounding Model for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Locates natural-language referring expressions into real pixel-coordinate bounding boxes
[xmin, ymin, xmax, ymax] derived from the neural grounding head in RemoteSensingVLMServer.
Zero keyword matching. Zero heuristic threshold cluster masks. Graceful absent entity rejection.
"""
import time
import logging
from typing import Dict, Any
from models.model_server import model_server

logger = logging.getLogger("satquery.grounding")


def ground_expression(image_path: str, expression: str) -> Dict[str, Any]:
    """
    Grounds a natural-language referring expression to actual pixel bounding box [xmin, ymin, xmax, ymax].
    Dispatches to model_server.generate_grounding for neural grounding head forward pass.
    """
    start_time = time.time()
    try:
        res = model_server.generate_grounding(image_path=image_path, expression=expression)
        elapsed_ms = res.get("latency_ms", round((time.time() - start_time) * 1000, 2))

        logger.info(
            f"[Grounding Specialist] Model: {res['model']} | Query: '{expression}' -> "
            f"Found: {res['found']}, BBox: {res['bbox']}, Conf: {res['confidence']}"
        )

        regions = []
        if res["found"] and res["bbox"]:
            regions.append({
                "bbox": res["bbox"],
                "normalized_bbox": res["normalized_bbox"],
                "confidence": res["confidence"],
                "entity": expression,
            })

        return {
            "task": "grounding",
            "status": "success",
            "query": expression,
            "expression": expression,
            "found": res["found"],
            "bbox": res["bbox"],
            "normalized_bbox": res["normalized_bbox"],
            "regions": regions,
            "confidence": res["confidence"],
            "model": res["model"],
            "evidence": res.get("evidence", []),
            "latency_ms": elapsed_ms,
            "image_dimensions": res.get("image_dimensions", {}),
            "message": res["message"],
        }
    except Exception as e:
        logger.error(f"[Grounding Specialist Failure]: {e}")
        raise RuntimeError(f"Grounding model inference failed for expression '{expression}': {str(e)}")
