"""
Dense Scene Captioning Model for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Generates authentic, image-grounded remote-sensing scene descriptions
via the RemoteSensingVLMServer.
Zero predefined static templates. Dynamic confidence derived from visual-language features.
"""
import time
import logging
from typing import Dict, Any
from models.model_server import model_server

logger = logging.getLogger("satquery.captioning")


def generate_caption(image_path: str) -> Dict[str, Any]:
    """
    Generates an authentic dense scene description for a single satellite raster.
    Dispatches to model_server.generate_caption for neural forward pass and decoding.
    """
    start_time = time.time()
    try:
        res = model_server.generate_caption(image_path=image_path)
        elapsed_ms = res.get("latency_ms", round((time.time() - start_time) * 1000, 2))

        logger.info(
            f"[Captioning Specialist] Model: {res['model']} | "
            f"Confidence: {res['confidence']} | Latency: {elapsed_ms}ms"
        )

        return {
            "task": "captioning",
            "status": "success",
            "caption": res["caption"],
            "confidence": res["confidence"],
            "model": res["model"],
            "evidence": res.get("evidence", []),
            "latency_ms": elapsed_ms,
            "features_detected": res.get("features_detected", []),
        }
    except Exception as e:
        logger.error(f"[Captioning Specialist Failure]: {e}")
        raise RuntimeError(f"Scene captioning model inference failed for '{image_path}': {str(e)}")
