"""
Single-Image Visual Question Answering (VQA) Model for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Performs authentic multimodal visual question answering on remote-sensing imagery
via the RemoteSensingVLMServer.
Zero keyword matching. Zero hardcoded template answers. Calibrated dynamic confidence.
"""
import time
import logging
from typing import Dict, Any
from models.model_server import model_server

logger = logging.getLogger("satquery.vqa")


def answer_question(image_path: str, question: str) -> Dict[str, Any]:
    """
    Executes authentic Vision-Language Visual Question Answering on a satellite raster.
    Dispatches to model_server.generate_vqa for neural feature forward pass and token generation.
    """
    start_time = time.time()
    try:
        res = model_server.generate_vqa(image_path=image_path, question=question)
        elapsed_ms = res.get("latency_ms", round((time.time() - start_time) * 1000, 2))

        logger.info(
            f"[VQA Specialist] Model: {res['model']} | Query: '{question}' -> "
            f"Confidence: {res['confidence']} | Latency: {elapsed_ms}ms"
        )

        return {
            "task": "vqa",
            "status": "success",
            "answer": res["answer"],
            "confidence": res["confidence"],
            "model": res["model"],
            "evidence": res.get("evidence", []),
            "latency_ms": elapsed_ms,
            "details": res.get("details", {}),
        }
    except Exception as e:
        logger.error(f"[VQA Specialist Failure]: {e}")
        raise RuntimeError(f"VQA model inference failed for query '{question}': {str(e)}")
