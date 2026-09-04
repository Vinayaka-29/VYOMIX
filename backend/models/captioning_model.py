"""
Single-Image Dense Captioning Model for SatQuery AI
Generates comprehensive remote sensing scene descriptions covering land cover,
man-made infrastructure, hydrology, and spatial topography.
"""
import time
from typing import Any, Dict
from models.model_server import model_server


def generate_caption(image_path: str) -> Dict[str, Any]:
    started = time.perf_counter()
    caption = model_server.generate(
        image_path,
        "Describe this remote-sensing image precisely, including visible land cover, water, infrastructure, and spatial context.",
        max_new_tokens=192,
    )
    return {
        "task": "caption",
        "caption": caption,
        "confidence": None,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "model": model_server.model_name,
        "checkpoint": model_server.checkpoint,
        "status": "success",
        "features_detected": [],
        "metadata": model_server.status(),
    }
