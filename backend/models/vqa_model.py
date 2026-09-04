"""
Single-Image Visual Question Answering (VQA) Model
Wraps the remote-sensing adapted vision-language model for geospatial question-answering.
"""
import time
from typing import Any, Dict
from models.model_server import model_server


def answer_question(image_path: str, question: str) -> Dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("A non-empty VQA question is required")
    started = time.perf_counter()
    answer = model_server.generate(
        image_path,
        f"Answer the following remote-sensing question using only the image.\nQuestion: {question.strip()}\nAnswer:",
    )
    return {
        "task": "vqa",
        "answer": answer,
        "confidence": None,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "model": model_server.model_name,
        "checkpoint": model_server.checkpoint,
        "status": "success",
        "details": {"query": question, "device": model_server.device},
        "metadata": model_server.status(),
    }
