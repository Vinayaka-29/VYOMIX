"""
Referring-Expression Grounding Model for SatQuery AI
Localizes natural-language referring expressions into pixel-coordinate bounding boxes
[x_min, y_min, x_max, y_max] with graceful 'not found' rejection.
"""
import os
import time
from typing import Any, Dict, List, Optional
from models.model_server import model_server


_grounding_model = None
_grounding_processor = None
_grounding_device = None


def _load_grounding_model() -> None:
    global _grounding_model, _grounding_processor, _grounding_device
    if _grounding_model is not None:
        return
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    checkpoint = os.getenv("GROUNDING_MODEL", "IDEA-Research/grounding-dino-base")
    requested = os.getenv("DEVICE", "auto").lower()
    _grounding_device = "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
    _grounding_processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    _grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        checkpoint, trust_remote_code=True
    ).to(_grounding_device)
    _grounding_model.eval()


def ground_expression(image_path: str, expression: str) -> Dict[str, Any]:
    """
    Grounds a natural-language referring expression to a bounding box [xmin, ymin, xmax, ymax].
    Returns:
      {
        "found": bool,
        "bbox": Optional[List[int]], # [xmin, ymin, xmax, ymax]
        "normalized_bbox": Optional[List[float]], # [0.0 - 1.0]
        "confidence": float,
        "expression": str,
        "latency_ms": float,
        "model": str
      }
    """
    if not expression or not expression.strip():
        raise ValueError("A non-empty grounding expression is required")
    started = time.perf_counter()
    _load_grounding_model()
    image = model_server.load_image(image_path)
    inputs = _grounding_processor(images=image, text=expression.strip(), return_tensors="pt")
    inputs = {key: value.to(_grounding_device) if hasattr(value, "to") else value for key, value in inputs.items()}
    import torch
    with torch.inference_mode():
        outputs = _grounding_model(**inputs)
    results = _grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=float(os.getenv("GROUNDING_THRESHOLD", "0.25")),
        text_threshold=float(os.getenv("GROUNDING_TEXT_THRESHOLD", "0.20")),
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = results.get("boxes", []).detach().cpu().tolist()
    scores = results.get("scores", []).detach().cpu().tolist()
    labels = results.get("text_labels", results.get("labels", []))
    width, height = image.size
    structured_boxes: List[Dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels):
        structured_boxes.append({"box": [round(value) for value in box], "score": float(score), "label": str(label)})
    first_box: Optional[List[int]] = structured_boxes[0]["box"] if structured_boxes else None
    first_score = structured_boxes[0]["score"] if structured_boxes else None
    return {
        "task": "grounding",
        "found": bool(structured_boxes),
        "bbox": first_box,
        "boxes": structured_boxes,
        "normalized_bbox": ([round(first_box[0] / width, 4), round(first_box[1] / height, 4), round(first_box[2] / width, 4), round(first_box[3] / height, 4)] if first_box else None),
        "confidence": first_score,
        "expression": expression,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "model": os.getenv("GROUNDING_MODEL", "IDEA-Research/grounding-dino-base"),
        "image_dimensions": {"width": width, "height": height},
        "status": "success",
        "message": "Grounding detections returned." if structured_boxes else "No matching region was returned by the grounding model.",
    }
