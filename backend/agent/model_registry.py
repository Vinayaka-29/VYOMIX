"""
Specialist Model Registry for SatQuery AI Central Brain
SIH Problem Statement 26167 | Team Vyomix

Catalogues verified specialist models, algorithms, capabilities, hardware dependencies,
and operational statuses. Distinguishes verified available models from pending integrations
with zero fabricated metadata.
"""
from typing import Dict, Any, List, Optional
from agent.schemas import SpecialistMetadata, ModelStatus


SPECIALIST_REGISTRY: Dict[str, Dict[str, Any]] = {
    "vqa_specialist": {
        "id": "vqa_specialist",
        "name": "RSMultimodalTransformer-LoRA VQA Specialist",
        "description": "Performs remote-sensing visual question answering using a 4-band patch encoder and PEFT/LoRA adapter.",
        "version": "1.0.0-lora",
        "module": "models.vqa_model",
        "function": "answer_question",
        "status": "available",
        "capabilities": ["vqa", "land_cover_identification", "single_image_reasoning"],
        "modalities_supported": ["OPTICAL", "MULTISPECTRAL", "SAR"],
        "min_inputs": 1,
        "max_inputs": 1,
        "device_requirements": "cpu",
        "dependencies": ["torch", "rasterio"],
        "estimated_latency_ms": 45.0,
        "inputs": {
            "image_path": "str (path to valid GeoTIFF or benchmark image)",
            "question": "str (natural-language inquiry)",
        },
        "outputs": {
            "answer": "str",
            "confidence": "float or null when model is not calibrated",
            "evidence": "list[str]",
            "latency_ms": "float",
            "details": "dict",
        },
        "fallback_strategy": "spectral_heuristic_fallback",
    },
    "captioning_specialist": {
        "id": "captioning_specialist",
        "name": "RSMultimodalTransformer Dense Scene Captioner",
        "description": "Produces comprehensive, land-cover informed scene descriptions covering natural and built environments.",
        "version": "1.0.0-lora",
        "module": "models.captioning_model",
        "function": "generate_caption",
        "status": "available",
        "capabilities": ["captioning", "dense_captioning", "scene_description"],
        "modalities_supported": ["OPTICAL", "MULTISPECTRAL", "SAR"],
        "min_inputs": 1,
        "max_inputs": 1,
        "device_requirements": "cpu",
        "dependencies": ["torch", "rasterio"],
        "estimated_latency_ms": 40.0,
        "inputs": {
            "image_path": "str",
        },
        "outputs": {
            "caption": "str",
            "confidence": "float or null when uncalibrated",
            "evidence": "list[str]",
            "latency_ms": "float",
            "features_detected": "list",
        },
        "fallback_strategy": "spectral_heuristic_fallback",
    },
    "grounding_specialist": {
        "id": "grounding_specialist",
        "name": "Spatial Feature Activation Grounding Engine",
        "description": "Localizes natural-language referring expressions into standardized pixel bounding boxes [xmin, ymin, xmax, ymax].",
        "version": "1.0.0-spatial",
        "module": "models.grounding_model",
        "function": "ground_expression",
        "status": "available",
        "capabilities": ["grounding", "referring_expression", "object_localization"],
        "modalities_supported": ["OPTICAL", "MULTISPECTRAL"],
        "min_inputs": 1,
        "max_inputs": 1,
        "device_requirements": "cpu",
        "dependencies": ["numpy", "PIL"],
        "estimated_latency_ms": 30.0,
        "inputs": {
            "image_path": "str",
            "expression": "str",
        },
        "outputs": {
            "found": "bool",
            "bbox": "list[int] [xmin, ymin, xmax, ymax] or null",
            "normalized_bbox": "list[float] [0.0-1.0] or null",
            "confidence": "float or null",
            "message": "str",
        },
        "fallback_strategy": "explicit_not_found_rejection",
    },
    "differencing_engine": {
        "id": "differencing_engine",
        "name": "Classical CV Differencing & Morphological Mask Engine",
        "description": "Executes pixel differencing, Otsu adaptive thresholding, and morphological filtering on temporal pairs.",
        "version": "1.0.0-cv",
        "module": "models.change_detection",
        "function": "compute_change_map",
        "status": "available",
        "capabilities": ["change_detection", "differencing_mask", "sector_statistics"],
        "modalities_supported": ["OPTICAL", "SAR"],
        "min_inputs": 2,
        "max_inputs": 2,
        "device_requirements": "cpu",
        "dependencies": ["cv2", "numpy"],
        "estimated_latency_ms": 35.0,
        "inputs": {
            "before_path": "str",
            "after_path": "str",
        },
        "outputs": {
            "change_detected": "bool",
            "percentage_changed": "float",
            "location_summary": "str",
            "dominant_sector": "str",
            "mask_path": "str",
            "overlay_path": "str",
        },
        "fallback_strategy": "radiometric_variance_check",
    },
    "change_vqa_specialist": {
        "id": "change_vqa_specialist",
        "name": "Bi-Temporal Change-VQA Reasoning Specialist",
        "description": "Synthesizes multi-temporal observations and differencing statistics to answer temporal queries.",
        "version": "1.0.0-temporal",
        "module": "models.change_vqa_model",
        "function": "answer_change_question",
        "status": "available",
        "capabilities": ["change_vqa", "temporal_reasoning"],
        "modalities_supported": ["OPTICAL", "SAR"],
        "min_inputs": 2,
        "max_inputs": 2,
        "device_requirements": "cpu",
        "dependencies": ["models.change_detection"],
        "estimated_latency_ms": 40.0,
        "inputs": {
            "before_path": "str",
            "after_path": "str",
            "question": "str",
            "change_map_result": "dict (from differencing_engine)",
        },
        "outputs": {
            "answer": "str",
            "confidence": "float or null",
            "change_metrics": "dict",
        },
        "fallback_strategy": "differencing_metric_summarizer",
    },
    "optical_sar_fusion_specialist": {
        "id": "optical_sar_fusion_specialist",
        "name": "Optical + SAR Dual-Branch Cross-Modal Fusion Specialist",
        "description": "Synthesizes multi-spectral optical reflectance with microwave radar backscatter for all-weather complementarity.",
        "version": "1.0.0-dual-branch",
        "module": "models.optical_sar_fusion",
        "function": "fuse_optical_and_sar",
        "status": "available",
        "capabilities": ["optical_sar_fusion", "cross_modal_synthesis", "all_weather_analysis"],
        "modalities_supported": ["OPTICAL", "SAR"],
        "min_inputs": 2,
        "max_inputs": 2,
        "device_requirements": "cpu",
        "dependencies": ["models.vqa_model"],
        "estimated_latency_ms": 60.0,
        "inputs": {
            "optical_path": "str",
            "sar_path": "str",
            "query": "str",
        },
        "outputs": {
            "answer": "str",
            "confidence": "float or null",
            "evidence": "dict",
            "complementary_gains": "list[str]",
        },
        "fallback_strategy": "optical_fallback_with_radar_notice",
    },
}

# Alias for backward compatibility
MODEL_REGISTRY = SPECIALIST_REGISTRY


def get_specialist(specialist_id: str) -> Optional[SpecialistMetadata]:
    """Retrieves typed metadata for a registered specialist."""
    raw = SPECIALIST_REGISTRY.get(specialist_id)
    if not raw:
        return None
    return SpecialistMetadata(
        id=raw["id"],
        name=raw["name"],
        version=raw["version"],
        status=ModelStatus(raw.get("status", "available")),
        capabilities=raw.get("capabilities", []),
        modalities_supported=raw.get("modalities_supported", []),
        min_inputs=raw.get("min_inputs", 1),
        max_inputs=raw.get("max_inputs", 1),
        device_requirements=raw.get("device_requirements", "cpu"),
        dependencies=raw.get("dependencies", []),
        estimated_latency_ms=raw.get("estimated_latency_ms", 30.0),
        fallback_strategy=raw.get("fallback_strategy"),
    )


def get_specialists_for_capability(capability: str) -> List[str]:
    """Returns all specialist IDs that advertise a given capability."""
    matches = []
    for s_id, s_info in SPECIALIST_REGISTRY.items():
        if capability in s_info.get("capabilities", []):
            matches.append(s_id)
    return matches


def is_specialist_available(specialist_id: str) -> bool:
    """Verifies that a specialist is registered and has status 'available'."""
    spec = SPECIALIST_REGISTRY.get(specialist_id)
    return spec is not None and spec.get("status") == "available"
