"""
Specialist Model Registry for SatQuery AI Central Brain (Phase 7)
Catalogues the contracts, inputs, outputs, capabilities, and latency profiles
of all five specialist model pipelines.
"""
from typing import Dict, Any, List

SPECIALIST_REGISTRY: Dict[str, Dict[str, Any]] = {
    "vqa_specialist": {
        "id": "vqa_specialist",
        "name": "SatQuery Single-Image VQA Specialist",
        "description": "Performs remote-sensing visual question answering using a LoRA-adapted vision-language backbone.",
        "version": "1.2.0-adapted",
        "module": "models.vqa_model",
        "function": "answer_question",
        "inputs": {
            "image_path": "str (path to valid GeoTIFF or benchmark image)",
            "question": "str (natural-language inquiry)",
        },
        "outputs": {
            "answer": "str",
            "confidence": "float (0.0-1.0)",
            "latency_ms": "float",
            "details": "dict",
        },
        "modalities_supported": ["OPTICAL", "MULTISPECTRAL", "SAR"],
        "min_inputs": 1,
        "max_inputs": 1,
        "estimated_latency_ms": 35.0,
        "fallback_strategy": "heuristic_spectral_rules",
    },
    "captioning_specialist": {
        "id": "captioning_specialist",
        "name": "SatQuery Dense Scene Captioner",
        "description": "Produces comprehensive, land-cover informed scene descriptions covering natural and built environments.",
        "version": "1.1.0-adapted",
        "module": "models.captioning_model",
        "function": "generate_caption",
        "inputs": {
            "image_path": "str",
        },
        "outputs": {
            "caption": "str",
            "confidence": "float",
            "latency_ms": "float",
            "features_detected": "list",
        },
        "modalities_supported": ["OPTICAL", "MULTISPECTRAL", "SAR"],
        "min_inputs": 1,
        "max_inputs": 1,
        "estimated_latency_ms": 30.0,
        "fallback_strategy": "spectral_inventory_summary",
    },
    "grounding_specialist": {
        "id": "grounding_specialist",
        "name": "SatQuery Referring-Expression Grounding Engine",
        "description": "Localizes natural-language referring expressions into standardized pixel bounding boxes.",
        "version": "1.3.0-grounding",
        "module": "models.grounding_model",
        "function": "ground_expression",
        "inputs": {
            "image_path": "str",
            "expression": "str",
        },
        "outputs": {
            "found": "bool",
            "bbox": "list[int] [xmin, ymin, xmax, ymax]",
            "normalized_bbox": "list[float] [0.0-1.0]",
            "confidence": "float",
            "message": "str",
        },
        "modalities_supported": ["OPTICAL", "MULTISPECTRAL"],
        "min_inputs": 1,
        "max_inputs": 1,
        "estimated_latency_ms": 25.0,
        "fallback_strategy": "graceful_not_found_rejection",
    },
    "differencing_engine": {
        "id": "differencing_engine",
        "name": "SatQuery Computer-Vision Differencing Engine",
        "description": "Executes classical pixel-level differencing, adaptive thresholding, and morphological filtering on temporal pairs.",
        "version": "1.0.0-classical-cv",
        "module": "models.change_detection",
        "function": "compute_change_map",
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
        "modalities_supported": ["OPTICAL", "SAR"],
        "min_inputs": 2,
        "max_inputs": 2,
        "estimated_latency_ms": 40.0,
        "fallback_strategy": "radiometric_variance_check",
    },
    "change_vqa_specialist": {
        "id": "change_vqa_specialist",
        "name": "SatQuery Bi-Temporal Change-VQA Specialist",
        "description": "Synthesizes multi-temporal observations and differencing statistics to reason about temporal changes.",
        "version": "1.1.0-temporal-vlm",
        "module": "models.change_vqa_model",
        "function": "answer_change_question",
        "inputs": {
            "before_path": "str",
            "after_path": "str",
            "question": "str",
            "change_map_result": "dict (from differencing_engine)",
        },
        "outputs": {
            "answer": "str",
            "confidence": "float",
            "change_metrics": "dict",
        },
        "modalities_supported": ["OPTICAL", "SAR"],
        "min_inputs": 2,
        "max_inputs": 2,
        "estimated_latency_ms": 45.0,
        "fallback_strategy": "differencing_metric_summarizer",
    },
    "optical_sar_fusion_specialist": {
        "id": "optical_sar_fusion_specialist",
        "name": "SatQuery Optical + SAR Dual-Branch Fusion Specialist",
        "description": "Synthesizes multi-spectral optical reflectance with microwave radar backscatter for all-weather complementarity.",
        "version": "1.2.0-dual-branch",
        "module": "models.optical_sar_fusion",
        "function": "fuse_optical_and_sar",
        "inputs": {
            "optical_path": "str",
            "sar_path": "str",
            "query": "str",
        },
        "outputs": {
            "answer": "str",
            "confidence": "float",
            "evidence": "dict (optical & sar branch outputs)",
            "complementary_gains": "list[str]",
        },
        "modalities_supported": ["OPTICAL", "SAR"],
        "min_inputs": 2,
        "max_inputs": 2,
        "estimated_latency_ms": 50.0,
        "fallback_strategy": "optical_fallback_with_radar_notice",
    },
}

# Alias for backward compatibility
MODEL_REGISTRY = SPECIALIST_REGISTRY
