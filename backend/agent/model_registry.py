"""
Model Registry for SatQuery AI Agentic Controller (Phase 8)
Catalogues available specialist models, inputs, parameters, and capabilities.
"""
from typing import Dict, Any

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "vqa_model": {
        "name": "SatQuery Single-Image VQA Specialist",
        "module": "models.vqa_model",
        "function": "answer_question",
        "version": "1.0.0-lora-adapted",
        "inputs": ["image_path", "question"],
        "capabilities": ["land_cover_query", "object_counting", "spectral_inquiry", "terrain_characterization"],
        "modalities_supported": ["OPTICAL", "MULTISPECTRAL", "SAR"],
    },
    "captioning_model": {
        "name": "SatQuery Dense Scene Captioner",
        "module": "models.captioning_model",
        "function": "generate_caption",
        "version": "1.0.0-lora-adapted",
        "inputs": ["image_path"],
        "capabilities": ["dense_scene_description", "environmental_survey", "land_use_summary"],
        "modalities_supported": ["OPTICAL", "MULTISPECTRAL", "SAR"],
    },
    "grounding_model": {
        "name": "SatQuery Referring-Expression Grounding Engine",
        "module": "models.grounding_model",
        "function": "ground_expression",
        "version": "1.0.0-lora-adapted",
        "inputs": ["image_path", "expression"],
        "capabilities": ["spatial_localization", "bounding_box_generation", "entity_grounding"],
        "modalities_supported": ["OPTICAL", "MULTISPECTRAL"],
    },
    "change_detection": {
        "name": "SatQuery Computer-Vision Differencing Engine",
        "module": "models.change_detection",
        "function": "compute_change_map",
        "version": "1.0.0-classical-cv",
        "inputs": ["before_path", "after_path"],
        "capabilities": ["pixel_differencing", "change_mask_generation", "spatial_quadrant_analysis"],
        "modalities_supported": ["OPTICAL", "SAR"],
    },
    "change_vqa_model": {
        "name": "SatQuery Bi-Temporal Change-VQA Specialist",
        "module": "models.change_vqa_model",
        "function": "answer_change_question",
        "version": "1.0.0-temporal-vlm",
        "inputs": ["before_path", "after_path", "question", "change_map_result"],
        "capabilities": ["temporal_expansion_reasoning", "loss_analysis", "urban_sprawl_inquiry"],
        "modalities_supported": ["OPTICAL", "SAR"],
    },
    "optical_sar_fusion": {
        "name": "SatQuery Cross-Modal Dual-Branch Fusion Specialist",
        "module": "models.optical_sar_fusion",
        "function": "fuse_optical_and_sar",
        "version": "1.0.0-dual-branch-llm",
        "inputs": ["optical_path", "sar_path", "query"],
        "capabilities": ["cross_modal_synthesis", "all_weather_validation", "complementary_extraction"],
        "modalities_supported": ["OPTICAL", "SAR"],
    },
}
