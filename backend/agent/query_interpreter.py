"""
Query Interpreter for SatQuery AI Agentic Controller (Phase 8)
Converts natural language geospatial queries into structured intent JSON.
"""
import re
from typing import Dict, Any, List


def interpret_query(query_text: str) -> Dict[str, Any]:
    """
    Parses a natural-language Earth Observation query into a structured execution intent.
    Returns:
      {
        "task": "captioning" | "grounding" | "change_vqa" | "optical_sar_fusion" | "single_image_vqa",
        "target_entity": str,
        "requires_modalities": List[str],
        "requires_multi_temporal": bool,
        "raw_query": str,
        "confidence": float
      }
    """
    q = query_text.lower().strip()

    # 1. Grounding / Spatial Highlighting Queries
    ground_triggers = ["highlight", "ground", "locate", "where is", "bounding box", "delineate", "find the", "draw a box"]
    for trigger in ground_triggers:
        if trigger in q:
            # Extract target entity
            target = q.split(trigger)[-1].replace("in this image", "").replace("in the satellite image", "").strip()
            target = re.sub(r"^[a-z\s]+(the|a|an)\s+", "", target)
            return {
                "task": "grounding",
                "target_entity": target or "target_object",
                "requires_modalities": ["OPTICAL"],
                "requires_multi_temporal": False,
                "raw_query": query_text,
                "confidence": 0.95,
            }

    # 2. Optical + SAR Cross-Modal Fusion Queries
    if ("optical" in q and "sar" in q) or ("radar" in q and "optical" in q) or ("together" in q and ("sensor" in q or "sar" in q)):
        return {
            "task": "optical_sar_fusion",
            "target_entity": "cross_modal_synthesis",
            "requires_modalities": ["OPTICAL", "SAR"],
            "requires_multi_temporal": False,
            "raw_query": query_text,
            "confidence": 0.96,
        }

    # 3. Bi-Temporal Change Detection Queries
    change_triggers = ["change", "changed", "before and after", "between these two", "increased", "decreased", "expansion", "reduction", "loss", "growth"]
    if any(k in q for k in change_triggers):
        target = "surface_alteration"
        if "built-up" in q or "urban" in q:
            target = "built_up_area"
        elif "water" in q:
            target = "hydrology"
        elif "forest" in q or "vegetation" in q:
            target = "vegetation_cover"

        return {
            "task": "change_vqa",
            "target_entity": target,
            "requires_modalities": ["OPTICAL"],
            "requires_multi_temporal": True,
            "raw_query": query_text,
            "confidence": 0.94,
        }

    # 4. Dense Captioning Queries
    caption_triggers = ["describe", "caption", "scene description", "overview of the image", "summarize"]
    if any(k in q for k in caption_triggers):
        return {
            "task": "captioning",
            "target_entity": "scene_level",
            "requires_modalities": ["OPTICAL"],
            "requires_multi_temporal": False,
            "raw_query": query_text,
            "confidence": 0.92,
        }

    # 5. Default: Single-Image VQA
    return {
        "task": "single_image_vqa",
        "target_entity": "spectral_properties",
        "requires_modalities": ["OPTICAL"],
        "requires_multi_temporal": False,
        "raw_query": query_text,
        "confidence": 0.90,
    }
