"""
Query Interpretation Engine for SatQuery AI Central Brain (Phase 4)
Parses natural-language Earth Observation queries into structured intent representations.
Extracts task type, target entities, spatial constraints, and temporal markers.
"""
import re
from typing import Dict, Any, List, Optional


def interpret_query(query_text: str) -> Dict[str, Any]:
    """
    Parses a natural-language geospatial query into a structured execution intent.
    Returns:
      {
        "task": "captioning" | "grounding" | "change_vqa" | "optical_sar_fusion" | "single_image_vqa",
        "target_entity": str,
        "spatial_constraint": Optional[str],
        "temporal_markers": List[str],
        "requires_modalities": List[str],
        "requires_multi_temporal": bool,
        "intent_confidence": float,
        "raw_query": str
      }
    """
    q_clean = query_text.strip()
    q_lower = q_clean.lower()

    # 1. Detect spatial sector constraints
    spatial_sectors = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "central"]
    detected_sector = None
    for s in spatial_sectors:
        if s in q_lower or f"{s}ern" in q_lower:
            detected_sector = f"{s}ern" if not s.endswith("ern") else s
            break

    # 2. Detect temporal markers
    temporal_keywords = ["before", "after", "between", "difference", "growth", "expansion", "reduction", "loss", "pre", "post", "t0", "t1"]
    detected_temporals = [w for w in temporal_keywords if w in q_lower]

    # --- ROUTING RULE 1: Optical + SAR Cross-Modal Fusion ---
    # Trigger: explicit mention of optical and SAR / radar, or complementary extraction
    is_cross_modal = (
        ("optical" in q_lower and ("sar" in q_lower or "radar" in q_lower))
        or ("both sensors" in q_lower or "together" in q_lower and ("sar" in q_lower or "radar" in q_lower or "optical" in q_lower))
        or ("complementary" in q_lower and ("sensor" in q_lower or "modal" in q_lower))
    )
    if is_cross_modal:
        return {
            "task": "optical_sar_fusion",
            "target_entity": "cross_modal_complementarity",
            "spatial_constraint": detected_sector,
            "temporal_markers": detected_temporals,
            "requires_modalities": ["OPTICAL", "SAR"],
            "requires_multi_temporal": False,
            "intent_confidence": 0.98,
            "raw_query": q_clean,
        }

    # --- ROUTING RULE 2: Bi-Temporal Change Detection & Change-VQA ---
    # Trigger: temporal comparison, difference, change detection, expansion, loss
    is_change = (
        any(k in q_lower for k in ["change", "changed", "before and after", "between two dates", "between these two", "increased", "decreased", "expansion", "reduction", "loss", "growth"])
        or (len(detected_temporals) >= 2)
    )
    if is_change:
        target = "surface_alteration"
        if "built-up" in q_lower or "urban" in q_lower:
            target = "built_up_area"
        elif "water" in q_lower:
            target = "hydrology"
        elif "forest" in q_lower or "vegetation" in q_lower or "crop" in q_lower:
            target = "vegetation_cover"

        return {
            "task": "change_vqa",
            "target_entity": target,
            "spatial_constraint": detected_sector,
            "temporal_markers": detected_temporals,
            "requires_modalities": ["OPTICAL"],
            "requires_multi_temporal": True,
            "intent_confidence": 0.96,
            "raw_query": q_clean,
        }

    # --- ROUTING RULE 3: Referring-Expression Grounding ---
    # Trigger: explicit requests to highlight, locate, draw bounding box, delineate
    ground_patterns = [
        r"(?:highlight|ground|locate|where is|delineate|find the|draw a box around|box the)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9_\s\-]+?)(?:\s+in\s+this|\s+in\s+the|\s+raster|\s+satellite|\s+scene|$|\.|\?)",
    ]
    for pattern in ground_patterns:
        match = re.search(pattern, q_lower)
        if match:
            target = match.group(1).strip()
            # Clean trailing noise words
            target = re.sub(r"\s+(image|satellite|raster|scene|photo)$", "", target).strip()
            return {
                "task": "grounding",
                "target_entity": target or "region_of_interest",
                "spatial_constraint": detected_sector,
                "temporal_markers": [],
                "requires_modalities": ["OPTICAL"],
                "requires_multi_temporal": False,
                "intent_confidence": 0.95,
                "raw_query": q_clean,
            }

    # Direct keyword fallback for grounding
    if any(k in q_lower for k in ["highlight", "ground", "bounding box", "delineate"]):
        target = "region_of_interest"
        for phrase in ["highlight", "ground", "bounding box", "delineate"]:
            if phrase in q_lower:
                parts = q_lower.split(phrase)
                if len(parts) > 1 and parts[1].strip():
                    target = parts[1].replace("in this image", "").replace("the", "").strip()
                    break

        return {
            "task": "grounding",
            "target_entity": target,
            "spatial_constraint": detected_sector,
            "temporal_markers": [],
            "requires_modalities": ["OPTICAL"],
            "requires_multi_temporal": False,
            "intent_confidence": 0.92,
            "raw_query": q_clean,
        }

    # --- ROUTING RULE 4: Dense Scene Captioning ---
    # Trigger: summarize, describe, provide overview, caption
    caption_triggers = ["describe", "caption", "scene description", "overview of the image", "summarize", "what is shown in this scene"]
    if any(k in q_lower for k in caption_triggers):
        return {
            "task": "captioning",
            "target_entity": "scene_level",
            "spatial_constraint": detected_sector,
            "temporal_markers": [],
            "requires_modalities": ["OPTICAL"],
            "requires_multi_temporal": False,
            "intent_confidence": 0.94,
            "raw_query": q_clean,
        }

    # --- ROUTING RULE 5: Default Single-Image VQA ---
    # Trigger: inquiries, questions, land cover queries
    return {
        "task": "single_image_vqa",
        "target_entity": "spectral_properties",
        "spatial_constraint": detected_sector,
        "temporal_markers": [],
        "requires_modalities": ["OPTICAL"],
        "requires_multi_temporal": False,
        "intent_confidence": 0.91,
        "raw_query": q_clean,
    }
