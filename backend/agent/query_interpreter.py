"""
Query Interpretation Engine for SatQuery AI Central Brain
SIH Problem Statement 26167 | Team Vyomix

Parses natural-language Earth Observation queries into structured intent representations.
Extracts task type, target entities, spatial constraints, temporal markers, sensor requirements,
and semantic signals without hardcoded or arbitrary confidence scores.
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from agent.schemas import QueryIntent, TaskType


# =========================================================================
# Spatial & Temporal Signal Patterns
# =========================================================================

SPATIAL_SECTORS = [
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "central", "northern", "southern", "eastern", "western",
    "northeastern", "northwestern", "southeastern", "southwestern"
]

TEMPORAL_PATTERNS = [
    r"\b(before|after|between|pre|post|t0|t1)\b",
    r"\b(difference|growth|expansion|reduction|loss|decrease|increase|alteration|modified)\b",
    r"\b(20\d\d)\b",  # Years e.g. 2024, 2026
    r"\b(over time|over the years|temporal|timeline|past)\b",
    r"\b(changed|changes|changing)\b",
]

# Modality indicator patterns
OPTICAL_KEYWORDS = ["optical", "rgb", "vis", "vnir", "multispectral", "color", "visual", "spectral", "sentinel-2"]
SAR_KEYWORDS = ["sar", "radar", "microwave", "backscatter", "dielectric", "sentinel-1", "all-weather", "roughness"]

# Grounding trigger patterns
GROUNDING_TRIGGERS = [
    r"(?:highlight|ground|locate|where is|where are|delineate|find the|draw a box around|box the|mark all|box all|detect the location of)\s+(?:the\s+|a\s+|an\s+|all\s+)?([a-zA-Z0-9_\s\-]+?)(?:\s+in\s+this|\s+in\s+the|\s+raster|\s+satellite|\s+scene|$|\.|\?)",
    r"\b(where are the [a-zA-Z0-9_\s\-]+)\b",
    r"\b(locate [a-zA-Z0-9_\s\-]+)\b",
    r"\b(bounding box)\b",
]

# Captioning trigger patterns
CAPTIONING_TRIGGERS = [
    r"\b(describe|caption|summarize|summary|overview of|what is shown in this scene|give me a caption|describe this scene|describe this satellite image)\b",
]

# Change trigger patterns
CHANGE_TRIGGERS = [
    r"\b(what changed|how has .* changed|compare .* and .*|difference between|expansion of|loss of|growth of|has .* increased|has .* decreased|deforestation|urban sprawl)\b",
    r"\b(between 20\d\d and 20\d\d)\b",
    r"\b(before and after)\b",
]

# Cross-modal trigger patterns
CROSS_MODAL_TRIGGERS = [
    r"\b(optical and sar|sar and optical|optical and radar|radar and optical)\b",
    r"\b(compare optical and sar|what does sar reveal|sar provide that optical does not|cross-modal|both sensors)\b",
]


def _extract_spatial_constraint(text_lower: str) -> Tuple[Optional[str], List[str]]:
    """Extracts cardinal or intermediate geographical sector constraints."""
    signals = []
    detected_sector = None
    for s in SPATIAL_SECTORS:
        pattern = rf"\b{s}\b"
        if re.search(pattern, text_lower):
            norm = s if s.endswith("ern") or s == "central" else f"{s}ern"
            detected_sector = norm
            signals.append(f"spatial:{norm}")
            break
    return detected_sector, signals


def _extract_temporal_signals(text_lower: str) -> Tuple[List[str], List[str]]:
    """Extracts temporal markers and references to multi-temporal comparisons."""
    markers = []
    signals = []
    for pat in TEMPORAL_PATTERNS:
        matches = re.findall(pat, text_lower)
        for m in matches:
            if isinstance(m, str) and m not in markers:
                markers.append(m)
                signals.append(f"temporal:{m}")
    return markers, signals


def _extract_target_entity(text_lower: str, task: str) -> Optional[str]:
    """Extracts the primary physical land-cover entity or feature mentioned."""
    for pattern in GROUNDING_TRIGGERS[:1]:
        match = re.search(pattern, text_lower)
        if match:
            target = match.group(1).strip()
            target = re.sub(r"\s+(image|satellite|raster|scene|photo|area|features?)$", "", target).strip()
            if target:
                return target

    # Common remote sensing entities
    entities = [
        "agricultural field", "agriculture", "crop", "farmland", "pasture",
        "forest", "woodland", "vegetation", "canopy",
        "water body", "river", "lake", "canal", "reservoir", "ocean", "wetland",
        "building", "buildings", "urban area", "built-up", "built up area", "infrastructure",
        "runway", "airport", "road", "highway", "bridge", "industrial unit"
    ]
    for e in entities:
        if re.search(rf"\b{e}\b", text_lower):
            return e.replace(" ", "_")

    return None


def parse_query_intent(query_text: str) -> QueryIntent:
    """
    Parses a natural-language geospatial query into a strongly typed QueryIntent object.
    Grounded in verifiable lexical and semantic signals with zero fabricated confidence values.
    """
    q_clean = query_text.strip()
    q_lower = q_clean.lower()
    signals: List[str] = []

    # 1. Spatial & Temporal Signals
    sector, spatial_signals = _extract_spatial_constraint(q_lower)
    signals.extend(spatial_signals)

    temporal_markers, temp_signals = _extract_temporal_signals(q_lower)
    signals.extend(temp_signals)

    # 2. Modality Signal Detection
    has_optical = any(re.search(rf"\b{k}\b", q_lower) for k in OPTICAL_KEYWORDS)
    has_sar = any(re.search(rf"\b{k}\b", q_lower) for k in SAR_KEYWORDS)

    if has_optical and has_sar:
        signals.append("modality:optical+sar")
    elif has_sar:
        signals.append("modality:sar")
    elif has_optical:
        signals.append("modality:optical")

    # ---------------------------------------------------------------------
    # Priority 1: Optical + SAR Cross-Modal Fusion
    # ---------------------------------------------------------------------
    is_cross_modal = (
        (has_optical and has_sar)
        or any(re.search(p, q_lower) for p in CROSS_MODAL_TRIGGERS)
        or ("complementary" in q_lower and ("sensor" in q_lower or "radar" in q_lower or "optical" in q_lower))
    )
    if is_cross_modal:
        signals.append("intent:optical_sar_fusion")
        return QueryIntent(
            task=TaskType.OPTICAL_SAR,
            target_entity=_extract_target_entity(q_lower, "optical_sar") or "cross_modal_complementarity",
            spatial_constraint=sector,
            temporal_markers=temporal_markers,
            requires_modalities=["OPTICAL", "SAR"],
            requires_multi_temporal=False,
            required_inputs=2,
            signals=signals,
            raw_query=q_clean,
            confidence=None,  # No fake confidence
        )

    # ---------------------------------------------------------------------
    # Priority 2: Bi-Temporal Change Detection & Change-VQA
    # ---------------------------------------------------------------------
    has_explicit_change_keyword = any(re.search(p, q_lower) for p in CHANGE_TRIGGERS)
    change_words = ["what changed", "between", "difference", "growth", "expansion", "loss", "pre-event", "post-event"]
    has_change_word = any(re.search(rf"\b{re.escape(w)}\b", q_lower) for w in change_words)
    is_multi_temporal = has_explicit_change_keyword or len(temporal_markers) >= 2 or has_change_word
    if is_multi_temporal:
        signals.append("intent:change_vqa")
        target = _extract_target_entity(q_lower, "change_vqa") or "surface_alteration"
        return QueryIntent(
            task=TaskType.CHANGE_VQA,
            target_entity=target,
            spatial_constraint=sector,
            temporal_markers=temporal_markers,
            requires_modalities=["OPTICAL"],
            requires_multi_temporal=True,
            required_inputs=2,
            signals=signals,
            raw_query=q_clean,
            confidence=None,
        )

    # ---------------------------------------------------------------------
    # Priority 3: Referring-Expression Grounding
    # ---------------------------------------------------------------------
    is_grounding = any(re.search(p, q_lower) for p in [
        r"\b(highlight|ground|locate|delineate|where is|where are|find the|draw a box|mark all|box the)\b",
        r"\b(bounding box)\b"
    ])
    if is_grounding:
        signals.append("intent:grounding")
        target = _extract_target_entity(q_lower, "grounding") or "region_of_interest"
        return QueryIntent(
            task=TaskType.GROUNDING,
            target_entity=target,
            spatial_constraint=sector,
            temporal_markers=[],
            requires_modalities=["OPTICAL"],
            requires_multi_temporal=False,
            required_inputs=1,
            signals=signals,
            raw_query=q_clean,
            confidence=None,
        )

    # ---------------------------------------------------------------------
    # Priority 4: Dense Scene Captioning
    # ---------------------------------------------------------------------
    is_captioning = any(re.search(p, q_lower) for p in CAPTIONING_TRIGGERS)
    if is_captioning:
        signals.append("intent:captioning")
        return QueryIntent(
            task=TaskType.CAPTIONING,
            target_entity="scene_level",
            spatial_constraint=sector,
            temporal_markers=[],
            requires_modalities=["OPTICAL"],
            requires_multi_temporal=False,
            required_inputs=1,
            signals=signals,
            raw_query=q_clean,
            confidence=None,
        )

    # ---------------------------------------------------------------------
    # Priority 5: Visual Question Answering (VQA)
    # ---------------------------------------------------------------------
    signals.append("intent:vqa")
    target = _extract_target_entity(q_lower, "vqa") or "spectral_properties"
    return QueryIntent(
        task=TaskType.VQA,
        target_entity=target,
        spatial_constraint=sector,
        temporal_markers=[],
        requires_modalities=["OPTICAL"],
        requires_multi_temporal=False,
        required_inputs=1,
        signals=signals,
        raw_query=q_clean,
        confidence=None,
    )


def interpret_query(query_text: str) -> Dict[str, Any]:
    """
    Backwards-compatible API wrapper returning a standard dictionary
    expected by existing pipeline stages and tests.
    """
    intent = parse_query_intent(query_text)
    
    # Task mapping to preserve exact legacy strings expected in older tests
    legacy_task_str = intent.task.value
    if intent.task == TaskType.OPTICAL_SAR:
        legacy_task_str = "optical_sar_fusion"
    elif intent.task == TaskType.VQA:
        legacy_task_str = "single_image_vqa"

    d = {
        "task": legacy_task_str,
        "target_entity": intent.target_entity,
        "spatial_constraint": intent.spatial_constraint,
        "temporal_markers": intent.temporal_markers,
        "requires_modalities": intent.requires_modalities,
        "requires_multi_temporal": intent.requires_multi_temporal,
        "required_inputs": intent.required_inputs,
        "signals": intent.signals,
        "raw_query": intent.raw_query,
        "confidence": None,  # Purged fake confidence numbers (0.98, 0.96, etc.)
    }
    return d
