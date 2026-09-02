"""
Task Classification & Input Requirements Engine for SatQuery AI (Phase 5)
Evaluates query intent compatibility against staged rasters and detected modalities.
Prevents silent model failures by enforcing strict precondition checking and diagnostic feedback.
"""
from typing import Dict, Any, List, Tuple


def classify_and_validate_task(
    intent: Dict[str, Any], 
    manifest_files: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates query intent against available rasters and detected sensor modalities.
    Returns:
      (is_compatible: bool, diagnostic_message: str, validated_config: dict)
    """
    task = intent["task"]
    available_slots = list(manifest_files.keys())
    total_images = len(available_slots)

    if total_images == 0:
        return False, "No satellite imagery has been uploaded. Please upload at least one raster.", {}

    # Extract detected sensor modalities
    detected_modalities = {
        slot: info.get("modality", {}).get("modality", "UNKNOWN")
        for slot, info in manifest_files.items()
    }
    all_modalities = set(detected_modalities.values())

    # --- TASK 1: Bi-Temporal Change Detection & Change-VQA ---
    if task == "change_vqa" or intent.get("requires_multi_temporal"):
        if total_images < 2:
            return False, (
                f"Validation Error [E-TEMP-01]: The query '{intent['raw_query']}' specifies temporal change analysis, "
                f"which requires two observations over time. Only {total_images} image was provided. "
                f"Please upload both a 'Before Image (T₀)' and an 'After Image (T₁)'."
            ), {}

        # Resolve before and after slots
        slot_before = "before" if "before" in manifest_files else available_slots[0]
        slot_after = "after" if "after" in manifest_files else available_slots[1]

        return True, "Bi-temporal change task validated.", {
            "task": "change_vqa",
            "pipeline_type": "multi_temporal",
            "before_slot": slot_before,
            "after_slot": slot_after,
            "detected_modalities": detected_modalities,
            "target_entity": intent.get("target_entity", "surface_alteration"),
            "spatial_constraint": intent.get("spatial_constraint"),
        }

    # --- TASK 2: Optical + SAR Cross-Modal Fusion ---
    if task == "optical_sar_fusion":
        has_opt = "OPTICAL" in all_modalities or "MULTISPECTRAL" in all_modalities or "optical" in manifest_files
        has_sar = "SAR" in all_modalities or "sar" in manifest_files

        if total_images < 2:
            return False, (
                f"Validation Error [E-FUSE-01]: Optical + SAR cross-modal fusion requires two complementary sensors. "
                f"Currently only {total_images} raster is uploaded. "
                f"Please upload both Optical imagery (VIS/VNIR) and SAR radar imagery."
            ), {}

        if not (has_opt and has_sar):
            return False, (
                f"Validation Error [E-FUSE-02]: Cross-modal analysis requires complementary sensor modalities (1 Optical + 1 SAR). "
                f"Currently detected modalities: {detected_modalities}. "
                f"If uploading radar data, ensure it is staged in the SAR slot or flagged as SAR."
            ), {}

        # Resolve optical and sar slots
        slot_opt = "optical" if "optical" in manifest_files else next(s for s, m in detected_modalities.items() if m in ("OPTICAL", "MULTISPECTRAL"))
        slot_sar = "sar" if "sar" in manifest_files else next(s for s, m in detected_modalities.items() if m == "SAR")

        return True, "Optical + SAR cross-modal task validated.", {
            "task": "optical_sar_fusion",
            "pipeline_type": "cross_modal",
            "optical_slot": slot_opt,
            "sar_slot": slot_sar,
            "detected_modalities": detected_modalities,
            "target_entity": intent.get("target_entity", "cross_modal_synthesis"),
            "spatial_constraint": intent.get("spatial_constraint"),
        }

    # --- TASK 3: Text-Guided Grounding ---
    if task == "grounding":
        slot_primary = "optical" if "optical" in manifest_files else available_slots[0]
        return True, "Grounding task validated.", {
            "task": "grounding",
            "pipeline_type": "single_image_grounding",
            "primary_slot": slot_primary,
            "target_entity": intent.get("target_entity", "region_of_interest"),
            "spatial_constraint": intent.get("spatial_constraint"),
            "detected_modalities": detected_modalities,
        }

    # --- TASK 4: Dense Scene Captioning ---
    if task == "captioning":
        slot_primary = "optical" if "optical" in manifest_files else available_slots[0]
        return True, "Captioning task validated.", {
            "task": "captioning",
            "pipeline_type": "single_image_captioning",
            "primary_slot": slot_primary,
            "target_entity": "scene_level",
            "spatial_constraint": intent.get("spatial_constraint"),
            "detected_modalities": detected_modalities,
        }

    # --- TASK 5: Single-Image VQA ---
    slot_primary = "optical" if "optical" in manifest_files else available_slots[0]
    return True, "Single-image VQA task validated.", {
        "task": "single_image_vqa",
        "pipeline_type": "single_image_vqa",
        "primary_slot": slot_primary,
        "target_entity": intent.get("target_entity", "spectral_properties"),
        "spatial_constraint": intent.get("spatial_constraint"),
        "detected_modalities": detected_modalities,
    }


# Backwards-compatible alias for existing imports
validate_intent_against_inputs = classify_and_validate_task
