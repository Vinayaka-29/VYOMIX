"""
Task Classifier & Compatibility Verifier for SatQuery AI (Phase 8)
Validates interpreted query intent against actual staged rasters and detected sensor modalities.
Prevents silent failures by providing clear, diagnostic feedback to the user.
"""
from typing import Dict, Any, List, Tuple


def validate_intent_against_inputs(
    intent: Dict[str, Any], 
    manifest_files: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Cross-checks the parsed intent against uploaded slots and detected modalities.
    Returns:
      (is_compatible: bool, error_message: str, validated_config: dict)
    """
    task = intent["task"]
    available_slots = list(manifest_files.keys())
    total_images = len(available_slots)

    if total_images == 0:
        return False, "No satellite imagery has been uploaded. Please stage at least one raster.", {}

    # Extract detected modalities
    detected_modalities = {
        slot: info.get("modality", {}).get("modality", "UNKNOWN")
        for slot, info in manifest_files.items()
    }

    # 1. Bi-Temporal Change Detection Validation
    if task == "change_vqa" or intent.get("requires_multi_temporal"):
        has_temporal_slots = ("before" in manifest_files and "after" in manifest_files)
        if total_images < 2:
            return False, (
                f"Your query asks for bi-temporal change analysis ('{intent['raw_query']}'), "
                f"but only {total_images} image was provided. "
                f"Please upload both a 'Before Image (T₀)' and an 'After Image (T₁)'."
            ), {}
        
        # If 2 slots exist, map them to before/after if not already named
        slot_before = "before" if "before" in manifest_files else available_slots[0]
        slot_after = "after" if "after" in manifest_files else available_slots[1]

        return True, "Valid bi-temporal configuration.", {
            "task": "change_vqa",
            "before_slot": slot_before,
            "after_slot": slot_after,
            "detected_modalities": detected_modalities,
        }

    # 2. Optical + SAR Cross-Modal Fusion Validation
    if task == "optical_sar_fusion":
        has_opt_slot = "optical" in manifest_files
        has_sar_slot = "sar" in manifest_files

        # Check if we have at least one detected SAR and one detected OPTICAL
        modalities_present = set(detected_modalities.values())
        has_sar_modality = "SAR" in modalities_present or has_sar_slot
        has_opt_modality = "OPTICAL" in modalities_present or "MULTISPECTRAL" in modalities_present or has_opt_slot

        if total_images < 2:
            return False, (
                f"Cross-modal analysis requires both Optical and SAR sensors. "
                f"Currently only {total_images} raster is uploaded. "
                f"Please upload both Optical imagery (RGB/VNIR) and SAR imagery (Radar)."
            ), {}

        if not (has_sar_modality and has_opt_modality):
            return False, (
                f"Cross-modal query requires complementary sensors (1 Optical + 1 SAR). "
                f"Detected modalities for uploaded rasters: {detected_modalities}. "
                f"Please upload a genuine Synthetic Aperture Radar (SAR) raster in the SAR slot."
            ), {}

        slot_optical = "optical" if "optical" in manifest_files else next(s for s, m in detected_modalities.items() if m in ("OPTICAL", "MULTISPECTRAL"))
        slot_sar = "sar" if "sar" in manifest_files else next(s for s, m in detected_modalities.items() if m == "SAR")

        return True, "Valid optical+SAR cross-modal configuration.", {
            "task": "optical_sar_fusion",
            "optical_slot": slot_optical,
            "sar_slot": slot_sar,
            "detected_modalities": detected_modalities,
        }

    # 3. Grounding / Captioning / Single-Image VQA
    primary_slot = "optical" if "optical" in manifest_files else available_slots[0]
    return True, "Valid single-image configuration.", {
        "task": task,
        "primary_slot": primary_slot,
        "target_entity": intent.get("target_entity"),
        "detected_modalities": detected_modalities,
    }
