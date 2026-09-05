"""
Task Classification & Precondition Validation Engine for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Evaluates query intent against available staged rasters, band structures,
and detected sensor modalities. Enforces strict precondition contracts to prevent
silent specialist failures and returns structured diagnostic error codes.
"""
from typing import Dict, Any, List, Tuple, Union, Optional
from agent.schemas import QueryIntent, TaskType, ValidationResult, TaskRequirement


def validate_task_requirements(
    intent: Union[QueryIntent, Dict[str, Any]], 
    manifest_files: Dict[str, Any]
) -> Tuple[ValidationResult, Optional[TaskRequirement]]:
    """
    Validates intent preconditions against available rasters and detected sensor modalities.
    Returns:
      (ValidationResult, Optional[TaskRequirement])
    """
    if isinstance(intent, QueryIntent):
        intent_dict = intent.to_dict()
        raw_task = intent.task.value
    else:
        intent_dict = intent
        raw_task = intent.get("task", "vqa")

    available_slots = list(manifest_files.keys())
    total_images = len(available_slots)

    if total_images == 0:
        return ValidationResult(
            is_valid=False,
            code="NO_INPUT_IMAGES",
            message="No satellite imagery has been uploaded. Please upload at least one raster.",
            details={"total_images": 0}
        ), None

    # Extract detected sensor modalities
    detected_modalities: Dict[str, str] = {}
    for slot, info in manifest_files.items():
        if isinstance(info, dict):
            mod = info.get("modality", {})
            if isinstance(mod, dict):
                detected_modalities[slot] = mod.get("modality", "UNKNOWN")
            elif isinstance(mod, str):
                detected_modalities[slot] = mod
            else:
                detected_modalities[slot] = "UNKNOWN"
        else:
            detected_modalities[slot] = "UNKNOWN"

    all_modalities = set(detected_modalities.values())

    # Normalize task string
    task_normalized = raw_task
    if raw_task in ("single_image_vqa", "vqa"):
        task_type = TaskType.VQA
    elif raw_task in ("captioning", "dense_captioning"):
        task_type = TaskType.CAPTIONING
    elif raw_task in ("grounding", "referring_grounding"):
        task_type = TaskType.GROUNDING
    elif raw_task in ("change_vqa", "change_analysis", "bi_temporal_change"):
        task_type = TaskType.CHANGE_VQA
    elif raw_task in ("optical_sar_fusion", "optical_sar", "cross_modal_fusion"):
        task_type = TaskType.OPTICAL_SAR
    else:
        task_type = TaskType.VQA

    # --- TASK 1: Bi-Temporal Change Detection & Change-VQA ---
    if task_type in (TaskType.CHANGE_VQA, TaskType.CHANGE_ANALYSIS) or intent_dict.get("requires_multi_temporal"):
        if total_images < 2:
            return ValidationResult(
                is_valid=False,
                code="E-TEMP-01",
                message=(
                    f"Validation Error [E-TEMP-01]: The query '{intent_dict.get('raw_query', '')}' specifies temporal change analysis, "
                    f"which requires two observations over time. Only {total_images} image was provided. "
                    f"Please upload both a 'Before Image (T₀)' and an 'After Image (T₁)'."
                ),
                details={"total_images": total_images, "required_images": 2}
            ), None

        slot_before = "before" if "before" in manifest_files else available_slots[0]
        slot_after = "after" if "after" in manifest_files else available_slots[1]

        req = TaskRequirement(
            task=TaskType.CHANGE_VQA,
            pipeline_type="multi_temporal",
            min_inputs=2,
            max_inputs=2,
            required_modalities=["OPTICAL"],
            temporal_pair_required=True,
            slots_assigned={"before_slot": slot_before, "after_slot": slot_after},
            target_entity=intent_dict.get("target_entity", "surface_alteration"),
            spatial_constraint=intent_dict.get("spatial_constraint"),
            detected_modalities=detected_modalities,
        )
        return ValidationResult(
            is_valid=True,
            code="VALIDATED_MULTI_TEMPORAL",
            message="Bi-temporal change task validated.",
            details={"slots": [slot_before, slot_after]}
        ), req

    # --- TASK 2: Optical + SAR Cross-Modal Fusion ---
    if task_type == TaskType.OPTICAL_SAR:
        has_opt = any(m in ("OPTICAL", "MULTISPECTRAL") for m in all_modalities) or "optical" in manifest_files
        has_sar = "SAR" in all_modalities or "sar" in manifest_files

        if total_images < 2:
            return ValidationResult(
                is_valid=False,
                code="E-FUSE-01",
                message=(
                    f"Validation Error [E-FUSE-01]: Optical + SAR cross-modal fusion requires two complementary sensors. "
                    f"Currently only {total_images} raster is uploaded. "
                    f"Please upload both Optical imagery (VIS/VNIR) and SAR radar imagery."
                ),
                details={"total_images": total_images, "required_images": 2}
            ), None

        if not (has_opt and has_sar):
            return ValidationResult(
                is_valid=False,
                code="E-FUSE-02",
                message=(
                    f"Validation Error [E-FUSE-02]: Cross-modal analysis requires complementary sensor modalities (1 Optical + 1 SAR). "
                    f"Currently detected modalities: {detected_modalities}. "
                    f"If uploading radar data, ensure it is staged in the SAR slot or flagged as SAR."
                ),
                details={"detected_modalities": detected_modalities}
            ), None

        slot_opt = "optical" if "optical" in manifest_files else next(
            s for s, m in detected_modalities.items() if m in ("OPTICAL", "MULTISPECTRAL")
        )
        slot_sar = "sar" if "sar" in manifest_files else next(
            s for s, m in detected_modalities.items() if m == "SAR"
        )

        req = TaskRequirement(
            task=TaskType.OPTICAL_SAR,
            pipeline_type="cross_modal",
            min_inputs=2,
            max_inputs=2,
            required_modalities=["OPTICAL", "SAR"],
            temporal_pair_required=False,
            slots_assigned={"optical_slot": slot_opt, "sar_slot": slot_sar},
            target_entity=intent_dict.get("target_entity", "cross_modal_synthesis"),
            spatial_constraint=intent_dict.get("spatial_constraint"),
            detected_modalities=detected_modalities,
        )
        return ValidationResult(
            is_valid=True,
            code="VALIDATED_CROSS_MODAL",
            message="Optical + SAR cross-modal task validated.",
            details={"optical_slot": slot_opt, "sar_slot": slot_sar}
        ), req

    # --- TASK 3: Text-Guided Grounding ---
    if task_type == TaskType.GROUNDING:
        slot_primary = "optical" if "optical" in manifest_files else available_slots[0]
        req = TaskRequirement(
            task=TaskType.GROUNDING,
            pipeline_type="single_image_grounding",
            min_inputs=1,
            max_inputs=1,
            required_modalities=["OPTICAL"],
            slots_assigned={"primary_slot": slot_primary},
            target_entity=intent_dict.get("target_entity", "region_of_interest"),
            spatial_constraint=intent_dict.get("spatial_constraint"),
            detected_modalities=detected_modalities,
        )
        return ValidationResult(
            is_valid=True,
            code="VALIDATED_GROUNDING",
            message="Grounding task validated.",
            details={"primary_slot": slot_primary}
        ), req

    # --- TASK 4: Dense Scene Captioning ---
    if task_type == TaskType.CAPTIONING:
        slot_primary = "optical" if "optical" in manifest_files else available_slots[0]
        req = TaskRequirement(
            task=TaskType.CAPTIONING,
            pipeline_type="single_image_captioning",
            min_inputs=1,
            max_inputs=1,
            required_modalities=["OPTICAL"],
            slots_assigned={"primary_slot": slot_primary},
            target_entity="scene_level",
            spatial_constraint=intent_dict.get("spatial_constraint"),
            detected_modalities=detected_modalities,
        )
        return ValidationResult(
            is_valid=True,
            code="VALIDATED_CAPTIONING",
            message="Captioning task validated.",
            details={"primary_slot": slot_primary}
        ), req

    # --- TASK 5: Single-Image VQA ---
    slot_primary = "optical" if "optical" in manifest_files else available_slots[0]
    req = TaskRequirement(
        task=TaskType.VQA,
        pipeline_type="single_image_vqa",
        min_inputs=1,
        max_inputs=1,
        required_modalities=["OPTICAL"],
        slots_assigned={"primary_slot": slot_primary},
        target_entity=intent_dict.get("target_entity", "spectral_properties"),
        spatial_constraint=intent_dict.get("spatial_constraint"),
        detected_modalities=detected_modalities,
    )
    return ValidationResult(
        is_valid=True,
        code="VALIDATED_VQA",
        message="Single-image VQA task validated.",
        details={"primary_slot": slot_primary}
    ), req


def classify_and_validate_task(
    intent: Union[QueryIntent, Dict[str, Any]], 
    manifest_files: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Backwards-compatible interface returning (is_compatible, message, validated_config_dict).
    """
    val_res, req = validate_task_requirements(intent, manifest_files)
    if not val_res.is_valid or req is None:
        return False, val_res.message, {}

    # Build validated config dictionary matching existing format
    legacy_task = req.task.value
    if req.task == TaskType.OPTICAL_SAR:
        legacy_task = "optical_sar_fusion"
    elif req.task == TaskType.VQA:
        legacy_task = "single_image_vqa"

    cfg: Dict[str, Any] = {
        "task": legacy_task,
        "pipeline_type": req.pipeline_type,
        "target_entity": req.target_entity,
        "spatial_constraint": req.spatial_constraint,
        "detected_modalities": req.detected_modalities,
    }
    cfg.update(req.slots_assigned)

    return True, val_res.message, cfg


# Backwards-compatible alias
validate_intent_against_inputs = classify_and_validate_task
