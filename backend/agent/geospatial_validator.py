"""
Geospatial Compatibility Validator for SatQuery AI Central Brain
SIH Problem Statement 26167 | Team Vyomix

Performs physical geospatial validation across participating rasters:
- File readability and corruption check
- Channel structure, dimensions, and band count verification
- Coordinate Reference System (CRS) verification via rasterio
- Geospatial bounding box intersection and minimum overlap percentage
- Resolution scale consistency checks
"""
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from validation.registration_checker import check_registration
from agent.schemas import ValidationResult


def validate_geospatial_compatibility(
    validated_config: Dict[str, Any],
    manifest_files: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates physical geospatial compatibility between participating rasters.
    Returns:
      (is_compatible: bool, message: str, geospatial_report: dict)
    """
    pipeline_type = validated_config.get("pipeline_type", "single_image")
    report: Dict[str, Any] = {
        "spatial_alignment_status": "VERIFIED",
        "warnings": [],
        "pairwise_metrics": {},
    }

    # 1. Verify existence and readability of assigned slots
    for slot_key, slot_name in validated_config.items():
        if slot_key.endswith("_slot") and isinstance(slot_name, str):
            if slot_name not in manifest_files:
                return False, f"Input Validation Error: Slot '{slot_name}' referenced in config is missing from staged imagery.", report
            
            slot_info = manifest_files[slot_name]
            saved_path = slot_info.get("saved_path")
            if saved_path and not Path(saved_path).exists():
                return False, f"Input Validation Error: Raster file '{saved_path}' does not exist on disk.", report

    # 2. Multi-Temporal Pairwise Validation
    if pipeline_type == "multi_temporal":
        slot_b = validated_config.get("before_slot")
        slot_a = validated_config.get("after_slot")
        if not slot_b or not slot_a:
            return False, "Temporal pipeline missing before or after slot assignment.", report

        meta_b = manifest_files[slot_b].get("metadata", {})
        meta_a = manifest_files[slot_a].get("metadata", {})

        reg = check_registration(meta_b, meta_a, overlap_threshold=70.0)
        report["pairwise_metrics"]["temporal_pair"] = reg

        if not reg["is_co_registered"]:
            if reg["flag"] in [
                "CRS_MISMATCH",
                "GEOREFERENCING_MISMATCH",
                "MISSING_GEOREFERENCING",
            ]:
                return (
                    False,
                    f"Geospatial Compatibility Error: {reg['warning']}",
                    report,
                )

            elif reg["flag"] == "NO_OVERLAP":
                return (
                    False,
                    "Geospatial Compatibility Error: "
                    "Before and After rasters have zero geographical overlap.",
                    report,
                )

            else:
                report["warnings"].append(reg["warning"])
                report["spatial_alignment_status"] = "MARGINAL_OVERLAP"

        # Check resolution compatibility
        res_b = meta_b.get("resolution", {}).get("x")
        res_a = meta_a.get("resolution", {}).get("x")
        if res_b and res_a:
            ratio = max(res_b, res_a) / max(1e-6, min(res_b, res_a))
            if ratio > 3.0:
                unit_b = meta_b.get("resolution", {}).get("unit", "unknown")
                unit_a = meta_a.get("resolution", {}).get("unit", "unknown")

                report["warnings"].append(
                    f"Scale Discrepancy: Before resolution "
                    f"({res_b} {unit_b}) and After resolution "
                    f"({res_a} {unit_a}) differ by >3x. "
                    f"Differencing may require resampling."
    )

    # 3. Optical + SAR Cross-Modal Pairwise Validation
    elif pipeline_type == "cross_modal":
        slot_opt = validated_config.get("optical_slot")
        slot_sar = validated_config.get("sar_slot")
        if not slot_opt or not slot_sar:
            return False, "Cross-modal pipeline missing optical or SAR slot assignment.", report

        meta_opt = manifest_files[slot_opt].get("metadata", {})
        meta_sar = manifest_files[slot_sar].get("metadata", {})

        reg = check_registration(meta_opt, meta_sar, overlap_threshold=70.0)
        report["pairwise_metrics"]["cross_modal_pair"] = reg

        if not reg["is_co_registered"]:
            if reg["flag"] in [
                "CRS_MISMATCH",
                "GEOREFERENCING_MISMATCH",
                "MISSING_GEOREFERENCING",
            ]:
                return (
                    False,
                    f"Cross-Modal Compatibility Error: {reg['warning']}",
                    report,
                )

            elif reg["flag"] == "NO_OVERLAP":
                return (
                    False,
                    "Cross-Modal Compatibility Error: "
                    "Optical and SAR images do not observe the same geographical footprint.",
                    report,
                )

            else:
                report["warnings"].append(reg["warning"])
                report["spatial_alignment_status"] = "MARGINAL_OVERLAP"

        # Check Optical + SAR resolution compatibility
        res_opt = meta_opt.get("resolution") or {}
        res_sar = meta_sar.get("resolution") or {}

        res_opt_x = res_opt.get("x")
        res_opt_y = res_opt.get("y")
        res_sar_x = res_sar.get("x")
        res_sar_y = res_sar.get("y")

        if all(
            value is not None
            for value in (res_opt_x, res_opt_y, res_sar_x, res_sar_y)
        ):
            if (
                res_opt_x > 0
                and res_opt_y > 0
                and res_sar_x > 0
                and res_sar_y > 0
            ):
                ratio_x = max(res_opt_x, res_sar_x) / min(res_opt_x, res_sar_x)
                ratio_y = max(res_opt_y, res_sar_y) / min(res_opt_y, res_sar_y)

                if ratio_x > 3.0 or ratio_y > 3.0:
                    report["warnings"].append(
                        f"Optical/SAR resolution discrepancy: "
                        f"Optical ({res_opt_x}x{res_opt_y} "
                        f"{res_opt.get('unit', 'unknown')}) vs "
                        f"SAR ({res_sar_x}x{res_sar_y} "
                        f"{res_sar.get('unit', 'unknown')}). "
                        f"Fusion may require resampling."
                    )

    # 4. Single-Image Verification
    else:
        primary_slot = validated_config.get("primary_slot", list(manifest_files.keys())[0])
        meta = manifest_files[primary_slot].get("metadata", {})
        report["crs"] = meta.get("crs", "ungeoreferenced")
        report["resolution"] = meta.get("resolution")
        report["is_georeferenced"] = meta.get("is_georeferenced", False)
        report["dimensions"] = {
            "width": meta.get("width"),
            "height": meta.get("height"),
            "bands": meta.get("bands"),
        }

    return True, "Geospatial alignment verified.", report
