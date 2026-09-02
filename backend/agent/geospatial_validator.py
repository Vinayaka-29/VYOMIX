"""
Geospatial Compatibility Validator for SatQuery AI Central Brain (Phase 6)
Integrates rasterio CRS verification, spatial overlap calculations,
resolution scale checks, and multi-sensor alignment into the Central Brain.
"""
from typing import Dict, Any, List, Tuple
from validation.registration_checker import check_registration


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

    # 1. Multi-Temporal Pairwise Validation
    if pipeline_type == "multi_temporal":
        slot_b = validated_config["before_slot"]
        slot_a = validated_config["after_slot"]
        meta_b = manifest_files[slot_b].get("metadata", {})
        meta_a = manifest_files[slot_a].get("metadata", {})

        reg = check_registration(meta_b, meta_a, overlap_threshold=70.0)
        report["pairwise_metrics"]["temporal_pair"] = reg

        if not reg["is_co_registered"]:
            if reg["flag"] == "CRS_MISMATCH":
                return False, f"Geospatial Compatibility Error: {reg['warning']}", report
            elif reg["flag"] == "NO_OVERLAP":
                return False, "Geospatial Compatibility Error: Before and After rasters have zero geographical overlap.", report
            else:
                report["warnings"].append(reg["warning"])
                report["spatial_alignment_status"] = "MARGINAL_OVERLAP"

        # Check resolution compatibility
        res_b = meta_b.get("resolution", {}).get("x", 1.0)
        res_a = meta_a.get("resolution", {}).get("x", 1.0)
        if res_b and res_a:
            ratio = max(res_b, res_a) / min(res_b, res_a)
            if ratio > 3.0:
                report["warnings"].append(
                    f"Scale Discrepancy: Before resolution ({res_b}m) and After resolution ({res_a}m) differ by >3x. Differencing will resample."
                )

    # 2. Optical + SAR Cross-Modal Pairwise Validation
    elif pipeline_type == "cross_modal":
        slot_opt = validated_config["optical_slot"]
        slot_sar = validated_config["sar_slot"]
        meta_opt = manifest_files[slot_opt].get("metadata", {})
        meta_sar = manifest_files[slot_sar].get("metadata", {})

        reg = check_registration(meta_opt, meta_sar, overlap_threshold=70.0)
        report["pairwise_metrics"]["cross_modal_pair"] = reg

        if not reg["is_co_registered"]:
            if reg["flag"] == "CRS_MISMATCH":
                return False, f"Cross-Modal Compatibility Error: {reg['warning']}", report
            elif reg["flag"] == "NO_OVERLAP":
                return False, "Cross-Modal Compatibility Error: Optical and SAR images do not observe the same geographical footprint.", report
            else:
                report["warnings"].append(reg["warning"])
                report["spatial_alignment_status"] = "MARGINAL_OVERLAP"

    # 3. Single-Image Verification
    else:
        slot = validated_config.get("primary_slot", list(manifest_files.keys())[0])
        meta = manifest_files[slot].get("metadata", {})
        report["crs"] = meta.get("crs", "ungeoreferenced")
        report["resolution"] = meta.get("resolution")
        report["is_georeferenced"] = meta.get("is_georeferenced", False)

    return True, "Geospatial alignment verified.", report
