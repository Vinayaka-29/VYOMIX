"""
Registration Checker for SatQuery AI
Verifies spatial co-registration between image pairs (Optical+SAR or Before+After).
Calculates geographic/pixel intersection overlap and flags un-registered rasters.
"""
from typing import Dict, Any, Tuple, Optional


def check_registration(
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    overlap_threshold: float = 70.0
) -> Dict[str, Any]:
    """
    Evaluates spatial compatibility between two rasters.

    Checks:
      - CRS compatibility
      - Geographic bounding-box overlap
      - Pixel resolution compatibility
      - Pixel transform/grid alignment when available
      - Image dimensions

    Important:
      Missing CRS does NOT imply co-registration.
      Matching image dimensions alone are not sufficient evidence
      of spatial registration.
    """

    crs1 = meta1.get("crs")
    crs2 = meta2.get("crs")

    bounds1 = meta1.get("bounds")
    bounds2 = meta2.get("bounds")

    # ---------------------------------------------------------
    # 1. Determine georeferencing status
    # ---------------------------------------------------------

    geo1 = bool(meta1.get("is_georeferenced")) and crs1 not in (
        None,
        "ungeoreferenced",
    )

    geo2 = bool(meta2.get("is_georeferenced")) and crs2 not in (
        None,
        "ungeoreferenced",
    )

    # ---------------------------------------------------------
    # 2. If one image is georeferenced and the other is not,
    #    they cannot be reliably registered.
    # ---------------------------------------------------------

    if geo1 != geo2:
        return {
            "is_co_registered": False,
            "overlap_percentage": 0.0,
            "crs_match": False,
            "flag": "GEOREFERENCING_MISMATCH",
            "warning": (
                "One raster is georeferenced while the other lacks "
                "valid georeferencing metadata. Spatial registration "
                "cannot be verified safely."
            ),
            "details": {
                "image1_georeferenced": geo1,
                "image2_georeferenced": geo2,
            },
        }

    # ---------------------------------------------------------
    # 3. Both images are georeferenced
    # ---------------------------------------------------------

    if geo1 and geo2:

        # ---- CRS check ----
        epsg1 = meta1.get("epsg")
        epsg2 = meta2.get("epsg")

        crs_match = (
            crs1 == crs2
            or (
                epsg1 is not None
                and epsg2 is not None
                and epsg1 == epsg2
            )
        )

        if not crs_match:
            return {
                "is_co_registered": False,
                "overlap_percentage": 0.0,
                "crs_match": False,
                "flag": "CRS_MISMATCH",
                "warning": (
                    f"Coordinate Reference Systems differ "
                    f"({crs1} vs {crs2}). Re-projection is required "
                    "before spatial comparison."
                ),
                "details": {
                    "crs1": crs1,
                    "crs2": crs2,
                    "epsg1": epsg1,
                    "epsg2": epsg2,
                },
            }

        # ---- Bounds check ----
        if not bounds1 or not bounds2:
            return {
                "is_co_registered": False,
                "overlap_percentage": 0.0,
                "crs_match": True,
                "flag": "MISSING_BOUNDS",
                "warning": (
                    "Both rasters have CRS information, but one or both "
                    "are missing spatial bounds. Registration cannot "
                    "be verified."
                ),
                "details": {
                    "bounds1_available": bool(bounds1),
                    "bounds2_available": bool(bounds2),
                },
            }

        b1 = bounds1.get("bbox_list")
        b2 = bounds2.get("bbox_list")

        if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
            return {
                "is_co_registered": False,
                "overlap_percentage": 0.0,
                "crs_match": True,
                "flag": "INVALID_BOUNDS",
                "warning": (
                    "Spatial bounds are missing or malformed. "
                    "Registration cannot be verified."
                ),
                "details": {},
            }

        # ---- Geographic intersection ----
        inter_min_x = max(b1[0], b2[0])
        inter_min_y = max(b1[1], b2[1])
        inter_max_x = min(b1[2], b2[2])
        inter_max_y = min(b1[3], b2[3])

        if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
            return {
                "is_co_registered": False,
                "overlap_percentage": 0.0,
                "crs_match": True,
                "flag": "NO_OVERLAP",
                "warning": (
                    "The image bounding boxes do not overlap "
                    "geographically."
                ),
                "details": {
                    "threshold": overlap_threshold,
                },
            }

        inter_area = (
            (inter_max_x - inter_min_x)
            * (inter_max_y - inter_min_y)
        )

        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

        union_area = area1 + area2 - inter_area

        overlap_pct = (
            inter_area / union_area * 100.0
            if union_area > 0
            else 0.0
        )

        overlap_pct = round(overlap_pct, 2)

        # -----------------------------------------------------
        # 4. Resolution compatibility
        # -----------------------------------------------------

        res1 = meta1.get("resolution") or {}
        res2 = meta2.get("resolution") or {}

        res1_x = res1.get("x")
        res1_y = res1.get("y")
        res2_x = res2.get("x")
        res2_y = res2.get("y")

        resolution_warning = None

        if all(
            value is not None
            for value in (res1_x, res1_y, res2_x, res2_y)
        ):
            if res1_x > 0 and res1_y > 0 and res2_x > 0 and res2_y > 0:

                ratio_x = max(res1_x, res2_x) / min(res1_x, res2_x)
                ratio_y = max(res1_y, res2_y) / min(res1_y, res2_y)

                if ratio_x > 3.0 or ratio_y > 3.0:
                    resolution_warning = (
                        f"Resolution differs significantly "
                        f"(image1: {res1_x}x{res1_y}, "
                        f"image2: {res2_x}x{res2_y})."
                    )

        # -----------------------------------------------------
        # 5. Pixel transform / grid comparison
        # -----------------------------------------------------

        transform1 = meta1.get("transform")
        transform2 = meta2.get("transform")

        transform_match = None

        if transform1 and transform2:
            if len(transform1) == 6 and len(transform2) == 6:

                transform_match = all(
                    abs(float(a) - float(b)) < 1e-6
                    for a, b in zip(transform1, transform2)
                )

        # -----------------------------------------------------
        # 6. Dimensions
        # -----------------------------------------------------

        width1 = meta1.get("width")
        height1 = meta1.get("height")

        width2 = meta2.get("width")
        height2 = meta2.get("height")

        dimensions_match = (
            width1 == width2
            and height1 == height2
            and width1 is not None
            and height1 is not None
        )

        # -----------------------------------------------------
        # 7. Final registration decision
        # -----------------------------------------------------

        overlap_ok = overlap_pct >= overlap_threshold

        # If transforms are available, use them as additional
        # evidence rather than ignoring them.
        if transform_match is False:
            alignment_warning = (
                "Images overlap geographically, but their pixel "
                "grids/transforms differ. Resampling or reprojection "
                "may be required."
            )
        else:
            alignment_warning = None

        is_aligned = overlap_ok and (
            transform_match is not False
        )

        warnings = []

        if resolution_warning:
            warnings.append(resolution_warning)

        if alignment_warning:
            warnings.append(alignment_warning)

        if not overlap_ok:
            warnings.append(
                f"Spatial overlap ({overlap_pct}%) is below the "
                f"{overlap_threshold}% threshold."
            )

        return {
            "is_co_registered": is_aligned,
            "overlap_percentage": overlap_pct,
            "crs_match": True,
            "flag": (
                "CO_REGISTERED"
                if is_aligned
                else "LOW_SPATIAL_OVERLAP"
                if not overlap_ok
                else "PIXEL_GRID_MISMATCH"
            ),
            "warning": "; ".join(warnings) if warnings else None,
            "details": {
                "intersection_bbox": [
                    round(inter_min_x, 4),
                    round(inter_min_y, 4),
                    round(inter_max_x, 4),
                    round(inter_max_y, 4),
                ],
                "threshold": overlap_threshold,
                "dimensions_match": dimensions_match,
                "transform_match": transform_match,
                "resolution_warning": resolution_warning,
                "resolution1": res1,
                "resolution2": res2,
            },
        }

    # ---------------------------------------------------------
    # 8. Neither image has valid georeferencing
    # ---------------------------------------------------------

    width1 = meta1.get("width", 0)
    height1 = meta1.get("height", 0)

    width2 = meta2.get("width", 0)
    height2 = meta2.get("height", 0)

    dimensions_match = (
        width1 > 0
        and height1 > 0
        and width2 > 0
        and height2 > 0
        and width1 == width2
        and height1 == height2
    )

    return {
        "is_co_registered": False,
        "overlap_percentage": 0.0,
        "crs_match": False,
        "flag": "MISSING_GEOREFERENCING",
        "warning": (
            "Images lack valid CRS/georeferencing metadata. "
            "Matching pixel dimensions alone cannot prove "
            "spatial co-registration."
        ),
        "details": {
            "dims_slot1": f"{width1}x{height1}",
            "dims_slot2": f"{width2}x{height2}",
            "dimensions_match": dimensions_match,
            "is_ungeoreferenced": True,
        },
    }