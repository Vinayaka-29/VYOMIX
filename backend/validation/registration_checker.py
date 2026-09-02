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
    Evaluates spatial co-registration between two rasters using their metadata.
    Returns:
      {
        "is_co_registered": bool,
        "overlap_percentage": float,
        "crs_match": bool,
        "flag": str,
        "warning": Optional[str],
        "details": dict
      }
    """
    crs1 = meta1.get("crs")
    crs2 = meta2.get("crs")
    bounds1 = meta1.get("bounds")
    bounds2 = meta2.get("bounds")

    # Case 1: Both are georeferenced
    if bounds1 and bounds2 and crs1 != "ungeoreferenced" and crs2 != "ungeoreferenced":
        crs_match = (crs1 == crs2) or (meta1.get("epsg") == meta2.get("epsg") and meta1.get("epsg") is not None)

        if not crs_match:
            return {
                "is_co_registered": False,
                "overlap_percentage": 0.0,
                "crs_match": False,
                "flag": "CRS_MISMATCH",
                "warning": f"Coordinate Reference Systems differ ({crs1} vs {crs2}). Re-projection required.",
                "details": {"crs1": crs1, "crs2": crs2},
            }

        # Calculate bounding box intersection
        b1 = bounds1["bbox_list"]  # [minx, miny, maxx, maxy]
        b2 = bounds2["bbox_list"]

        inter_min_x = max(b1[0], b2[0])
        inter_min_y = max(b1[1], b2[1])
        inter_max_x = min(b1[2], b2[2])
        inter_max_y = min(b1[3], b2[3])

        if inter_max_x > inter_min_x and inter_max_y > inter_min_y:
            inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union_area = area1 + area2 - inter_area

            # Overlap as IoU percentage
            overlap_pct = (inter_area / union_area * 100.0) if union_area > 0 else 0.0
            overlap_pct = round(overlap_pct, 2)
            is_aligned = overlap_pct >= overlap_threshold

            return {
                "is_co_registered": is_aligned,
                "overlap_percentage": overlap_pct,
                "crs_match": True,
                "flag": "CO_REGISTERED" if is_aligned else "LOW_SPATIAL_OVERLAP",
                "warning": None if is_aligned else f"Spatial overlap ({overlap_pct}%) is below the {overlap_threshold}% co-registration threshold.",
                "details": {
                    "intersection_bbox": [round(inter_min_x, 4), round(inter_min_y, 4), round(inter_max_x, 4), round(inter_max_y, 4)],
                    "threshold": overlap_threshold,
                },
            }
        else:
            return {
                "is_co_registered": False,
                "overlap_percentage": 0.0,
                "crs_match": True,
                "flag": "NO_OVERLAP",
                "warning": "The image bounding boxes do not overlap geographically.",
                "details": {"threshold": overlap_threshold},
            }

    # Case 2: Ungeoreferenced benchmark images (pixel space check)
    w1, h1 = meta1.get("width", 0), meta1.get("height", 0)
    w2, h2 = meta2.get("width", 0), meta2.get("height", 0)

    dim_ratio = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
    dim_match = (w1 == w2 and h1 == h2)

    return {
        "is_co_registered": dim_match or dim_ratio > 0.85,
        "overlap_percentage": 100.0 if dim_match else round(dim_ratio * 100.0, 1),
        "crs_match": False,
        "flag": "PIXEL_GRID_MATCH" if dim_match else "UNREFERENCED_DIM_CHECK",
        "warning": None if dim_match else "Images lack CRS georeferencing metadata; verified via pixel dimensions.",
        "details": {
            "dims_slot1": f"{w1}x{h1}",
            "dims_slot2": f"{w2}x{h2}",
            "is_ungeoreferenced": True,
        },
    }
