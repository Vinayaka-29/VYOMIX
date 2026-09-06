"""
Geospatial Measurements Module for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Computes real physical measurements from raster data:
- Region area (m², hectares)
- Changed area from change masks
- AOI percentage
- Bounding box dimensions

All measurements require valid resolution metadata. Never produces
measurements from arbitrary assumptions. Clearly labels estimates.
"""
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("satquery.agent.geo_measurements")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def compute_pixel_area(
    resolution_x: float,
    resolution_y: float,
    unit: str = "meters",
) -> Optional[float]:
    """
    Computes the physical area of a single pixel in m².
    Returns None if resolution unit is not meters.
    """
    if unit not in ("meters", "m"):
        return None
    return abs(resolution_x * resolution_y)


def compute_region_area(
    pixel_count: int,
    resolution_x: float,
    resolution_y: float,
    unit: str = "meters",
) -> Dict[str, Any]:
    """
    Computes physical area for a region given pixel count and resolution.
    Returns area in m² and hectares. Returns None values if resolution is in pixels/degrees.
    """
    pixel_area = compute_pixel_area(resolution_x, resolution_y, unit)
    if pixel_area is None:
        return {
            "pixel_count": pixel_count,
            "area_m2": None,
            "area_hectares": None,
            "note": f"Physical area cannot be computed: resolution unit is '{unit}', not meters.",
            "is_estimate": False,
        }

    area_m2 = pixel_count * pixel_area
    area_ha = area_m2 / 10000.0

    return {
        "pixel_count": pixel_count,
        "area_m2": round(area_m2, 2),
        "area_hectares": round(area_ha, 4),
        "resolution_m": [round(resolution_x, 4), round(resolution_y, 4)],
        "is_estimate": False,
    }


def compute_change_area(
    change_mask: np.ndarray,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes physical area of changed regions from a binary change mask
    using the resolution from extracted metadata.
    """
    changed_pixels = int(np.count_nonzero(change_mask))
    total_pixels = int(change_mask.size)
    pct_changed = round((changed_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0

    resolution = metadata.get("resolution", {})
    res_x = resolution.get("x", 1.0)
    res_y = resolution.get("y", 1.0)
    res_unit = resolution.get("unit", "pixels")

    area_info = compute_region_area(changed_pixels, res_x, res_y, res_unit)
    area_info["percentage_of_scene"] = pct_changed
    area_info["total_pixels"] = total_pixels

    return area_info


# Alias for backward compatibility
compute_change_measurements = compute_change_area


def compute_aoi_percentage(aoi_pixels: int, total_pixels: int) -> float:
    """Computes percentage of scene area occupied by an AOI or region."""
    if total_pixels <= 0:
        return 0.0
    return round((aoi_pixels / total_pixels) * 100.0, 2)


def compute_bbox_dimensions(
    pixel_bbox: list,
    resolution_x: float,
    resolution_y: float,
    unit: str = "meters",
) -> Dict[str, Any]:
    """
    Computes physical dimensions of a bounding box.
    pixel_bbox: [xmin, ymin, xmax, ymax]
    """
    if len(pixel_bbox) != 4:
        return {"error": "Invalid bbox format"}

    xmin, ymin, xmax, ymax = pixel_bbox
    width_px = xmax - xmin
    height_px = ymax - ymin

    result = {
        "width_px": width_px,
        "height_px": height_px,
    }

    pixel_area = compute_pixel_area(resolution_x, resolution_y, unit)
    if pixel_area is not None:
        result["width_m"] = round(width_px * abs(resolution_x), 2)
        result["height_m"] = round(height_px * abs(resolution_y), 2)
        result["area_m2"] = round(width_px * height_px * pixel_area, 2)
        result["area_hectares"] = round(result["area_m2"] / 10000.0, 4)
    else:
        result["width_m"] = None
        result["height_m"] = None
        result["area_m2"] = None
        result["note"] = f"Physical dimensions unavailable: resolution unit is '{unit}'."

    return result


def extract_measurements_from_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts resolution and measurement capabilities from an image file.
    Returns metadata needed for measurement computation.
    """
    if not HAS_RASTERIO:
        return {
            "can_measure": False,
            "note": "Rasterio not available for geospatial measurement extraction.",
        }

    try:
        with rasterio.open(image_path) as src:
            if src.crs is None:
                return {
                    "can_measure": False,
                    "note": "Image is not georeferenced. Physical measurements unavailable.",
                }

            res_x, res_y = src.res
            epsg = src.crs.to_epsg()
            unit = "meters" if epsg and epsg != 4326 else "degrees"

            return {
                "can_measure": unit == "meters",
                "resolution_x": float(res_x),
                "resolution_y": float(res_y),
                "unit": unit,
                "crs": src.crs.to_string(),
                "epsg": epsg,
                "image_width": src.width,
                "image_height": src.height,
                "note": None if unit == "meters" else "CRS is geographic (degrees). Area computation requires projected CRS.",
            }
    except Exception as e:
        return {
            "can_measure": False,
            "note": f"Failed to extract measurement metadata: {e}",
        }
