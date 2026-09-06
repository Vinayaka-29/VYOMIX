"""
Georeferenced Evidence Module for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Transforms pixel coordinates to geographic/projected coordinates
using rasterio geotransform metadata.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("satquery.agent.geo_evidence")

try:
    import rasterio
    from rasterio.transform import xy as rasterio_xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def pixel_to_geo(
    row: int,
    col: int,
    image_path: str,
) -> Optional[Tuple[float, float]]:
    """
    Converts a pixel (row, col) to geographic coordinates (x, y) using the image's geotransform.
    Returns None if the image is not georeferenced or rasterio is unavailable.
    """
    if not HAS_RASTERIO:
        return None
    try:
        with rasterio.open(image_path) as src:
            if src.crs is None:
                return None
            x, y = rasterio_xy(src.transform, row, col)
            return (round(x, 6), round(y, 6))
    except Exception as e:
        logger.warning(f"[GeoEvidence] Failed to convert pixel ({row},{col}) to geo: {e}")
        return None


def bbox_pixel_to_geo(
    pixel_bbox: List[int],
    image_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Converts a pixel bounding box [xmin, ymin, xmax, ymax] to geographic coordinates.
    Returns a dict with geographic bbox and CRS, or None if not georeferenced.
    """
    if not HAS_RASTERIO or not pixel_bbox or len(pixel_bbox) != 4:
        return None

    xmin_px, ymin_px, xmax_px, ymax_px = pixel_bbox

    try:
        with rasterio.open(image_path) as src:
            if src.crs is None:
                return None

            # Convert corner pixels to geographic coordinates
            # rasterio_xy(transform, row, col) — note row=y, col=x
            x_min, y_max = rasterio_xy(src.transform, ymin_px, xmin_px)
            x_max, y_min = rasterio_xy(src.transform, ymax_px, xmax_px)

            crs_str = src.crs.to_string()
            epsg = src.crs.to_epsg()

            return {
                "geographic_bbox": [
                    round(x_min, 6), round(y_min, 6),
                    round(x_max, 6), round(y_max, 6),
                ],
                "crs": crs_str,
                "epsg": epsg,
                "unit": "meters" if epsg and epsg != 4326 else "degrees" if epsg == 4326 else "unknown",
            }
    except Exception as e:
        logger.warning(f"[GeoEvidence] Failed to convert bbox to geo: {e}")
        return None


def enrich_grounding_with_geo(
    grounding_result: Dict[str, Any],
    image_path: str,
) -> Dict[str, Any]:
    """
    Enriches a grounding result dict with geographic coordinates if the image is georeferenced.
    Adds 'geographic_bbox' and 'geographic_crs' fields.
    If not georeferenced, adds a clear 'geographic_note'.
    """
    result = dict(grounding_result)
    pixel_bbox = result.get("bbox")

    if pixel_bbox and len(pixel_bbox) == 4:
        geo_info = bbox_pixel_to_geo(pixel_bbox, image_path)
        if geo_info:
            result["geographic_bbox"] = geo_info["geographic_bbox"]
            result["geographic_crs"] = geo_info["crs"]
            result["geographic_epsg"] = geo_info["epsg"]
            result["geographic_unit"] = geo_info["unit"]
        else:
            result["geographic_bbox"] = None
            result["geographic_note"] = (
                "Geographic coordinates cannot be derived: input image is not georeferenced "
                "or lacks a valid CRS/geotransform."
            )
    else:
        result["geographic_bbox"] = None
        result["geographic_note"] = "No bounding box available for geographic coordinate derivation."

    return result
