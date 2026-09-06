"""
Area of Interest (AOI) Module for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Supports AOI specification as rectangle, polygon, or image-relative coordinates.
Provides validation, normalization, and image cropping.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger("satquery.agent.aoi")


class AOISpec:
    """Validated Area of Interest specification."""

    def __init__(
        self,
        aoi_type: str,  # "rectangle" | "polygon"
        coordinates: List[float],
        coord_system: str = "pixel",  # "pixel" | "normalized" | "geographic"
        crs: Optional[str] = None,
    ):
        self.aoi_type = aoi_type
        self.coordinates = coordinates
        self.coord_system = coord_system
        self.crs = crs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aoi_type": self.aoi_type,
            "coordinates": self.coordinates,
            "coord_system": self.coord_system,
            "crs": self.crs,
        }


def parse_aoi(aoi_dict: Dict[str, Any]) -> Optional[AOISpec]:
    """
    Parses an AOI dict from the API request into a validated AOISpec.

    Expected formats:
      Rectangle: {"type": "rectangle", "bbox": [xmin, ymin, xmax, ymax], "coord_system": "pixel"|"normalized"}
      Polygon: {"type": "polygon", "points": [[x1,y1],[x2,y2],...], "coord_system": "pixel"|"normalized"}
    """
    if not aoi_dict:
        return None

    aoi_type = aoi_dict.get("type", "rectangle")
    coord_system = aoi_dict.get("coord_system", "pixel")

    if aoi_type == "rectangle":
        bbox = aoi_dict.get("bbox", [])
        if len(bbox) != 4:
            logger.warning(f"[AOI] Invalid rectangle bbox length: {len(bbox)}")
            return None
        try:
            coords = [float(c) for c in bbox]
        except (ValueError, TypeError):
            logger.warning(f"[AOI] Non-numeric bbox values: {bbox}")
            return None
        return AOISpec(aoi_type="rectangle", coordinates=coords, coord_system=coord_system)

    elif aoi_type == "polygon":
        points = aoi_dict.get("points", [])
        if len(points) < 3:
            logger.warning(f"[AOI] Polygon needs at least 3 points, got {len(points)}")
            return None
        flat_coords = []
        for pt in points:
            if len(pt) < 2:
                return None
            flat_coords.extend([float(pt[0]), float(pt[1])])
        return AOISpec(aoi_type="polygon", coordinates=flat_coords, coord_system=coord_system)

    logger.warning(f"[AOI] Unknown AOI type: {aoi_type}")
    return None


def normalize_aoi_to_pixels(
    aoi: AOISpec,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Converts AOI coordinates to pixel bbox [xmin, ymin, xmax, ymax].
    For polygons, returns the bounding box of the polygon.
    """
    if aoi.coord_system == "normalized":
        if aoi.aoi_type == "rectangle":
            xmin = int(aoi.coordinates[0] * image_width)
            ymin = int(aoi.coordinates[1] * image_height)
            xmax = int(aoi.coordinates[2] * image_width)
            ymax = int(aoi.coordinates[3] * image_height)
        else:
            xs = [aoi.coordinates[i] * image_width for i in range(0, len(aoi.coordinates), 2)]
            ys = [aoi.coordinates[i] * image_height for i in range(1, len(aoi.coordinates), 2)]
            xmin, xmax = int(min(xs)), int(max(xs))
            ymin, ymax = int(min(ys)), int(max(ys))
    else:
        if aoi.aoi_type == "rectangle":
            xmin, ymin, xmax, ymax = [int(c) for c in aoi.coordinates]
        else:
            xs = [int(aoi.coordinates[i]) for i in range(0, len(aoi.coordinates), 2)]
            ys = [int(aoi.coordinates[i]) for i in range(1, len(aoi.coordinates), 2)]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

    # Clamp to image bounds
    xmin = max(0, min(xmin, image_width - 1))
    ymin = max(0, min(ymin, image_height - 1))
    xmax = max(xmin + 1, min(xmax, image_width))
    ymax = max(ymin + 1, min(ymax, image_height))

    return xmin, ymin, xmax, ymax


def crop_to_aoi(image: Image.Image, aoi: AOISpec) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Crops a PIL image to the specified AOI.
    Returns (cropped_image, crop_info_dict).
    """
    w, h = image.size
    xmin, ymin, xmax, ymax = normalize_aoi_to_pixels(aoi, w, h)

    cropped = image.crop((xmin, ymin, xmax, ymax))

    crop_info = {
        "original_size": {"width": w, "height": h},
        "aoi_pixel_bbox": [xmin, ymin, xmax, ymax],
        "aoi_normalized_bbox": [
            round(xmin / w, 4),
            round(ymin / h, 4),
            round(xmax / w, 4),
            round(ymax / h, 4),
        ],
        "cropped_size": {"width": cropped.width, "height": cropped.height},
    }

    logger.info(f"[AOI] Cropped {w}x{h} -> {cropped.width}x{cropped.height} (bbox: {[xmin, ymin, xmax, ymax]})")
    return cropped, crop_info


def crop_file_to_aoi(image_path: str, aoi: AOISpec, output_path: str) -> Dict[str, Any]:
    """
    Crops an image file to AOI and saves the cropped version.
    Returns crop metadata.
    """
    img = Image.open(image_path)
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    cropped, crop_info = crop_to_aoi(img, aoi)
    cropped.save(output_path)
    crop_info["output_path"] = output_path
    return crop_info
