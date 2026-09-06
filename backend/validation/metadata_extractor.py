"""
Metadata Extractor for SatQuery AI
Uses rasterio (with fallback to PIL/tifffile) to extract spatial CRS,
spatial resolution (GSD), bounding box, band count, data types,
band descriptions, NoData values, pixel transform, tags, and
acquisition dates.

Gracefully handles ungeoreferenced rasters/images without inventing CRS.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def _extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Attempts to extract acquisition date from common remote sensing
    naming conventions.

    Examples:
      S2A_MSIL2A_20240315T103021
      Landsat LC08_..._20230512
      simple YYYYMMDD
    """
    patterns = [
        r"(20\d{2}[01]\d[0-3]\d)T\d{6}",
        r"(20\d{2}[-_][01]\d[-_][0-3]\d)",
        r"(20\d{2}[01]\d[0-3]\d)",
    ]

    for pat in patterns:
        m = re.search(pat, filename)

        if m:
            raw = m.group(1).replace("_", "-")

            if len(raw) == 8 and raw.isdigit():
                return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

            return raw

    return None


def _serialize_transform(transform) -> Optional[list]:
    """
    Converts Rasterio's Affine transform into a JSON-friendly list.

    Format:
      [a, b, c, d, e, f]

    This represents the pixel-to-coordinate transformation:
      x = a * col + b * row + c
      y = d * col + e * row + f
    """
    if transform is None:
        return None

    return [
        round(float(transform.a), 10),
        round(float(transform.b), 10),
        round(float(transform.c), 10),
        round(float(transform.d), 10),
        round(float(transform.e), 10),
        round(float(transform.f), 10),
    ]


def _get_resolution_unit(src, epsg_code: Optional[int]) -> str:
    """
    Determines the coordinate unit used by the raster resolution.

    Uses CRS information when available and avoids assuming that
    every CRS uses metres.
    """
    try:
        if src.crs:
            # Rasterio/PROJ exposes the linear or angular unit here.
            units_factor = src.crs.linear_units_factor

            if units_factor:
                unit_name = src.crs.linear_units

                if unit_name:
                    unit_name = unit_name.lower()

                    if "metre" in unit_name or "meter" in unit_name:
                        return "meters"

                    if "foot" in unit_name or "feet" in unit_name:
                        return "feet"

                    return unit_name

            # Geographic CRS such as EPSG:4326
            if src.crs.is_geographic:
                return "degrees"

    except Exception:
        pass

    return "unknown"


def extract_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extracts standardized geospatial and image metadata.

    Important:
    - CRS is never invented.
    - Missing CRS is represented as ungeoreferenced.
    - GeoTIFF metadata is preserved where available.
    - Rasterio failures are recorded before falling back to PIL.
    """
    path = Path(file_path)
    filename = path.name
    date_from_filename = _extract_date_from_filename(filename)

    rasterio_error = None

    # ---------------------------------------------------------
    # Primary reader: Rasterio
    # ---------------------------------------------------------
    if HAS_RASTERIO:
        try:
            with rasterio.open(file_path) as src:

                # -----------------------------
                # CRS
                # -----------------------------
                crs_str = None
                epsg_code = None
                is_georeferenced = False

                if src.crs:
                    crs_str = src.crs.to_string()
                    epsg_code = src.crs.to_epsg()
                    is_georeferenced = True

                # -----------------------------
                # Bounds
                # -----------------------------
                bounds = None

                if is_georeferenced:
                    b = src.bounds

                    bounds = {
                        "min_x": round(float(b.left), 6),
                        "min_y": round(float(b.bottom), 6),
                        "max_x": round(float(b.right), 6),
                        "max_y": round(float(b.top), 6),
                        "bbox_list": [
                            round(float(b.left), 6),
                            round(float(b.bottom), 6),
                            round(float(b.right), 6),
                            round(float(b.top), 6),
                        ],
                    }

                # -----------------------------
                # Resolution
                # -----------------------------
                res_x, res_y = src.res

                resolution = {
                    "x": round(float(res_x), 4) if res_x else None,
                    "y": round(float(res_y), 4) if res_y else None,
                    "unit": _get_resolution_unit(src, epsg_code),
                }

                # -----------------------------
                # Pixel transform
                # -----------------------------
                transform = _serialize_transform(src.transform)

                # -----------------------------
                # NoData
                # -----------------------------
                nodata_by_band = [
                    src.nodatavals[i]
                    for i in range(src.count)
                ]

                # JSON-friendly representation
                nodata_by_band = [
                    float(value) if value is not None else None
                    for value in nodata_by_band
                ]

                nodata = (
                    nodata_by_band[0]
                    if src.count > 0
                    and all(value == nodata_by_band[0] for value in nodata_by_band)
                    else None
                )

                # -----------------------------
                # Band descriptions
                # -----------------------------
                band_descriptions = [
                    src.descriptions[i]
                    if src.descriptions[i]
                    else f"Band_{i + 1}"
                    for i in range(src.count)
                ]

                # -----------------------------
                # Dataset tags
                # -----------------------------
                tags = src.tags()

                # -----------------------------
                # Acquisition date
                # -----------------------------
                acquisition_date = (
                    tags.get("TIFFTAG_DATETIME")
                    or tags.get("ACQUISITION_DATE")
                    or tags.get("DATETIME")
                    or date_from_filename
                )

                # -----------------------------
                # CRS WKT
                # -----------------------------
                crs_wkt = src.crs.to_wkt() if src.crs else None

                return {
                    # Georeferencing
                    "is_georeferenced": is_georeferenced,
                    "crs": crs_str or "ungeoreferenced",
                    "epsg": epsg_code,
                    "crs_wkt": crs_wkt,

                    # Raster dimensions
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "dtypes": [str(dt) for dt in src.dtypes],

                    # Band information
                    "band_descriptions": band_descriptions,

                    # Spatial information
                    "resolution": resolution,
                    "bounds": bounds,
                    "transform": transform,

                    # NoData
                    "nodata": nodata,
                    "nodata_by_band": nodata_by_band,

                    # Metadata
                    "tags": tags,
                    "acquisition_date": acquisition_date,

                    # File information
                    "driver": src.driver,
                    "source": "rasterio",
                }

        except Exception as e:
            rasterio_error = str(e)

    # ---------------------------------------------------------
    # Fallback reader: PIL
    # ---------------------------------------------------------
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            bands = (
                len(img.getbands())
                if hasattr(img, "getbands")
                else 1
            )
            mode = img.mode

            result = {
                # PIL images do not provide reliable geospatial CRS
                "is_georeferenced": False,
                "crs": "ungeoreferenced",
                "epsg": None,
                "crs_wkt": None,

                # Dimensions
                "width": w,
                "height": h,
                "bands": bands,
                "dtypes": [str(mode)],

                # Band information
                "band_descriptions": [
                    f"Band_{i + 1}"
                    for i in range(bands)
                ],

                # Pixel-space information
                "resolution": {
                    "x": 1.0,
                    "y": 1.0,
                    "unit": "pixels",
                },
                "bounds": None,
                "transform": None,

                # NoData
                "nodata": None,
                "nodata_by_band": [None] * bands,

                # Metadata
                "tags": {},
                "acquisition_date": date_from_filename,

                # File information
                "driver": img.format or "PIL",
                "source": "pillow_benchmark_fallback",
            }

            if rasterio_error:
                result["rasterio_error"] = rasterio_error

            return result

    except Exception as e:
        return {
            "is_georeferenced": False,
            "crs": "ungeoreferenced",
            "epsg": None,
            "crs_wkt": None,

            "width": 0,
            "height": 0,
            "bands": 0,

            "resolution": {
                "x": None,
                "y": None,
                "unit": "unknown",
            },

            "bounds": None,
            "transform": None,

            "nodata": None,
            "nodata_by_band": [],

            "band_descriptions": [],
            "tags": {},

            "acquisition_date": date_from_filename,

            "driver": None,
            "source": "failed",

            "rasterio_error": rasterio_error,
            "error": f"Failed to extract metadata: {str(e)}",
        }