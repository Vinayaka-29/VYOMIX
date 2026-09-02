"""
Metadata Extractor for SatQuery AI
Uses rasterio (with fallback to PIL/tifffile) to extract spatial CRS,
spatial resolution (GSD), bounding box, band count, data types, and acquisition dates.
Gracefully handles ungeoreferenced rasters/images.
"""
import re
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def _extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Attempts to extract acquisition date from common remote sensing naming conventions
    e.g. S2A_MSIL2A_20240315T... or Landsat LC08_..._20230512 or simple YYYYMMDD.
    """
    patterns = [
        r"(20\d{2}[01]\d[0-3]\d)T\d{6}",      # 20240315T103021
        r"(20\d{2}[-_][01]\d[-_][0-3]\d)",    # 2024-03-15 or 2024_03_15
        r"(20\d{2}[01]\d[0-3]\d)",            # 20240315
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if m:
            raw = m.group(1).replace("_", "-")
            if len(raw) == 8 and raw.isdigit():
                return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
            return raw
    return None


def extract_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extracts complete geospatial and image metadata.
    Returns a standardized dictionary.
    """
    path = Path(file_path)
    filename = path.name
    date_from_filename = _extract_date_from_filename(filename)

    # First attempt reading with rasterio if available
    if HAS_RASTERIO:
        try:
            with rasterio.open(file_path) as src:
                crs_str = None
                epsg_code = None
                is_georeferenced = False

                if src.crs:
                    crs_str = src.crs.to_string()
                    epsg_code = src.crs.to_epsg()
                    is_georeferenced = True

                bounds = None
                if is_georeferenced:
                    b = src.bounds
                    bounds = {
                        "min_x": round(b.left, 6),
                        "min_y": round(b.bottom, 6),
                        "max_x": round(b.right, 6),
                        "max_y": round(b.top, 6),
                        "bbox_list": [round(b.left, 6), round(b.bottom, 6), round(b.right, 6), round(b.top, 6)],
                    }

                # Spatial resolution (ground sample distance)
                res_x, res_y = src.res
                resolution = {
                    "x": round(float(res_x), 4) if res_x else None,
                    "y": round(float(res_y), 4) if res_y else None,
                    "unit": "meters" if epsg_code and epsg_code != 4326 else ("degrees" if epsg_code == 4326 else "pixel_units"),
                }

                # Tags / acquisition date
                tags = src.tags()
                acquisition_date = (
                    tags.get("TIFFTAG_DATETIME")
                    or tags.get("ACQUISITION_DATE")
                    or tags.get("DATETIME")
                    or date_from_filename
                )

                band_descriptions = [src.descriptions[i] or f"Band_{i+1}" for i in range(src.count)]

                return {
                    "is_georeferenced": is_georeferenced,
                    "crs": crs_str or "ungeoreferenced",
                    "epsg": epsg_code,
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "dtypes": [str(dt) for dt in src.dtypes],
                    "resolution": resolution,
                    "bounds": bounds,
                    "acquisition_date": acquisition_date,
                    "driver": src.driver,
                    "source": "rasterio",
                }
        except Exception as e:
            # Fall back to standard image reader for non-geotiff files
            pass

    # Fallback for standard benchmark images (PNG, JPEG) or non-rasterio rasters
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            bands = len(img.getbands()) if hasattr(img, "getbands") else 1
            mode = img.mode

            return {
                "is_georeferenced": False,
                "crs": "ungeoreferenced",
                "epsg": None,
                "width": w,
                "height": h,
                "bands": bands,
                "dtypes": [str(mode)],
                "resolution": {
                    "x": 1.0,
                    "y": 1.0,
                    "unit": "pixels",
                },
                "bounds": None,
                "acquisition_date": date_from_filename,
                "driver": img.format or "PIL",
                "source": "pillow_benchmark_fallback",
            }
    except Exception as e:
        return {
            "is_georeferenced": False,
            "crs": "ungeoreferenced",
            "error": f"Failed to extract metadata: {str(e)}",
            "width": 0,
            "height": 0,
            "bands": 0,
            "acquisition_date": date_from_filename,
        }
