"""
File Validator for SatQuery AI
Ensures uploaded files conform to geospatial specifications (GeoTIFF/TIFF primary,
or PNG/JPEG for benchmark mode).
"""
import os
from pathlib import Path
from typing import Tuple, Dict, Any

ALLOWED_GEO_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
ALLOWED_BENCHMARK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

MAGIC_NUMBERS = {
    # TIFF little-endian
    b"II*\x00": "TIFF (Little Endian)",
    # TIFF big-endian
    b"MM\x00*": "TIFF (Big Endian)",
    # BigTIFF
    b"II+\x00": "BigTIFF",
    b"MM\x00+": "BigTIFF",
    # PNG
    b"\x89PNG\r\n\x1a\n": "PNG",
    # JPEG
    b"\xff\xd8\xff": "JPEG",
}


def validate_file_format(file_path: str, is_benchmark_input: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates the uploaded file by both extension and header magic bytes.
    Returns: (is_valid: bool, message: str, details: dict)
    """
    path = Path(file_path)
    if not path.exists():
        return False, f"File does not exist: {file_path}", {}

    extension = path.suffix.lower()
    file_size = os.path.getsize(file_path)

    # Read the first 16 bytes for magic check
    header = b""
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
    except Exception as e:
        return False, f"Failed to read file header: {str(e)}", {}

    detected_type = "unknown"
    for magic, desc in MAGIC_NUMBERS.items():
        if header.startswith(magic) or (magic == b"\xff\xd8\xff" and header[:3] == b"\xff\xd8\xff"):
            detected_type = desc
            break

    is_geotiff = extension in ALLOWED_GEO_EXTENSIONS or "TIFF" in detected_type
    is_benchmark = extension in ALLOWED_BENCHMARK_EXTENSIONS or detected_type in ("PNG", "JPEG")

    details = {
        "extension": extension,
        "size_bytes": file_size,
        "detected_magic_type": detected_type,
        "is_geotiff": is_geotiff,
        "is_benchmark_format": is_benchmark,
        "benchmark_mode_flagged": is_benchmark_input,
    }

    if is_geotiff:
        return True, "Valid geospatial raster (GeoTIFF/TIFF).", details

    if is_benchmark:
        if is_benchmark_input:
            return True, "Valid benchmark dataset format (PNG/JPEG).", details
        else:
            return False, (
                f"File '{path.name}' is a standard image ({extension.upper()}). "
                f"For raw satellite imagery, GeoTIFF (.tif) is required. "
                f"If using a benchmark dataset (e.g. RSVQA, VRSBench), please enable the "
                f"'Benchmark Dataset Input' toggle in the upload panel."
            ), details

    return False, (
        f"Unsupported file format '{extension}'. "
        f"Allowed formats: GeoTIFF (.tif, .tiff) or benchmark images (.png, .jpg) with benchmark toggle enabled."
    ), details
