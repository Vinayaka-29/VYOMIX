from pathlib import Path
from typing import Any, Dict

from affine import Affine
import numpy as np
import rasterio


def load_raster(file_path: str) -> Dict[str, Any]:
    """
    Safely load a raster and expose its core data + geospatial metadata.

    The original raster file is never modified.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {file_path}")

    if path.suffix.lower() not in {".tif", ".tiff", ".geotiff"}:
        raise ValueError(
            f"Unsupported raster format: {path.suffix}. "
            "Expected GeoTIFF/TIFF."
        )

    try:
        with rasterio.open(path) as src:
            data = src.read()

            if data.size == 0:
                raise ValueError("Raster contains no pixel data.")

            if data.ndim != 3:
                raise ValueError(
                    f"Unexpected raster shape: {data.shape}. "
                    "Expected (bands, height, width)."
                )

            return {
                "data": data,
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": [str(dtype) for dtype in src.dtypes],
                "crs": src.crs.to_string() if src.crs else None,
                "transform": tuple(src.transform),
                "nodata": src.nodata,
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
                "driver": src.driver,
                "path": str(path),
            }

    except rasterio.errors.RasterioIOError as exc:
        raise ValueError(
            f"Raster could not be opened or is not a valid readable raster: {exc}"
        ) from exc

def normalize_raster(
    data: np.ndarray,
    nodata=None,
) -> Dict[str, Any]:
    """
    Create a model-ready normalized representation of raster data.

    - Preserves the original number of bands.
    - Converts values to float32.
    - Scales valid pixels to [0, 1] independently per band.
    - Keeps NoData pixels as NaN.
    - Never modifies the original input array.
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("Raster data must be a NumPy array.")

    if data.ndim != 3:
        raise ValueError(
            f"Expected raster data with shape (bands, height, width), "
            f"got {data.shape}."
        )

    normalized = data.astype(np.float32, copy=True)

    valid_mask = np.ones(data.shape, dtype=bool)

    if nodata is not None:
        valid_mask = data != nodata

    for band_index in range(data.shape[0]):
        band = normalized[band_index]
        mask = valid_mask[band_index]

        if not np.any(mask):
            normalized[band_index] = np.nan
            continue

        valid_values = band[mask]

        band_min = np.min(valid_values)
        band_max = np.max(valid_values)

        if band_max > band_min:
            band[mask] = (
                (valid_values - band_min)
                / (band_max - band_min)
            )
        else:
            band[mask] = 0.0

        band[~mask] = np.nan

    return {
        "data": normalized,
        "bands": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "dtype": "float32",
        "range": [0.0, 1.0],
        "nodata_value": "NaN",
    }

def resize_raster(
    data: np.ndarray,
    target_height: int,
    target_width: int,
    transform=None,
    nodata=None,
) -> Dict[str, Any]:
    """
    Resize raster data while preserving the spatial relationship.

    Returns both the resized array and an updated affine transform.

    CRS is not changed or invented.
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("Raster data must be a NumPy array.")

    if data.ndim != 3:
        raise ValueError(
            f"Expected raster data with shape (bands, height, width), "
            f"got {data.shape}."
        )

    if target_height <= 0 or target_width <= 0:
        raise ValueError(
            "Target height and width must be positive integers."
        )

    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    bands, height, width = data.shape

    # Pixel-size scaling required by the resize.
    scale_x = width / target_width
    scale_y = height / target_height

    if transform is None:
        source_transform = Affine.identity()
    else:
        source_transform = Affine(*transform)

    # Keep the same geographic origin while adjusting pixel size.
    resized_transform = source_transform @ Affine.scale(
        scale_x,
        scale_y,
    )
    profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": bands,
    "dtype": data.dtype,
    "transform": source_transform,
    }

    if nodata is not None:
        profile["nodata"] = nodata



    with MemoryFile() as memfile:
        with memfile.open(**profile) as src:
            src.write(data)

            if nodata is None:
                resized = src.read(
                    out_shape=(bands, target_height, target_width),
                    resampling=Resampling.bilinear,
                )

            else:
            # Replace NoData with NaN before interpolation.
                # Resize each band using rasterio's resampling.
                resized = src.read(
                    out_shape=(bands, target_height, target_width),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
        # Explicitly preserve the source NoData region.
                valid_mask = data != nodata

                with MemoryFile() as mask_memfile:
                    mask_profile = {
                        "driver": "GTiff",
                        "height": height,
                        "width": width,
                        "count": bands,
                        "dtype": "uint8",
                        "transform": source_transform,
                    }

                    with mask_memfile.open(**mask_profile) as mask_src:
                        mask_src.write(valid_mask.astype(np.uint8))

                        resized_mask = mask_src.read(
                            out_shape=(bands, target_height, target_width),
                            resampling=Resampling.average,
                        ).astype(np.float32)

                resized[resized_mask < 1.0] = np.nan
    return {
        "data": resized,
        "transform": resized_transform,
        "original_shape": (bands, height, width),
        "shape": resized.shape,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
def tile_raster(
    data: np.ndarray,
    tile_height: int,
    tile_width: int,
    transform=None,
) -> list:
    """
    Split a raster into manageable tiles while preserving
    the geospatial transform of each tile.

    The original raster is never modified.
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("Raster data must be a NumPy array.")

    if data.ndim != 3:
        raise ValueError(
            f"Expected raster data with shape (bands, height, width), "
            f"got {data.shape}."
        )

    if tile_height <= 0 or tile_width <= 0:
        raise ValueError("Tile dimensions must be greater than zero.")

    bands, height, width = data.shape

    if transform is None:
        source_transform = Affine.identity()
    else:
        source_transform = Affine(*transform)

    tiles = []

    for row in range(0, height, tile_height):
        for col in range(0, width, tile_width):

            row_end = min(row + tile_height, height)
            col_end = min(col + tile_width, width)

            tile_data = data[:, row:row_end, col:col_end]

            tile_transform = (
                source_transform
                @ Affine.translation(col, row)
            )

            tiles.append({
                "data": tile_data,
                "transform": tile_transform,
                "row": row,
                "col": col,
                "height": row_end - row,
                "width": col_end - col,
            })

    return tiles
