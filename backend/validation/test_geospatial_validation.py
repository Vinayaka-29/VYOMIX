import numpy as np
import rasterio
import pytest
from rasterio.transform import from_origin
from pathlib import Path

from validation.metadata_extractor import extract_metadata
from validation.raster_preprocessor import (
    load_raster,
    normalize_raster,
    resize_raster,
    tile_raster,
)
from validation.registration_checker import check_registration


BIGEARTHNET = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "bigearthnet_patches"
    / "S2A_BEN_0000_Urban_fabric.tif"
)


def create_raster(path, data, crs="EPSG:32643", transform=None, nodata=None):
    """Create a small synthetic GeoTIFF for an isolated test case."""
    data = np.asarray(data)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    bands, height, width = data.shape

    if transform is None:
        transform = from_origin(320000, 2150000, 10, 10)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_real_bigearthnet_metadata():
    """Real BigEarthNet input should expose usable geospatial metadata."""
    metadata = extract_metadata(str(BIGEARTHNET))

    assert metadata["is_georeferenced"] is True
    assert metadata["epsg"] == 32643
    assert metadata["width"] == 128
    assert metadata["height"] == 128
    assert metadata["bands"] == 4
    assert len(metadata["dtypes"]) == 4


def test_real_bigearthnet_preserves_multiband_data():
    """The shared loader must preserve all source bands."""
    raster = load_raster(str(BIGEARTHNET))

    assert raster["data"].shape == (4, 128, 128)
    assert raster["bands"] == 4
    assert raster["dtype"] == ["uint8", "uint8", "uint8", "uint8"]


def test_normalization_preserves_band_count_and_range():
    data = np.array(
        [
            [[0, 10], [20, 30]],
            [[100, 200], [300, 400]],
        ],
        dtype=np.uint16,
    )

    result = normalize_raster(data)

    assert result["data"].shape == (2, 2, 2)
    assert result["bands"] == 2
    assert result["dtype"] == "float32"
    assert np.isclose(np.nanmin(result["data"]), 0.0)
    assert np.isclose(np.nanmax(result["data"]), 1.0)


def test_nodata_is_explicitly_preserved_as_nan():
    data = np.array([[[0, 1], [2, 255]]], dtype=np.uint8)

    result = normalize_raster(data, nodata=255)

    assert np.isnan(result["data"][0, 1, 1])
    assert result["nodata_value"] == "NaN"


def test_resize_updates_spatial_resolution():
    raster = load_raster(str(BIGEARTHNET))

    result = resize_raster(
        raster["data"],
        64,
        64,
        transform=raster["transform"],
    )

    assert result["data"].shape == (4, 64, 64)
    assert result["original_shape"] == (4, 128, 128)
    assert result["scale_x"] == pytest.approx(2.0)
    assert result["scale_y"] == pytest.approx(2.0)

    assert result["transform"].a == pytest.approx(20.0)
    assert result["transform"].e == pytest.approx(-20.0)


def test_tiling_includes_edge_tiles_without_dropping_pixels():
    raster = load_raster(str(BIGEARTHNET))

    tiles = tile_raster(
        raster["data"],
        50,
        50,
        transform=raster["transform"],
    )

    assert len(tiles) == 9

    shapes = [tile["data"].shape for tile in tiles]

    assert (4, 50, 50) in shapes
    assert (4, 50, 28) in shapes
    assert (4, 28, 50) in shapes
    assert (4, 28, 28) in shapes

    positions = [(tile["row"], tile["col"]) for tile in tiles]

    assert (0, 0) in positions
    assert (50, 50) in positions
    assert (100, 100) in positions


def test_registration_accepts_matching_georeferenced_rasters(tmp_path):
    data = np.ones((1, 4, 4), dtype=np.uint8)

    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"

    create_raster(first, data)
    create_raster(second, data)

    result = check_registration(
    extract_metadata(str(first)),
    extract_metadata(str(second)),
)

    assert result["flag"] == "CO_REGISTERED"
    assert result["crs_match"] is True
    assert result["details"]["dimensions_match"] is True
    assert result["details"]["transform_match"] is True


def test_registration_rejects_crs_mismatch(tmp_path):
    data = np.ones((1, 4, 4), dtype=np.uint8)

    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"

    create_raster(first, data, crs="EPSG:32643")
    create_raster(second, data, crs="EPSG:4326")

    result = check_registration(
    extract_metadata(str(first)),
    extract_metadata(str(second)),
)

    assert result["crs_match"] is False
    assert result["flag"] != "CO_REGISTERED"


def test_registration_flags_missing_georeferencing(tmp_path):
    data = np.ones((1, 4, 4), dtype=np.uint8)

    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"

    create_raster(first, data, crs=None)
    create_raster(second, data, crs=None)

    result = check_registration(
    extract_metadata(str(first)),
    extract_metadata(str(second)),
)

    assert result["flag"] == "MISSING_GEOREFERENCING"


def test_registration_rejects_non_overlapping_rasters(tmp_path):
    data = np.ones((1, 4, 4), dtype=np.uint8)

    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"

    create_raster(
        first,
        data,
        transform=from_origin(0, 40, 10, 10),
    )

    create_raster(
        second,
        data,
        transform=from_origin(1000, 1040, 10, 10),
    )

    result = check_registration(
    extract_metadata(str(first)),
    extract_metadata(str(second)),
)

    assert result["flag"] != "CO_REGISTERED"
    assert result["overlap_percentage"] == 0.0


def test_unreadable_raster_is_rejected(tmp_path):
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"this is not a valid raster")

    with pytest.raises(ValueError):
        load_raster(str(corrupt))
