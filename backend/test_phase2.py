"""
Phase 2 Automated Verification Script
Tests GeoTIFF validation, metadata extraction, modality detection,
and pairwise co-registration checker.
"""
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient
from main import app
from validation.file_validator import validate_file_format
from validation.metadata_extractor import extract_metadata
from validation.modality_detector import detect_modality
from validation.registration_checker import check_registration

TEST_DIR = Path(__file__).resolve().parent / "data" / "test_scratch"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_geotiff(
    filename: str, 
    crs_epsg: int = 32643,  # WGS 84 / UTM Zone 43N (India)
    origin_x: float = 300000.0, 
    origin_y: float = 2100000.0,
    res: float = 10.0,      # Sentinel-2 10m
    bands: int = 3,
    is_sar: bool = False
) -> Path:
    """Generates a synthetic georeferenced TIFF with real CRS and transform."""
    width, height = 64, 64
    filepath = TEST_DIR / filename
    transform = from_origin(origin_x, origin_y, res, res)

    if is_sar:
        # 1-band SAR with speckle noise (Rayleigh / Gamma distribution)
        data = np.random.gamma(shape=2.0, scale=100.0, size=(1, height, width)).astype(np.float32)
        count = 1
        dtype = "float32"
    else:
        # 3-band Optical RGB
        data = np.random.randint(20, 240, size=(bands, height, width), dtype=np.uint8)
        count = bands
        dtype = "uint8"

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=rasterio.crs.CRS.from_epsg(crs_epsg),
        transform=transform,
    ) as dst:
        dst.write(data)
        dst.update_tags(ACQUISITION_DATE="2026-03-15", SENSOR="Sentinel-2A")

    return filepath


def run_phase2_tests():
    print("========================================")
    print("  SatQuery AI - Phase 2 Verification")
    print("========================================")

    # 1. Create real GeoTIFF rasters
    optical_tif = create_synthetic_geotiff("test_optical.tif", origin_x=300000.0, origin_y=2100000.0, bands=3, is_sar=False)
    sar_tif = create_synthetic_geotiff("test_sar.tif", origin_x=300000.0, origin_y=2100000.0, bands=1, is_sar=True)
    offset_tif = create_synthetic_geotiff("test_far_offset.tif", origin_x=900000.0, origin_y=5000000.0, bands=3, is_sar=False)

    # 2. Test File Validator
    print("\n[1] Testing file_validator.py...")
    valid, msg, details = validate_file_format(str(optical_tif), is_benchmark_input=False)
    assert valid, f"Optical GeoTIFF rejected: {msg}"
    print(" -> PASS: GeoTIFF accepted:", msg)

    # Test rejection of non-benchmark PNG when toggle is off
    dummy_png = TEST_DIR / "unflagged.png"
    dummy_png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...")
    valid_png, msg_png, _ = validate_file_format(str(dummy_png), is_benchmark_input=False)
    assert not valid_png, "Standard image without benchmark flag should be rejected"
    print(" -> PASS: Standard PNG without toggle correctly rejected with clear explanation.")

    valid_png_flagged, _, _ = validate_file_format(str(dummy_png), is_benchmark_input=True)
    assert valid_png_flagged, "Standard image with benchmark toggle enabled should be accepted"
    print(" -> PASS: Standard PNG accepted when benchmark toggle is enabled.")

    # 3. Test Metadata Extractor
    print("\n[2] Testing metadata_extractor.py...")
    meta_opt = extract_metadata(str(optical_tif))
    assert meta_opt["is_georeferenced"] is True
    assert meta_opt["epsg"] == 32643
    assert meta_opt["bands"] == 3
    assert meta_opt["resolution"]["x"] == 10.0
    assert meta_opt["bounds"]["min_x"] == 300000.0
    print(" -> PASS: Metadata correctly extracted:", {
        "crs": meta_opt["crs"],
        "resolution": meta_opt["resolution"],
        "bands": meta_opt["bands"],
        "acquisition_date": meta_opt["acquisition_date"]
    })

    # 4. Test Modality Detector
    print("\n[3] Testing modality_detector.py...")
    mod_opt = detect_modality(str(optical_tif))
    assert mod_opt["modality"] == "OPTICAL", f"Expected OPTICAL, got {mod_opt['modality']}"
    print(f" -> PASS: Optical image detected as {mod_opt['modality']} (conf={mod_opt['confidence']})")

    mod_sar = detect_modality(str(sar_tif), slot_hint="sar")
    assert mod_sar["modality"] == "SAR", f"Expected SAR, got {mod_sar['modality']}"
    print(f" -> PASS: SAR image detected as {mod_sar['modality']} (conf={mod_sar['confidence']})")

    # 5. Test Registration Checker
    print("\n[4] Testing registration_checker.py...")
    meta_sar = extract_metadata(str(sar_tif))
    reg_aligned = check_registration(meta_opt, meta_sar, overlap_threshold=70.0)
    assert reg_aligned["is_co_registered"] is True
    assert reg_aligned["overlap_percentage"] == 100.0
    print(f" -> PASS: Co-registered pair overlap: {reg_aligned['overlap_percentage']}% ({reg_aligned['flag']})")

    meta_offset = extract_metadata(str(offset_tif))
    reg_offset = check_registration(meta_opt, meta_offset, overlap_threshold=70.0)
    assert reg_offset["is_co_registered"] is False
    print(f" -> PASS: Non-overlapping pair correctly flagged as: {reg_offset['flag']}")

    # 6. Test Endpoint Integration via FastAPI TestClient
    print("\n[5] Testing POST /upload with Phase 2 validation & registration...")
    client = TestClient(app)
    with open(optical_tif, "rb") as f_opt, open(sar_tif, "rb") as f_sar:
        res = client.post(
            "/upload",
            files={
                "optical": ("sentinel2_optical.tif", f_opt, "image/tiff"),
                "sar": ("sentinel1_sar.tif", f_sar, "image/tiff"),
            },
            data={"is_benchmark": "false"}
        )
    assert res.status_code == 200, f"Upload failed: {res.text}"
    data = res.json()
    assert "co_registration" in data
    assert data["files"]["optical"]["modality"]["modality"] == "OPTICAL"
    assert data["files"]["sar"]["modality"]["modality"] == "SAR"
    assert data["co_registration"]["optical_sar"]["is_co_registered"] is True
    print(" -> PASS: End-to-end /upload response contains full Phase 2 metadata & registration manifest!")

    print("\n========================================")
    print(" ALL PHASE 2 TESTS PASSED SUCCESSFULLY!")
    print("========================================")

if __name__ == "__main__":
    run_phase2_tests()
