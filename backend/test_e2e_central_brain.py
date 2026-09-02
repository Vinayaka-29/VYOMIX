"""
Phase 15: Master End-to-End System Tests for Central Brain
Validates the complete execution flow through the FastAPI test client across all 5 pipelines,
including mismatch rejections, PDF generation, and observable execution trace auditing.
"""
import os
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

TEST_DIR = Path(__file__).resolve().parent / "data" / "test_brain_e2e"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def create_test_geotiff(filename: str, width: int = 128, height: int = 128, bands: int = 3, is_sar: bool = False):
    filepath = TEST_DIR / filename
    transform = from_origin(300000.0, 2000000.0, 10.0, 10.0)
    crs = "EPSG:32643"

    if is_sar:
        # High speckle noise for SAR detection (CV > 0.45)
        np.random.seed(42)
        base = np.random.exponential(scale=50.0, size=(height, width)).astype(np.float32)
        data = np.clip(base, 0, 255).astype(np.uint8)
        with rasterio.open(
            filepath, "w",
            driver="GTiff",
            height=height, width=width,
            count=1, dtype="uint8",
            crs=crs, transform=transform
        ) as dst:
            dst.write(data, 1)
    else:
        # Clean RGB optical bands
        data = np.zeros((bands, height, width), dtype=np.uint8)
        data[0, :, :] = 180  # Red
        data[1, :, :] = 140  # Green
        data[2, :, :] = 90   # Blue
        with rasterio.open(
            filepath, "w",
            driver="GTiff",
            height=height, width=width,
            count=bands, dtype="uint8",
            crs=crs, transform=transform
        ) as dst:
            for b in range(1, bands + 1):
                dst.write(data[b - 1], b)

    return str(filepath)


def run_e2e_tests():
    print("==========================================================")
    print("  SatQuery AI - Phase 15 End-to-End System Tests          ")
    print("==========================================================")

    # 1. Generate Synthetic Multi-Modal Rasters
    print("\n[Step 1] Synthesizing Multi-Modal GeoTIFF Test Set...")
    opt_path = create_test_geotiff("test_optical.tif", bands=3, is_sar=False)
    sar_path = create_test_geotiff("test_sar.tif", bands=1, is_sar=True)
    before_path = create_test_geotiff("test_before.tif", bands=3, is_sar=False)
    
    # After image has 30% area modification
    after_path = str(TEST_DIR / "test_after.tif")
    transform = from_origin(300000.0, 2000000.0, 10.0, 10.0)
    data_after = np.zeros((3, 128, 128), dtype=np.uint8)
    data_after[:, 40:100, 40:100] = 240
    with rasterio.open(
        after_path, "w",
        driver="GTiff",
        height=128, width=128,
        count=3, dtype="uint8",
        crs="EPSG:32643", transform=transform
    ) as dst:
        for b in range(1, 4):
            dst.write(data_after[b - 1], b)

    print(" -> PASS: Synthetic rasters generated with CRS EPSG:32643.")

    # 2. Upload 4-Slot Multi-Modal Imagery
    print("\n[Step 2] Uploading 4-Slot Imagery to POST /upload...")
    with open(opt_path, "rb") as f_opt, \
         open(sar_path, "rb") as f_sar, \
         open(before_path, "rb") as f_b, \
         open(after_path, "rb") as f_a:

        files = {
            "optical": ("test_optical.tif", f_opt, "image/tiff"),
            "sar": ("test_sar.tif", f_sar, "image/tiff"),
            "before": ("test_before.tif", f_b, "image/tiff"),
            "after": ("test_after.tif", f_a, "image/tiff"),
        }
        res_upload = client.post("/upload", files=files)
        assert res_upload.status_code == 200, f"Upload failed: {res_upload.text}"
        manifest = res_upload.json()
        upload_id = manifest["upload_id"]
        print(f" -> PASS: Upload Manifest Created: {upload_id}")
        assert manifest["files"]["sar"]["modality"]["modality"] == "SAR"
        assert manifest["files"]["optical"]["modality"]["modality"] == "OPTICAL"

    # 3. Pipeline 1: Single-Image VQA
    print("\n[Step 3] Testing Pipeline 1: Single-Image VQA via POST /query...")
    res_vqa = client.post("/query", json={
        "upload_id": upload_id,
        "query_text": "What is the dominant land cover in this satellite scene?",
    })
    assert res_vqa.status_code == 200, f"VQA failed: {res_vqa.text}"
    data_vqa = res_vqa.json()
    assert data_vqa["task"] == "single_image_vqa"
    assert data_vqa["confidence"] >= 0.80
    assert len(data_vqa["execution_trace"]["steps"]) >= 1
    print(f" -> PASS: VQA Answered: '{data_vqa['answer'][:60]}...' (Conf: {data_vqa['confidence']})")

    # 4. Pipeline 2: Dense Scene Captioning
    print("\n[Step 4] Testing Pipeline 2: Dense Scene Captioning via POST /query...")
    res_cap = client.post("/query", json={
        "upload_id": upload_id,
        "query_text": "Describe the scene and summarize the terrain features.",
    })
    assert res_cap.status_code == 200, f"Captioning failed: {res_cap.text}"
    data_cap = res_cap.json()
    assert data_cap["task"] == "captioning"
    print(f" -> PASS: Scene Captioned: '{data_cap['answer'][:60]}...'")

    # 5. Pipeline 3: Referring-Expression Grounding
    print("\n[Step 5] Testing Pipeline 3: Text-Guided Grounding via POST /query...")
    res_gnd = client.post("/query", json={
        "upload_id": upload_id,
        "query_text": "Highlight the built-up urban area in this image",
    })
    assert res_gnd.status_code == 200, f"Grounding failed: {res_gnd.text}"
    data_gnd = res_gnd.json()
    assert data_gnd["task"] == "grounding"
    assert data_gnd["visual_artifacts"]["bounding_box"] is not None
    assert len(data_gnd["visual_artifacts"]["bounding_box"]) == 4
    print(f" -> PASS: Target Grounded: BBox={data_gnd['visual_artifacts']['bounding_box']}")

    # 6. Pipeline 4: Bi-Temporal Change Detection & Change-VQA
    print("\n[Step 6] Testing Pipeline 4: Bi-Temporal Change-VQA via POST /query...")
    res_chg = client.post("/query", json={
        "upload_id": upload_id,
        "query_text": "What changed between the before and after dates?",
    })
    assert res_chg.status_code == 200, f"Change-VQA failed: {res_chg.text}"
    data_chg = res_chg.json()
    assert data_chg["task"] == "change_vqa"
    assert len(data_chg["execution_trace"]["steps"]) == 2  # Chained differencing + VQA
    assert data_chg["visual_artifacts"].get("change_overlay_path") is not None
    print(f" -> PASS: Bi-temporal change synthesized. Chained steps: {len(data_chg['execution_trace']['steps'])}")

    # 7. Pipeline 5: Optical + SAR Cross-Modal Fusion
    print("\n[Step 7] Testing Pipeline 5: Optical + SAR Fusion via POST /query...")
    res_fus = client.post("/query", json={
        "upload_id": upload_id,
        "query_text": "Use optical and SAR sensors together to extract land cover and water boundaries.",
    })
    assert res_fus.status_code == 200, f"Fusion failed: {res_fus.text}"
    data_fus = res_fus.json()
    assert data_fus["task"] == "optical_sar_fusion"
    assert "cross_modal_evidence" in data_fus["visual_artifacts"]
    assert "optical" in data_fus["visual_artifacts"]["cross_modal_evidence"]
    assert "sar" in data_fus["visual_artifacts"]["cross_modal_evidence"]
    print(f" -> PASS: Dual-branch cross-modal evidence corroborated.")

    # 8. Test Input Mismatch Diagnostic (Negative Test)
    print("\n[Step 8] Testing Precondition Mismatch Handling...")
    # Create single-image upload
    with open(opt_path, "rb") as f_opt:
        res_single = client.post("/upload", files={"optical": ("test_optical.tif", f_opt, "image/tiff")})
        single_upload_id = res_single.json()["upload_id"]

    res_mismatch = client.post("/query", json={
        "upload_id": single_upload_id,
        "query_text": "What changed between the before and after acquisitions?",
    })
    assert res_mismatch.status_code == 400
    err_detail = res_mismatch.json()["detail"]
    assert "E-TEMP-01" in err_detail["message"]
    print(f" -> PASS: Diagnostic error returned gracefully on mismatched input: '{err_detail['message'][:70]}...'")

    # 9. Test PDF Report Generation
    print("\n[Step 9] Testing Downloadable PDF Mission Report...")
    query_id = data_chg["query_id"]
    res_pdf = client.get(f"/report/{query_id}")
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert len(res_pdf.content) > 1000
    print(f" -> PASS: Mission Report PDF generated successfully ({len(res_pdf.content)} bytes).")

    print("\n==========================================================")
    print(" ALL PHASE 15 END-TO-END SYSTEM TESTS PASSED (100%)!       ")
    print("==========================================================")


if __name__ == "__main__":
    run_e2e_tests()
