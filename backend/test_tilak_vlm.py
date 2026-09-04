"""
Comprehensive Automated Test Suite for Tilak's Visual Intelligence Layer
SIH Problem Statement 26167 | Team Vyomix

Validates all 8 core capabilities:
  - Test 1: Real VQA on satellite raster
  - Test 2: Real Dense Captioning on satellite raster
  - Test 3: Real Grounding returning valid bounding boxes & absent entity rejection
  - Test 4: GeoTIFF preprocessing & physical spectral indices (NDVI, NDWI, SAR)
  - Test 5: Model loading singleton test (single load, reused across requests)
  - Test 6: LoRA Adapter checkpoint loading test (>1 MB weights, PEFT configuration)
  - Test 7: Invalid input rejection & error boundaries
  - Test 8: End-to-end integration with Central Brain /query endpoint
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient
from main import app

from models.model_server import model_server, RemoteSensingVLMServer
from models.vqa_model import answer_question
from models.captioning_model import generate_caption
from models.grounding_model import ground_expression

TEST_DIR = BACKEND_DIR / "data" / "test_scratch"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_geotiff(filename: str, bands: int = 4, is_sar: bool = False) -> Path:
    """Generates a realistic 128x128 GeoTIFF raster with CRS and transform."""
    width, height = 128, 128
    filepath = TEST_DIR / filename
    transform = from_origin(320000.0, 2150000.0, 10.0, 10.0)

    if is_sar:
        data = np.random.gamma(shape=2.5, scale=70.0, size=(1, height, width)).astype(np.float32)
        count = 1
        dtype = "float32"
    else:
        data = np.zeros((bands, height, width), dtype=np.uint8)
        # Background vegetation (R=45, G=140, B=50, NIR=220)
        data[0] = 45
        data[1] = 140
        data[2] = 50
        data[3] = 220
        # Water body patch in top-left (R=25, G=50, B=120, NIR=15)
        data[0, 10:45, 10:55] = 25
        data[1, 10:45, 10:55] = 50
        data[2, 10:45, 10:55] = 120
        data[3, 10:45, 10:55] = 15
        # Urban built-up patch in bottom-right (R=190, G=185, B=180, NIR=195)
        data[0, 70:115, 65:120] = 190
        data[1, 70:115, 65:120] = 185
        data[2, 70:115, 65:120] = 180
        data[3, 70:115, 65:120] = 195
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
        crs=rasterio.crs.CRS.from_epsg(32643),
        transform=transform,
    ) as dst:
        dst.write(data)

    return filepath


def run_tilak_vlm_tests():
    print("==========================================================")
    print(" SatQuery AI - Tilak's Visual Intelligence Validation ")
    print(" Problem Statement 26167 | Team Vyomix")
    print("==========================================================")

    # Prepare rasters
    optical_tif = create_sample_geotiff("vlm_test_optical.tif", bands=4, is_sar=False)
    sar_tif = create_sample_geotiff("vlm_test_sar.tif", bands=1, is_sar=True)

    # -----------------------------------------------------------------
    # Test 1: Real VQA on Satellite Raster
    # -----------------------------------------------------------------
    print("\n[Test 1] Real VQA on Satellite Raster...")
    q1 = "What is the dominant land cover class in this Sentinel-2 tile?"
    res_vqa = answer_question(str(optical_tif), q1)
    assert res_vqa["status"] == "success", f"VQA failed: {res_vqa}"
    assert len(res_vqa["answer"]) > 20, "VQA answer too short"
    assert 0.65 <= res_vqa["confidence"] <= 0.98, f"Confidence out of range: {res_vqa['confidence']}"
    assert res_vqa["model"] in ["SatQuery-RS-Adapted-VLM", "SatQuery-RS-VLM-Base"]
    assert "evidence" in res_vqa and len(res_vqa["evidence"]) > 0
    print(f" -> PASS: VQA generated answer with conf={res_vqa['confidence']} in {res_vqa['latency_ms']}ms:")
    print(f"    Answer: '{res_vqa['answer'][:90]}...'")

    # -----------------------------------------------------------------
    # Test 2: Real Dense Captioning on Satellite Raster
    # -----------------------------------------------------------------
    print("\n[Test 2] Real Dense Captioning on Satellite Raster...")
    res_cap = generate_caption(str(optical_tif))
    assert res_cap["status"] == "success", f"Captioning failed: {res_cap}"
    assert len(res_cap["caption"]) > 40, "Caption too short"
    assert 0.68 <= res_cap["confidence"] <= 0.98, f"Confidence out of range: {res_cap['confidence']}"
    assert "features_detected" in res_cap and len(res_cap["features_detected"]) > 0
    print(f" -> PASS: Dense caption generated (conf={res_cap['confidence']}):")
    print(f"    Features: {res_cap['features_detected']}")
    print(f"    Caption: '{res_cap['caption'][:95]}...'")

    # -----------------------------------------------------------------
    # Test 3: Real Grounding & Absent Entity Detection
    # -----------------------------------------------------------------
    print("\n[Test 3] Referring-Expression Grounding & Absent Entity Rejection...")
    # 3A. Present entity: Water
    res_ground_water = ground_expression(str(optical_tif), "the surface water body")
    assert res_ground_water["status"] == "success"
    assert res_ground_water["found"] is True, f"Water should be detected: {res_ground_water}"
    assert res_ground_water["bbox"] is not None
    assert len(res_ground_water["bbox"]) == 4
    assert res_ground_water["normalized_bbox"] is not None
    for coord in res_ground_water["normalized_bbox"]:
        assert 0.0 <= coord <= 1.0, f"Normalized coord out of [0, 1]: {coord}"
    print(f" -> PASS: Present entity grounded: BBox={res_ground_water['bbox']}, Norm={res_ground_water['normalized_bbox']}")

    # 3B. Present entity: Built-up
    res_ground_urban = ground_expression(str(optical_tif), "urban built-up buildings")
    assert res_ground_urban["found"] is True
    print(f" -> PASS: Urban entity grounded: BBox={res_ground_urban['bbox']}")

    # 3C. Absent entity rejection
    res_ground_absent = ground_expression(str(optical_tif), "African elephant herd in jungle")
    assert res_ground_absent["found"] is False, "Absent entity should NOT be found"
    assert res_ground_absent["bbox"] is None
    assert res_ground_absent["confidence"] < 0.35
    print(f" -> PASS: Absent entity correctly rejected (found=False, conf={res_ground_absent['confidence']}): '{res_ground_absent['message']}'")

    # -----------------------------------------------------------------
    # Test 4: GeoTIFF Preprocessing & Physical Spectral Indices
    # -----------------------------------------------------------------
    print("\n[Test 4] GeoTIFF Preprocessing & Earth Observation Indices...")
    # Optical 4-band inspection
    opt_info = model_server.inspect_raster_channels(str(optical_tif))
    assert opt_info["channels"] == 4
    assert opt_info["veg_index"] > 0.10, f"Expected high vegetation index for optical scene, got {opt_info['veg_index']}"
    assert opt_info["is_sar"] is False
    tensor_opt = model_server.prepare_input_tensor(opt_info)
    assert tensor_opt.shape == (1, 4, 128, 128)
    print(f" -> PASS: Optical inspection: NDVI={opt_info['veg_index']:.3f}, NDWI={opt_info['water_index']:.3f}, Brightness={opt_info['brightness']:.3f}")

    # SAR inspection
    sar_info = model_server.inspect_raster_channels(str(sar_tif))
    assert sar_info["channels"] == 1
    assert sar_info["is_sar"] is True
    tensor_sar = model_server.prepare_input_tensor(sar_info)
    assert tensor_sar.shape == (1, 4, 128, 128)
    print(f" -> PASS: SAR inspection: is_sar={sar_info['is_sar']}, Brightness={sar_info['brightness']:.3f}")

    # -----------------------------------------------------------------
    # Test 5: Model Loading Singleton Test
    # -----------------------------------------------------------------
    print("\n[Test 5] Model Loading Singleton Verification...")
    server1 = RemoteSensingVLMServer()
    server2 = RemoteSensingVLMServer()
    assert server1 is server2, "RemoteSensingVLMServer must be a singleton"
    t_start = time.time()
    server1.initialize()
    server2.initialize()
    t_init = (time.time() - t_start) * 1000
    assert t_init < 50.0, f"Re-initialization must be instantaneous, took {t_init}ms"
    print(f" -> PASS: Singleton verified (server1 is server2). Re-init overhead: {t_init:.2f}ms")

    # -----------------------------------------------------------------
    # Test 6: Adapter Checkpoint Loading & Sizing (>1 MB)
    # -----------------------------------------------------------------
    print("\n[Test 6] Adapter Checkpoint Loading & Weight Sizing (>1 MB)...")
    ckpt_dir = BACKEND_DIR / "models" / "checkpoints" / "lora_adapter"
    safetensors_file = ckpt_dir / "adapter_model.safetensors"
    bin_file = ckpt_dir / "adapter_model.bin"
    config_file = ckpt_dir / "adapter_config.json"

    assert config_file.exists(), f"Missing adapter_config.json at {config_file}"
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg.get("peft_type") == "LORA", f"Expected LORA peft_type, got {cfg.get('peft_type')}"
    assert cfg.get("r") == 32, f"Expected rank 32, got {cfg.get('r')}"

    assert safetensors_file.exists(), f"Missing adapter_model.safetensors at {safetensors_file}"
    st_size = safetensors_file.stat().st_size
    assert st_size > 1_000_000, f"Safetensors weight size must be > 1 MB, got {st_size} bytes ({st_size / (1024*1024):.2f} MB)"

    assert bin_file.exists(), f"Missing adapter_model.bin at {bin_file}"
    bin_size = bin_file.stat().st_size
    assert bin_size > 1_000_000, f"Bin weight size must be > 1 MB, got {bin_size} bytes ({bin_size / (1024*1024):.2f} MB)"

    counts = model_server.param_info
    assert counts.get("trainable_lora", 0) > 1_000_000, f"Expected >1M trainable LoRA params, got {counts}"
    print(f" -> PASS: LoRA adapter weights verified:")
    print(f"    adapter_model.safetensors: {st_size:,} bytes ({st_size / (1024*1024):.2f} MB)")
    print(f"    adapter_model.bin: {bin_size:,} bytes ({bin_size / (1024*1024):.2f} MB)")
    print(f"    Trainable LoRA parameters: {counts['trainable_lora']:,} ({counts['trainable_lora'] / counts['total'] * 100:.1f}%)")

    # -----------------------------------------------------------------
    # Test 7: Invalid Input Rejection
    # -----------------------------------------------------------------
    print("\n[Test 7] Invalid Input Rejection & Error Handling...")
    try:
        model_server.inspect_raster_channels("C:/non_existent/fake_image.tif")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        print(" -> PASS: Non-existent image correctly raised FileNotFoundError")

    # -----------------------------------------------------------------
    # Test 8: End-to-End Integration with Central Brain /query
    # -----------------------------------------------------------------
    print("\n[Test 8] End-to-End Central Brain Integration via FastAPI TestClient...")
    client = TestClient(app)

    with open(optical_tif, "rb") as f_opt, open(sar_tif, "rb") as f_sar:
        up_res = client.post(
            "/upload",
            files={
                "optical": ("test_optical.tif", f_opt, "image/tiff"),
                "sar": ("test_sar.tif", f_sar, "image/tiff"),
            },
            data={"is_benchmark": "false"}
        )
    assert up_res.status_code == 200
    upload_id = up_res.json()["upload_id"]

    # Test VQA query
    q_res = client.post("/query", json={
        "upload_id": upload_id,
        "query_text": "What is the dominant land cover class in this Sentinel-2 tile?"
    })
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert q_data["status"] == "completed"
    assert q_data["task"] == "single_image_vqa"
    assert len(q_data["answer"]) > 10
    assert "execution_trace" in q_data
    print(f" -> PASS: /query VQA completed: task={q_data['task']}, confidence={q_data['confidence']}")

    # Test Grounding query
    g_res = client.post("/query", json={
        "upload_id": upload_id,
        "query_text": "Highlight the water body in this image"
    })
    assert g_res.status_code == 200
    g_data = g_res.json()
    assert g_data["status"] == "completed"
    assert g_data["task"] == "grounding"
    assert "bounding_box" in g_data["visual_artifacts"]
    print(f" -> PASS: /query Grounding completed: task={g_data['task']}, artifact={g_data['visual_artifacts']['bounding_box']}")

    print("\n==========================================================")
    print(" ALL 8 TILAK VISUAL INTELLIGENCE TESTS PASSED 100%! ")
    print("==========================================================")


if __name__ == "__main__":
    run_tilak_vlm_tests()
