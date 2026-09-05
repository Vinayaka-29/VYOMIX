"""
Comprehensive End-to-End Verification Test Suite
Tests all 10 Phases of SatQuery AI:
  - Phase 1 & 2: GeoTIFF Upload & Validation
  - Phase 3: Single-Image VQA & Captioning
  - Phase 4: Text-Guided Grounding
  - Phase 5: LoRA Domain Adaptation Weights & Evaluation
  - Phase 6: Bi-Temporal Change Detection & Differencing
  - Phase 7: Optical + SAR Dual-Branch Fusion
  - Phase 8: Agentic Controller Query Interpretation & Auto-Routing
  - Phase 9: Auditable Execution Trace & Downloadable PDF Reports
  - Phase 10: Benchmark Splits & Summary Matrix
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

# Import evaluation modules to verify Phase 10
from evaluation.eval_vqa import run_vqa_evaluation
from evaluation.eval_grounding import run_grounding_eval
from evaluation.eval_change import run_change_eval
from evaluation.eval_optical_sar import run_optical_sar_eval
from training.lora_finetune_vlm import run_lora_finetuning

TEST_DIR = Path(__file__).resolve().parent / "data" / "test_scratch"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def create_test_raster(filename: str, bands: int = 3, is_sar: bool = False, shift: float = 0.0) -> Path:
    width, height = 128, 128
    filepath = TEST_DIR / filename
    transform = from_origin(350000.0 + shift, 2200000.0, 10.0, 10.0)

    if is_sar:
        data = np.random.gamma(shape=2.5, scale=80.0, size=(1, height, width)).astype(np.float32)
        count = 1
        dtype = "float32"
    else:
        data = np.random.randint(30, 230, size=(bands, height, width), dtype=np.uint8)
        # Create a bright patch for grounding
        data[:, 20:60, 30:80] = 245
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


def run_all_phase_tests():
    print("==========================================================")
    print("  SatQuery AI - Full 10-Phase End-to-End Verification")
    print("==========================================================")

    client = TestClient(app)

    # Prepare rasters
    opt_path = create_test_raster("test_opt.tif", bands=3, is_sar=False)
    sar_path = create_test_raster("test_sar.tif", bands=1, is_sar=True)
    before_path = create_test_raster("test_before.tif", bands=3, is_sar=False)
    after_path = create_test_raster("test_after.tif", bands=3, is_sar=False)

    # 1. Upload All 4 Slots
    print("\n[Phase 1 & 2] Uploading 4-slot multi-modal rasters...")
    with open(opt_path, "rb") as f_opt, open(sar_path, "rb") as f_sar, \
         open(before_path, "rb") as f_bef, open(after_path, "rb") as f_aft:
        up_res = client.post(
            "/upload",
            files={
                "optical": ("sentinel2_opt.tif", f_opt, "image/tiff"),
                "sar": ("sentinel1_sar.tif", f_sar, "image/tiff"),
                "before": ("before_t0.tif", f_bef, "image/tiff"),
                "after": ("after_t1.tif", f_aft, "image/tiff"),
            },
            data={"is_benchmark": "false"}
        )

    assert up_res.status_code == 200, f"Upload failed: {up_res.text}"
    up_data = up_res.json()
    upload_id = up_data["upload_id"]
    print(f" -> PASS: 4 slots uploaded & validated. Upload ID: {upload_id}")
    print(f" -> Detected Modalities: Optical={up_data['files']['optical']['modality']['modality']}, SAR={up_data['files']['sar']['modality']['modality']}")

    # 2. Phase 3: Single-Image VQA & Captioning
    print("\n[Phase 3] Testing Single-Image VQA & Captioning Direct Endpoints...")
    vqa_res = client.post("/debug/vqa", json={"upload_id": upload_id, "question": "What is the dominant land cover?"})
    assert vqa_res.status_code == 200
    print(f" -> PASS: VQA Answer: '{vqa_res.json()['answer'][:60]}...' (Conf: {vqa_res.json()['confidence']})")

    cap_res = client.post("/debug/caption", json={"upload_id": upload_id})
    assert cap_res.status_code == 200
    print(f" -> PASS: Dense Caption: '{cap_res.json()['caption'][:60]}...'")

    # 3. Phase 4: Text-Guided Grounding
    print("\n[Phase 4] Testing Referring-Expression Grounding...")
    ground_res = client.post("/debug/ground", json={"upload_id": upload_id, "expression": "the urban built-up area"})
    assert ground_res.status_code == 200
    g_data = ground_res.json()
    assert g_data["found"] is True
    assert g_data["bbox"] is not None
    print(f" -> PASS: Grounded Bounding Box: {g_data['bbox']} (Normalized: {g_data['normalized_bbox']})")

    # 4. Phase 5: LoRA Domain Adaptation
    print("\n[Phase 5] Verifying LoRA Adapter Checkpoint & Before/After Evaluation...")
    lora_config = run_lora_finetuning(stage="smoke", num_epochs=1)
    assert lora_config["peft_type"] == "LORA"
    vqa_eval_res = run_vqa_evaluation()
    assert "base_model" in vqa_eval_res and "adapted_model" in vqa_eval_res
    print(f" -> PASS: LoRA adaptation verified: Base F1={vqa_eval_res['base_model']['mean_token_f1']} -> Adapted F1={vqa_eval_res['adapted_model']['mean_token_f1']}")

    # 5. Phase 6: Bi-Temporal Change Detection & Change-VQA
    print("\n[Phase 6] Testing Bi-Temporal Differencing & Change-VQA...")
    change_res = client.post("/debug/change", json={
        "before_upload_id": upload_id,
        "after_upload_id": upload_id,
        "question": "Has the built-up area increased?"
    })
    assert change_res.status_code == 200
    c_data = change_res.json()
    assert "change_metrics" in c_data
    print(f" -> PASS: Change-VQA Answer: '{c_data['answer'][:60]}...' ({c_data['change_metrics']['percentage_changed']}% change)")

    # 6. Phase 7: Optical + SAR Fusion
    print("\n[Phase 7] Testing Optical + SAR Cross-Modal Fusion...")
    fusion_res = client.post("/debug/fusion", json={
        "optical_upload_id": upload_id,
        "sar_upload_id": upload_id,
        "question": "Use optical and SAR together to identify water and urban regions."
    })
    assert fusion_res.status_code == 200
    f_data = fusion_res.json()
    assert "optical" in f_data["evidence"] and "sar" in f_data["evidence"]
    print(f" -> PASS: Fused Answer: '{f_data['answer'][:70]}...'")

    # 7. Phase 8 & 9: Agentic Controller Auto-Routing & Execution Trace
    print("\n[Phase 8 & 9] Testing Agentic Auto-Routing on Unified /query Endpoint...")

    # Test Case A: Routing to Grounding
    q_ground = client.post("/query", json={"upload_id": upload_id, "query_text": "Highlight the water body in this image"})
    assert q_ground.status_code == 200
    qg_data = q_ground.json()
    assert qg_data["task"] == "grounding"
    assert "bounding_box" in qg_data["visual_artifacts"]
    print(f" -> PASS: Auto-routed 'Highlight water body' -> Task: {qg_data['task']}")

    # Test Case B: Routing to Optical+SAR Fusion
    q_fuse = client.post("/query", json={"upload_id": upload_id, "query_text": "Use optical and SAR sensors together to extract land cover"})
    assert q_fuse.status_code == 200
    qf_data = q_fuse.json()
    assert qf_data["task"] == "optical_sar_fusion"
    print(f" -> PASS: Auto-routed 'Optical and SAR together' -> Task: {qf_data['task']}")

    # Test Case C: Routing to Change-VQA
    q_chg = client.post("/query", json={"upload_id": upload_id, "query_text": "What changed between the before and after dates?"})
    assert q_chg.status_code == 200
    qc_data = q_chg.json()
    assert qc_data["task"] == "change_vqa"
    print(f" -> PASS: Auto-routed 'What changed between before and after' -> Task: {qc_data['task']}")

    # Test Case D: Check Auditable Execution Trace & Disagreement Flagging
    trace = qc_data["execution_trace"]
    assert "models_called" in trace
    assert "steps" in trace
    assert "final_confidence" in trace
    assert "disagreement_flagged" in trace
    query_id = qc_data["query_id"]
    print(f" -> PASS: Auditable trace verified ({len(trace['steps'])} steps, confidence={trace['final_confidence']})")

    # 8. Phase 9: PDF Report Generation
    print("\n[Phase 9] Testing Downloadable PDF Report Endpoint (/report/{query_id})...")
    pdf_res = client.get(f"/report/{query_id}")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF-"), "Invalid PDF binary signature"
    print(f" -> PASS: Generated valid mission report PDF ({len(pdf_res.content)} bytes)")

    # 9. Phase 10: Benchmark Evaluations
    print("\n[Phase 10] Running Full Benchmark Suite...")
    run_grounding_eval()
    run_change_eval()
    run_optical_sar_eval()
    print(" -> PASS: All benchmark evaluations generated successfully.")

    print("\n==========================================================")
    print(" ALL 10 PHASES FULLY VERIFIED AND OPERATIONAL! ")
    print("==========================================================")


if __name__ == "__main__":
    run_all_phase_tests()
