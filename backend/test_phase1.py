"""
Phase 1 Automated Verification Script
Tests /health, /upload, and /query endpoints.
"""
import io
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
from main import app

def run_tests():
    print("========================================")
    print("  SatQuery AI - Phase 1 Verification")
    print("========================================")

    client = TestClient(app)

    # 1. Test /health
    print("\n[1] Testing GET /health...")
    health_res = client.get("/health")
    assert health_res.status_code == 200, f"Health check failed: {health_res.status_code}"
    health_data = health_res.json()
    assert health_data.get("status") == "ok", f"Expected status 'ok', got {health_data}"
    print(" -> PASS: /health returned:", health_data)

    # 2. Test /upload
    print("\n[2] Testing POST /upload with multipart images...")
    from PIL import Image
    opt_buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(60, 150, 80)).save(opt_buf, format="TIFF")
    opt_buf.seek(0)

    sar_buf = io.BytesIO()
    Image.new("L", (64, 64), color=128).save(sar_buf, format="TIFF")
    sar_buf.seek(0)

    files = {
        "optical": ("test_sentinel2.tif", opt_buf, "image/tiff"),
        "sar": ("test_sentinel1.tif", sar_buf, "image/tiff"),
    }

    upload_res = client.post("/upload", files=files)
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.status_code} {upload_res.text}"
    upload_data = upload_res.json()
    assert "upload_id" in upload_data, "upload_id missing in response"
    assert upload_data["status"] == "success"
    assert upload_data["total_files"] == 2
    upload_id = upload_data["upload_id"]
    print(f" -> PASS: /upload returned upload_id: {upload_id}")
    print(" -> Files manifest:", upload_data["files"])

    # Verify files saved on disk
    saved_optical = Path(upload_data["files"]["optical"]["saved_path"])
    saved_sar = Path(upload_data["files"]["sar"]["saved_path"])
    assert saved_optical.exists(), f"Optical file not found on disk at {saved_optical}"
    assert saved_sar.exists(), f"SAR file not found on disk at {saved_sar}"
    print(f" -> PASS: Verified files exist on disk in {saved_optical.parent}")

    # 3. Test /query
    print("\n[3] Testing POST /query...")
    query_payload = {
        "upload_id": upload_id,
        "query_text": "What is the dominant land cover in this satellite image?"
    }
    query_res = client.post("/query", json=query_payload)
    assert query_res.status_code == 200, f"Query failed: {query_res.status_code} {query_res.text}"
    query_data = query_res.json()
    assert query_data["status"] in ["received", "completed"], f"Unexpected status: {query_data}"
    assert query_data["upload_id"] == upload_id
    print(" -> PASS: /query returned:", query_data["status"], "task:", query_data.get("task"))

    print("\n========================================")
    print(" ALL PHASE 1 TESTS PASSED SUCCESSFULLY!")
    print("========================================")

if __name__ == "__main__":
    run_tests()
