import requests
import json
from pathlib import Path

BACKEND_URL = "http://127.0.0.1:8000/analyze"
IMAGE_PATH = Path(r"c:\Users\Tilak M K\OneDrive\Pictures\Desktop\VYOMIX 2026\SatQuery-AI\backend\data\test_scratch\tilak_test_opt.tif")

print(f"[1] Target image exists: {IMAGE_PATH.exists()} ({IMAGE_PATH.stat().st_size} bytes)")

questions = [
    "What is the dominant land cover in this satellite image?",
    "Identify vegetation and built-up structures in this scene.",
    "Describe this satellite imagery."
]

for q in questions:
    print(f"\n==========================================")
    print(f"QUESTION ASKED: '{q}'")
    with open(IMAGE_PATH, "rb") as f:
        files = [("images", ("tilak_test_opt.tif", f, "image/tiff"))]
        data = {"query": q}
        response = requests.post(BACKEND_URL, data=data, files=files, timeout=40)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        res_json = response.json()
        print(f"ANSWER: {res_json.get('answer')}")
        print(f"CONFIDENCE: {res_json.get('confidence')}")
        print(f"MODEL USED: {res_json.get('model')}")
        print(f"TASK DETECTED: {res_json.get('task')}")
        trace = res_json.get("execution_trace", {})
        steps = trace.get("execution_steps", [])
        print(f"STEPS EXECUTED: {len(steps)} steps")
        for s in steps:
            print(f"  - Step: {s.get('specialist')} -> status: {s.get('status')}")
    else:
        print(f"Error Response: {response.text}")
