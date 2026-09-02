# SatQuery AI 🛰️

**Agentic Multi-Modal Remote Sensing Vision-Language Model Platform**  
*Smart India Hackathon (SIH) 2026 | Problem Statement 26167 (ISRO / Space Applications Centre)*  
*Developed by Team Vyomix*

---

## 📌 Problem Statement Overview (PS 26167)
Geospatial and satellite analysis requires extracting insights across diverse multi-modal sensors (Optical VNIR, Synthetic Aperture Radar / SAR) and multi-temporal sequences. Traditional computer vision pipelines are rigid and domain-specific. 

**SatQuery AI** bridges natural language and multi-modal Earth Observation by unifying:
1. **Single-Image Visual Question Answering (VQA)** (e.g. land-cover queries, object counting).
2. **Text-Guided Region Grounding & Dense Captioning**.
3. **Bi-Temporal Change Detection & Change-VQA** (pre/post disaster or urban growth).
4. **Optical + SAR Cross-Modal Fusion** (all-weather penetration + spectral discrimination).
5. **Agentic Query Auto-Routing & Auditable Execution Trace**.

---

## 🚀 Phase 1: Repo Scaffold, Containerization & Upload Plumbing

This repository represents **Phase 1** of the 10-phase architecture. It establishes the full-stack skeleton, proving end-to-end multipart upload plumbing, disk storage, and query stub execution before integrating heavyweight AI backbones.

### Technology Stack
- **Frontend**: React 18, Vite 5, Tailwind CSS, Lucide-React
- **Backend**: Python FastAPI, Uvicorn, Python-Multipart, Pydantic
- **Database**: PostgreSQL 16 with PostGIS 3.4
- **Containerization**: Docker & Docker Compose

---

## 🛠️ How to Run

### Method 1: Using Docker Compose (Recommended for Containerized Environments)

Ensure Docker and Docker Compose are installed, then run from the `SatQuery-AI` directory:

```bash
docker-compose up --build
```

- **Frontend UI**: [http://localhost:5173](http://localhost:5173)
- **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **PostGIS Database**: `localhost:5432` (`satquery_db`)

---

### Method 2: Running Locally on Host (No Docker Required)

If running directly on Windows/Linux without Docker:

#### 1. Start the FastAPI Backend
Open a terminal in `SatQuery-AI/backend`:
```powershell
# Install dependencies
pip install -r requirements.txt

# Start backend server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The backend will run at `http://localhost:8000`. Health check: `http://localhost:8000/health`.

#### 2. Start the React Frontend
Open a second terminal in `SatQuery-AI/frontend`:
```powershell
# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🧪 Testing Phase 1 End-to-End

1. Open [http://localhost:5173](http://localhost:5173) in your browser.
2. Check the header to confirm the green pulse **API Connected :8000**.
3. In **Sensor Imagery Input Slots**, click to attach an image to any slot (e.g. Optical or SAR).
4. Enter a query in the **Geospatial Query** box or click one of the quick prompt chips.
5. Click **Analyze Satellite Imagery**.
6. **Verify Results**:
   - Files are uploaded and stored under `backend/data/uploads/<upload_id>/`.
   - The UI shows the received `upload_id` and file manifest.
   - The backend `/query` stub responds with `{"status": "received", "task": "not_yet_implemented"}` and is cleanly rendered in the **Results Dashboard**.

---

## 🗺️ Roadmap Ahead
- **Phase 2**: Input Validation, GeoTIFF CRS/Resolution Extraction, Modality Auto-Detection (rasterio/GDAL).
- **Phase 3**: Single-Image VQA & Captioning Baseline (Pretrained Remote-Sensing VLM / GeoChat).
- **Phase 4**: Text-Guided Referring Expression Grounding & Bounding Box Overlays.
- **Phase 5**: LoRA Fine-Tuning on BigEarthNet / VRSBench.
- **Phase 6**: Bi-Temporal Change Detection & Change-VQA.
- **Phase 7**: Optical + SAR Cross-Modal Analysis (Dual-Branch + LLM Fusion).
- **Phase 8**: Agentic Controller: Intent Interpretation, Task Routing & Planner.
- **Phase 9**: Evidence Fusion, Confidence Scoring, and Auditable Execution Trace.
- **Phase 10**: Full Evaluation Benchmark, PDF Reports, and Final Demo Preparation.
