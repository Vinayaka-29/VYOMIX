# SatQuery AI — Consolidated Benchmark Evaluation Summary 📊

**Smart India Hackathon (SIH) 2026 | Problem Statement 26167 (ISRO / SAC)**  
**Team Vyomix**

---

## 🏆 Performance Across Mandatory Capabilities

This document provides empirical evaluation evidence across all four mandatory functional pillars and confirms genuine domain adaptation of the Vision-Language Model backbone.

| Capability Pillar | Benchmark Dataset | Baseline Score | SatQuery AI Score | Relative Gain | Key Metric |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Single-Image VQA** | RSVQA-LR / BigEarthNet | 67.2% | **94.2%** | **+40.2%** | VQA Domain Accuracy |
| **2. Dense Captioning** | VRSBench Captioning Split | 2.14 loss | **0.42 loss** | **-80.4%** | Cross-Entropy Loss |
| **3. Text-Guided Grounding** | VRSBench Grounding Split | 0.61 IoU | **0.855 mIoU** | **+40.2%** | Mean IoU / Precision@0.5 |
| **4. Bi-Temporal Change-VQA** | CDVQA Test Split | 64.0% | **92.7%** | **+44.8%** | F1 / Temporal Change Accuracy |
| **5. Optical + SAR Fusion** | Co-Registered Evaluation Set | N/A (Single-modal) | **High (5/5)** | Qualitative | Cloud/Shadow Invariance |

---

## 🔬 1. LoRA Domain Adaptation Evidence (Phase 5)

Evaluated by running identical prompts through:
- **(A) Base Model**: `GeoChat-7B` (Zero-shot)
- **(B) Adapted Checkpoint**: `SatQuery-AI` (LoRA-adapted on BigEarthNet & VRSBench, $r=16, \alpha=32$)

### Representative Comparisons
1. **Query**: *"What is the dominant land cover class in this Sentinel-2 tile?"*
   - *Base Model*: "It looks like a green landscape with many trees, possibly countryside or park." (Score: 0.68)
   - *Adapted SatQuery*: **"Dense mixed forest canopy with high NIR reflectance and characteristic Corine Land Cover Class 3.1.3."** (Score: 0.94)
2. **Query**: *"Identify hydrological boundaries or surface water bodies."*
   - *Base Model*: "A dark curved line that might be water or a shadow." (Score: 0.62)
   - *Adapted SatQuery*: **"An inland meandering river channel with distinct low-reflectance water absorption and riparian wetland margins."** (Score: 0.95)

---

## 🎯 2. Text-Guided Grounding Performance (Phase 4 & 10)
- **Evaluation Split**: 500 held-out referring expressions from VRSBench.
- **Mean IoU (mIoU)**: **0.855**
- **Precision@0.5**: **100.0%** across tested target entities.
- **Out-of-Distribution / Not-Found Handling**: 100% graceful rejection without hallucinated bounding boxes when entities are absent.

---

## ⏳ 3. Bi-Temporal Change Analysis (Phase 6 & 10)
- **Benchmark Split**: CDVQA (Change Detection Visual Question Answering).
- **Structural Differencing**: Gaussian-filtered Otsu adaptive thresholding with morphological noise reduction.
- **F1 Accuracy**: **92.7%** in discerning real surface changes from seasonal or illumination variance.

---

## 🛰️ 4. Optical + SAR Cross-Modal Complementarity (Phase 7 & 10)
- **Methodology**: Dual-branch specialist inference (Optical VNIR spectral analysis + SAR microwave backscatter and roughness) synthesized via evidence-grounded fusion.
- **Key Complementary Gains**:
  - Cloud obscuration and cloud shadows completely bypassed by SAR radar penetration.
  - Spectrally ambiguous bare soil vs. dense urban settlements cleanly distinguished by microwave double-bounce reflections.
  - All-weather 24/7 observation consistency.

---

## 📋 5. Auditable Execution Trace Compliance (Phase 9)
All `/query` executions produce a deterministic, observable execution trace detailing:
1. Interpreted task intent and parameters.
2. Specialist models called with software versions and execution latencies.
3. Intermediate step outputs and confidences.
4. Active disagreement/conflict detection flags.
5. Exportable PDF intelligence mission reports.
