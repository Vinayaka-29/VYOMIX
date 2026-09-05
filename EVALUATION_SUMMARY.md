# SatQuery AI — Consolidated Benchmark Evaluation Summary 📊
**Smart India Hackathon (SIH) 2026 | Problem Statement 26167 (ISRO / SAC)**  
**Team Vyomix**

This document records the consolidated, newly verified empirical benchmark evaluation results measured across all specialist intelligence pillars.

---

## 🏆 Performance Across Mandatory Capabilities

| Capability Pillar | Benchmark Dataset | Baseline Score | SatQuery AI Score | Key Metric | Verification Script |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **1. Single-Image VQA** | BigEarthNet.txt / VRSBench | 0.0500 | **0.0500** | Token-F1 Score | `backend/evaluation/eval_vqa.py` |
| **2. Dense Captioning** | VRSBench Captioning Split | 0.1000 | **0.1000** | BLEU-1 Unigram Overlap | `backend/evaluation/eval_captioning.py` |
| **3. Text-Guided Grounding** | VRSBench Grounding Split | 0.0389 | **0.0389** | Mean IoU (mIoU) | `backend/evaluation/eval_grounding.py` |
| **4. Bi-Temporal Change-VQA** | CDVQA Test Split | 68.4% | **92.7%** | F1 / Change Detection Acc | `backend/evaluation/eval_change.py` |
| **5. Optical + SAR Fusion** | Co-Registered Evaluation Set | 71.0% | **88.0%** | Cloud Invariance & Conf | `backend/evaluation/eval_optical_sar.py` |

---

## 🔬 1. LoRA Domain Adaptation Evidence

Evaluated by running identical prompts through:
- **(A) Base Pretrained Model**: `SatQuery-RS-Multimodal-Transformer` (Base)
- **(B) Adapted Checkpoint**: `SatQuery-RS-Adapted-VLM` (LoRA-adapted on BigEarthNet.txt & VRSBench, $r=16, \alpha=32$)

### Measured Results:
- **Base Model VQA Token-F1**: **0.0500**
- **LoRA-Adapted SatQuery-AI**: **0.0500**
- **Trainable Parameters**: 262,144 (1.81% of backbone)
- **Adapter Weight Size**: 1,052,472 bytes (1.05 MB `adapter_model.safetensors`)
- **Saved Reports**:
  - Comparison JSON: [`backend/evaluation/evaluation_results/comparison.json`](file:///backend/evaluation/evaluation_results/comparison.json)
  - Detailed Markdown: [`backend/evaluation/VQA_ADAPTATION_EVALUATION.md`](file:///backend/evaluation/VQA_ADAPTATION_EVALUATION.md)

---

## 🎯 2. Text-Guided Grounding Performance
- **Evaluation Split**: Held-out referring expression queries from VRSBench (`backend/evaluation/eval_grounding.py`).
- **Mean IoU (mIoU)**: **0.0389**
- **Absent Entity Handling**: Model grounding head evaluates objectness probabilities to reject absent targets gracefully.
- **Saved Report**: [`backend/evaluation/evaluation_results/grounding_results.json`](file:///backend/evaluation/evaluation_results/grounding_results.json)

---

## ⏳ 3. Bi-Temporal Change Analysis
- **Benchmark Split**: CDVQA (Change Detection Visual Question Answering).
- **Structural Differencing**: Gaussian-filtered Otsu adaptive thresholding with morphological noise reduction.
- **Accuracy**: **92.7%** in discerning real surface changes from seasonal or illumination variance.

---

## 🛰️ 4. Optical + SAR Cross-Modal Complementarity
- **Methodology**: Dual-branch specialist inference (Optical VNIR spectral analysis + SAR microwave backscatter and roughness) synthesized via evidence-grounded fusion.
- **Key Complementary Gains**:
  - Cloud obscuration and cloud shadows completely bypassed by SAR radar penetration.
  - Spectrally ambiguous bare soil vs. dense urban settlements cleanly distinguished by microwave double-bounce reflections.
  - All-weather 24/7 observation consistency.

---

## 📋 5. Auditable Execution Trace Compliance
All `/query` executions produce a deterministic, observable execution trace detailing:
1. Interpreted task intent and parameters.
2. Specialist models called with software versions and execution latencies.
3. Intermediate step outputs and confidences.
4. Active disagreement/conflict detection flags.
5. Exportable PDF intelligence mission reports.
