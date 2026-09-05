# Remote Sensing VLM LoRA Domain Adaptation Evaluation Report
**SIH 2026 | Problem Statement 26167 (ISRO / SAC) | Team Vyomix**

This document records the empirical results measured across the Remote Sensing Vision-Language Model
subsystem, comparing the **Base Pretrained Backbone** against the **LoRA-Adapted Checkpoint**.

- **Timestamp**: `2026-09-05T17:16:46Z`
- **Execution Device**: `cpu` (`Host CPU`)
- **CUDA Available**: `False`
- **Adapter Directory**: `C:\Users\Tilak M K\OneDrive\Pictures\Desktop\VYOMIX 2026\SatQuery-AI\backend\models\checkpoints\lora_adapter`

---

## 📊 Measured Benchmark Performance Summary

| Capability / Benchmark Task | Metric | Base Pretrained Model | LoRA-Adapted Checkpoint | Measured Gain |
| :--- | :--- | :---: | :---: | :---: |
| **Visual Question Answering (VQA)** | Token-F1 Score | **0.05** | **0.05** | **+0.0%** |
| **Dense Scene Captioning** | BLEU-1 Unigram Overlap | **0.1** | **0.1** | **+0.0%** |
| **Referring Expression Grounding** | Mean IoU (mIoU) | **0.0389** | **0.0389** | Evaluated |
| **Absent Entity Rejection Rate** | Detection Rejection | N/A | **0.0%** | Verified |

---

## 🔬 Qualitative VQA Sample Comparisons

### Sample 1: "What is the dominant land cover class in this Sentinel-2 tile?"
- **Ground Truth Target**: *"coniferous and mixed forest vegetation canopy"*
- **Base Model Prediction**: `"Patterns"` (Token-F1: `0.0`, Conf: `0.05`)
- **Adapted Model Prediction**: `"Patterns"` (Token-F1: `0.0`, Conf: `0.05`)

---
### Sample 2: "Are there industrial units or commercial structures present?"
- **Ground Truth Target**: *"industrial commercial structures and paved impervious surfaces"*
- **Base Model Prediction**: `"Patterns"` (Token-F1: `0.0`, Conf: `0.05`)
- **Adapted Model Prediction**: `"Patterns"` (Token-F1: `0.0`, Conf: `0.05`)

---
### Sample 3: "Identify hydrological surface water bodies."
- **Ground Truth Target**: *"inland water river channel surface water"*
- **Base Model Prediction**: `"Sclerophyllous patterns"` (Token-F1: `0.0`, Conf: `0.05`)
- **Adapted Model Prediction**: `"Sclerophyllous patterns"` (Token-F1: `0.0`, Conf: `0.05`)

---
### Sample 4: "Assess agricultural cropland and pasture parcels."
- **Ground Truth Target**: *"arable land pastures and complex cultivation patterns"*
- **Base Model Prediction**: `"Backscatter patterns patterns"` (Token-F1: `0.2`, Conf: `0.05`)
- **Adapted Model Prediction**: `"Backscatter patterns patterns"` (Token-F1: `0.2`, Conf: `0.05`)

---

## 🛰️ Verification Artifacts
- Comparison JSON: [`backend/evaluation/evaluation_results/comparison.json`](./evaluation_results/comparison.json)
- Before LoRA JSON: [`backend/evaluation/evaluation_results/before_lora.json`](./evaluation_results/before_lora.json)
- After LoRA JSON: [`backend/evaluation/evaluation_results/after_lora.json`](./evaluation_results/after_lora.json)
- Grounding Results: [`backend/evaluation/evaluation_results/grounding_results.json`](./evaluation_results/grounding_results.json)
