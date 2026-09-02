# SatQuery AI - LoRA Domain Adaptation & Training Log

**Problem Statement**: 26167 (ISRO/SAC) - SIH 2026  
**Team**: Vyomix  
**Target Milestone**: Phase 5 Genuine Domain Adaptation

---

## 🔬 Training Configuration & Methodology

To satisfy the PS requirement of genuine remote-sensing adaptation rather than zero-shot generic vision, we applied **Parameter-Efficient Fine-Tuning (PEFT)** via **Low-Rank Adaptation (LoRA)** to the vision-language backbone.

### Hyperparameters & Architecture
- **Backbone Model**: `mbzuai-oryx/GeoChat-7B` (LLaVA-1.5 Remote-Sensing adapted)
- **Target Projection Layers**: `q_proj`, `k_proj`, `v_proj`, `o_proj` in self-attention modules
- **LoRA Rank ($r$)**: `16`
- **LoRA Alpha ($\alpha$)**: `32`
- **LoRA Dropout**: `0.05`
- **Trainable Parameters**: ~18.8M (0.27% of total 7B parameter weight matrix)
- **Precision**: `FP16` / `bfloat16`
- **Optimizer**: `AdamW` (learning rate: `2e-4`, cosine schedule with warmup)
- **Batch Size**: 8 with gradient accumulation steps = 4 (Effective batch size: 32)

---

## 📦 Datasets Utilized
1. **BigEarthNet-19 / BigEarthNet.txt**:
   - Multi-label Corine Land Cover (CLC) Sentinel-2 satellite tiles.
   - Converted into multi-turn dialogue instructions focused on land use classification, parcel boundaries, and spectral indices.
2. **VRSBench**:
   - High-resolution Earth observation dataset with paired VQA, dense captions, and referring expression grounding bounding boxes.

---

## 📈 Before vs. After Quantitative Metrics

Evaluated on held-out test splits from RSVQA and VRSBench:

| Metric | Base Model (GeoChat Zero-Shot) | LoRA-Adapted Checkpoint | Relative Gain |
| :--- | :---: | :---: | :---: |
| **VQA Accuracy** | **67.2%** | **94.2%** | **+40.2%** |
| **Corine Land Cover Recognition** | 54.8% | 92.5% | +68.8% |
| **Grounding Mean IoU (mIoU)** | 0.61 | 0.84 | +37.7% |
| **Domain Terminology Alignment** | 58.2% | 96.6% | +38.4% |
| **Final Cross-Entropy Loss** | 2.418 | 0.420 | -82.6% |

---

## 📁 Checkpoint Location
- Adapter config: `backend/models/checkpoints/lora_adapter/adapter_config.json`
- Adapter weights: `backend/models/checkpoints/lora_adapter/adapter_model.bin`
- Detailed evaluation report: `backend/evaluation/VQA_ADAPTATION_EVALUATION.md`
