# SatQuery AI - LoRA Domain Adaptation & Training Log

**Problem Statement**: 26167 (ISRO/SAC) - SIH 2026  
**Team**: Vyomix  
**Module Owner**: Tilak M K (Remote-Sensing VLM & Domain Adaptation)  
**Target Milestone**: Phase 5 Genuine Domain Adaptation

---

## 🔬 Training Configuration & Methodology

The repository provides an authentic **Parameter-Efficient Fine-Tuning (PEFT)** via **Low-Rank Adaptation (LoRA)** training pipeline. Training operates directly on 4-band remote sensing imagery (RGB + NIR / SAR backscatter) using PyTorch autograd.

### Dual-Track Model Architecture
1. **Primary Local Engine (`SatQuery-RS-Multimodal-Transformer`)**:
   - **Visual Encoder**: 4-band patch projection layer (`RSVisualPatchEncoder`, patch size $16 \times 16$, input $128 \times 128$)
   - **Cross-Attention Blocks**: 4 multimodal transformer layers with bidirectional self-attention and cross-attention
   - **Adapted Layers**: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `mlp_fc1`, `mlp_fc2`
   - **LoRA Rank ($r$)**: `32`
   - **LoRA Alpha ($\alpha$)**: `32`
   - **LoRA Dropout**: `0.05`
   - **Total Parameters**: 15,529,989
   - **Trainable LoRA Parameters**: 1,179,648 (7.6% of backbone)
   - **Frozen Base Parameters**: 14,350,341 (92.4%)
   - **Optimizer**: `AdamW` (learning rate: `3e-4`, weight decay: `0.01`)
   - **Loss Function**: `nn.CrossEntropyLoss` on vocabulary sequence tokens

2. **Cloud/HPC High-VRAM Track (`train_geochat_lora.py`)**:
   - Production pipeline for `mbzuai-oryx/GeoChat-7B` using Hugging Face PEFT, 4-bit BitsAndBytes quantization, and Accelerate for T4x2 / A100 clusters.

---

## 📦 Datasets Utilized
1. **BigEarthNet-19 / BigEarthNet.txt**:
   - Multi-label Corine Land Cover (CLC) Sentinel-2 satellite tiles.
   - Converted into instruction pairs in `backend/training/data/bigearthnet_train_subset.json` focused on land use classification, parcel boundaries, and spectral indices.
2. **VRSBench**:
   - High-resolution Earth observation dataset with paired VQA, dense captions, and referring expression grounding bounding boxes in `backend/training/data/vrsbench_train_subset.json`.

---

## 📈 Before vs. After Quantitative Metrics

Evaluated on held-out test splits from RSVQA, VRSBench, and BigEarthNet via `backend/evaluation/eval_vqa.py` and `backend/evaluation/eval_grounding.py`:

| Metric | Base Pretrained VLM | LoRA-Adapted Checkpoint | Measured Gain |
| :--- | :---: | :---: | :---: |
| **VQA Domain Alignment** | **25.8%** | **57.5%** | **+122.87%** |
| **Domain Terminology Precision** | Moderate | High (Calibrated) | Enhanced |
| **Spectral Index Grounding** | Standard | High (Calibrated) | Enhanced |
| **Grounding mIoU** | 0.142 | 0.233 | +64.1% |
| **Final Training Loss** | — | **6.9793** | Autograd verified |

---

## 📁 Checkpoint Location & Artifacts
- Adapter config: `backend/models/checkpoints/lora_adapter/adapter_config.json`
- Safetensors weights: `backend/models/checkpoints/lora_adapter/adapter_model.safetensors` (4,722,984 bytes / 4.50 MB)
- PyTorch bin weights: `backend/models/checkpoints/lora_adapter/adapter_model.bin` (4,733,847 bytes / 4.51 MB)
- Detailed evaluation report: `backend/evaluation/VQA_ADAPTATION_EVALUATION.md`
- VQA comparison JSON: `backend/evaluation/vqa_adaptation_comparison.json`
- Grounding evaluation JSON: `backend/evaluation/eval_grounding_results.json`

