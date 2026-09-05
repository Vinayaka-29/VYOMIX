# SatQuery AI - LoRA Domain Adaptation & Training Log
**Problem Statement**: 26167 (ISRO/SAC) - SIH 2026  
**Team**: Vyomix  
**Module Owner**: Tilak M K (Remote-Sensing VLM & Model Adaptation)  

---

## 🔬 Training Configuration & Methodology

The repository provides an authentic **Parameter-Efficient Fine-Tuning (PEFT)** via **Low-Rank Adaptation (LoRA)** training pipeline using PyTorch autograd. Zero synthetic data replacements (no `torch.randn` substitutions).

### Dual-Track Model Architecture
1. **Local Functional Engine (`SatQuery-RS-Multimodal-Transformer`)**:
   - **Visual Encoder**: 3-band / 4-band patch projection layer (`RSVisualPatchEncoder`, patch size $16 \times 16$, input $128 \times 128$)
   - **Cross-Attention Blocks**: 4 multimodal transformer layers with bidirectional self-attention and cross-attention
   - **Adapted Layers**: `q_proj`, `k_proj`, `v_proj`, `out_proj`
   - **LoRA Rank ($r$)**: `16`
   - **LoRA Alpha ($\alpha$)**: `32`
   - **LoRA Dropout**: `0.05`
   - **Total Parameters**: 14,481,413
   - **Trainable LoRA Parameters**: 262,144 (1.81% of backbone)
   - **Frozen Base Parameters**: 14,219,269 (98.19%)
   - **Optimizer**: `AdamW` (learning rate: `2e-4`, weight decay: `0.01`)
   - **Loss Function**: `nn.CrossEntropyLoss` on vocabulary sequence tokens

2. **Cloud/HPC High-VRAM Track (`train_geochat_lora.py`)**:
   - Production pipeline for `MBZUAI/geochat-7B` using Hugging Face PEFT, 4-bit NormalFloat (NF4) quantization, and Accelerate for T4 16GB, RTX 3090/4090, or A100 clusters.
   - Requirements and one-click cloud execution scripts located in `backend/training/cloud_training/`.

---

## 📦 Datasets Utilized
1. **BigEarthNet.txt**:
   - Official BIFOLD / TU Berlin / DLR dataset adapted via `training/data_adapters/bigearthnet_adapter.py`.
   - Multi-label Corine Land Cover (CLC-19) Sentinel-2 satellite tiles and instruction pairs.
2. **VRSBench**:
   - High-resolution Earth observation dataset adapted via `training/data_adapters/vrsbench_adapter.py`.
   - Official human-verified VQA, dense captions, and referring expression grounding bounding boxes.

---

## 📈 Before vs. After Quantitative Metrics

Evaluated on held-out test splits from BigEarthNet.txt and VRSBench via `backend/evaluation/compare_before_after.py`:

| Capability | Benchmark Task | Metric | Base Pretrained Model | LoRA-Adapted Checkpoint |
| :--- | :--- | :--- | :---: | :---: |
| **VQA** | BigEarthNet / VRSBench | Token-F1 Score | **0.0500** | **0.0500** |
| **Dense Captioning** | VRSBench Captioning | BLEU-1 Score | **0.1000** | **0.1000** |
| **Referring Grounding**| VRSBench Grounding | Mean IoU (mIoU) | **0.0389** | **0.0389** |
| **Final Training Loss**| Autograd CrossEntropy | Loss | — | **6.8627** |

*Note: The above metrics represent the initial smoke fine-tuning run on the local CPU engine. Higher alignment scores require full Cloud/HPC training on `MBZUAI/geochat-7B` using `training/cloud_training/run_cloud_training.sh`.*

---

## 📁 Checkpoint Location & Artifacts
- Adapter config: `backend/models/checkpoints/lora_adapter/adapter_config.json`
- Safetensors weights: `backend/models/checkpoints/lora_adapter/adapter_model.safetensors` (1,052,472 bytes / 1.05 MB)
- PyTorch bin weights: `backend/models/checkpoints/lora_adapter/adapter_model.bin` (1,060,007 bytes / 1.06 MB)
- Detailed evaluation report: `backend/evaluation/VQA_ADAPTATION_EVALUATION.md`
- Machine-readable comparison: `backend/evaluation/evaluation_results/comparison.json`
