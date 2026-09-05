# Cloud GPU / HPC Training Guide for GeoChat-7B (Phase 28)
**SIH 2026 Problem Statement 26167 | Team Vyomix**

This directory provides a turnkey execution package for training **MBZUAI/geochat-7B** with 4-bit NormalFloat (NF4) QLoRA on Cloud GPU and High-Performance Computing (HPC) environments.

---

## 🖥️ Minimum Hardware Requirements

| Configuration | Target Model | Minimum VRAM | Recommended Cloud GPU Platform |
| :--- | :--- | :---: | :--- |
| **4-bit QLoRA (NF4)** | `MBZUAI/geochat-7B` | **14 GB** | Google Colab (T4 16GB), Kaggle GPU (T4x2), RunPod RTX 3090 (24GB) |
| **8-bit LoRA** | `MBZUAI/geochat-7B` | **18 GB** | NVIDIA RTX 4090 (24GB), A10G (24GB) |
| **Full FP16 LoRA** | `MBZUAI/geochat-7B` | **28 GB** | NVIDIA A100 (40GB / 80GB), H100 (80GB) |

> [!NOTE]
> **Why Cloud GPU is required for GeoChat-7B:**
> The local machine is equipped with an **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)** and currently has **~1.9 GB free system RAM** with a CPU-only PyTorch build.
> A 7-Billion parameter Vision-Language Model requires ~14 GB of memory just to load model weights in 16-bit precision, or ~5-6 GB for 4-bit quantized inference alone. Running 7B backpropagation locally would trigger an immediate Out-Of-Memory (OOM) operating system crash.

---

## 🚀 Quickstart on Google Colab / Kaggle / Linux GPU Node

### Step 1: Clone Repository & Navigate
```bash
git clone https://github.com/Vinayaka-29/VYOMIX.git
cd VYOMIX/SatQuery-AI/backend
```

### Step 2: Install CUDA & GPU Dependencies
```bash
pip install -r training/cloud_training/requirements_gpu.txt
```

### Step 3: Prepare Authentic Training Datasets
```bash
python training/prepare_bigearthnet.py --mode small
python training/prepare_vrsbench.py --mode small
```

### Step 4: Launch QLoRA Fine-Tuning
```bash
python training/train_geochat_lora.py \
  --base_model MBZUAI/geochat-7B \
  --use_qlora_4bit \
  --lora_rank 16 \
  --lora_alpha 32 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 2e-4
```

### Step 5: Verify Saved LoRA Checkpoint
The trained LoRA adapter weights and PEFT configuration will be saved to:
`backend/models/checkpoints/geochat_lora_adapter/`
- `adapter_config.json`
- `adapter_model.safetensors`
