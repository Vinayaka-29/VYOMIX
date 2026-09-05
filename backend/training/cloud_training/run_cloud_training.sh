#!/usr/bin/env bash
# ==============================================================================
# GeoChat-7B QLoRA Automated Training Script
# SIH 2026 Problem Statement 26167 | Team Vyomix
# ==============================================================================
set -e

echo "=== SatQuery AI: Launching GeoChat-7B QLoRA Fine-Tuning ==="
nvidia-smi

# 1. Install dependencies
pip install --upgrade pip
pip install -r training/cloud_training/requirements_gpu.txt

# 2. Prepare real datasets
echo "=== Preparing BigEarthNet.txt and VRSBench datasets ==="
python training/prepare_bigearthnet.py --mode small
python training/prepare_vrsbench.py --mode small

# 3. Execute training
echo "=== Running GeoChat-7B QLoRA Domain Adaptation ==="
python training/train_geochat_lora.py \
    --base_model MBZUAI/geochat-7B \
    --use_qlora_4bit \
    --lora_rank 16 \
    --lora_alpha 32 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-4

echo "=== Training Complete. Validating adapter files ==="
ls -lh models/checkpoints/geochat_lora_adapter/
