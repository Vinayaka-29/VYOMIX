"""
LoRA Fine-Tuning Module for Remote Sensing VLM (Phase 5)
Executes Parameter-Efficient Fine-Tuning (PEFT / LoRA) on BigEarthNet & VRSBench
to genuinely adapt the vision-language backbone to Earth Observation tasks.
"""
import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "lora_adapter"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def run_lora_finetuning(
    dataset_name: str = "local-jsonl",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Train a real PEFT adapter from ``TRAINING_DATA`` JSONL samples.

    Each line must contain ``image`` and ``answer`` and may contain ``prompt``.
    Images are passed to the processor and the model computes the training loss.
    """
    data_path = Path(os.getenv("TRAINING_DATA", ""))
    if not data_path.is_file():
        raise FileNotFoundError("Set TRAINING_DATA to a JSONL file containing real image/prompt/answer samples")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoProcessor
    model_name = os.getenv("MODEL_CHECKPOINT", os.getenv("MODEL_NAME", "mbzuai-oryx/GeoChat-7B"))
    seed = int(os.getenv("TRAINING_SEED", "42"))
    random.seed(seed)
    torch.manual_seed(seed)
    with data_path.open("r", encoding="utf-8") as handle:
        samples: List[Dict[str, Any]] = [json.loads(line) for line in handle if line.strip()]
    if not samples:
        raise ValueError(f"Training dataset is empty: {data_path}")
    if any(not sample.get("image") or not sample.get("answer") for sample in samples):
        raise ValueError("Every training sample requires image and answer fields")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    target_modules = [value.strip() for value in os.getenv("LORA_TARGET_MODULES", "q_proj,v_proj").split(",")]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_history: List[Dict[str, Any]] = []

    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            texts = [f"{item.get('prompt', 'Answer the question about this image.')}\n{item['answer']}" for item in batch]
            images = [Image.open(item["image"]).convert("RGB") for item in batch]
            encoded = processor(text=texts, images=images, return_tensors="pt", padding=True)
            encoded = {key: value for key, value in encoded.items() if hasattr(value, "to")}
            labels = encoded["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            output = model(**encoded, labels=labels)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(output.loss.detach().cpu()))
        if not epoch_losses:
            raise RuntimeError("No training batches were produced")
        loss_history.append({"epoch": epoch, "training_loss": sum(epoch_losses) / len(epoch_losses)})

    model.save_pretrained(CHECKPOINT_DIR)
    processor.save_pretrained(CHECKPOINT_DIR)
    metadata = {
        "base_model_name_or_path": model_name,
        "dataset": str(data_path),
        "dataset_name": dataset_name,
        "samples": len(samples),
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "loss_history": loss_history,
        "adapter_path": str(CHECKPOINT_DIR),
    }
    with (CHECKPOINT_DIR / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


if __name__ == "__main__":
    run_lora_finetuning()
