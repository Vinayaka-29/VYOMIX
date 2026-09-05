"""
Dataset Preparation for VRSBench Remote-Sensing VLM Adaptation
SIH Problem Statement 26167 | Team Vyomix

Prepares authentic Earth Observation VQA, captioning, and referring-expression
grounding instruction samples from official VRSBench annotations (NeurIPS 2024 Track).
Zero fake synthetic noise rasters. Strict sample validation.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

from training.data_adapters.vrsbench_adapter import VRSBenchAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satquery.prep_vrs")


def prepare_vrsbench_data(mode: str = "smoke") -> str:
    """Prepares and saves authentic VRSBench instruction samples."""
    logger.info(f"Preparing VRSBench instruction dataset in '{mode}' mode...")
    adapter = VRSBenchAdapter()
    samples = adapter.create_instruction_dataset(mode=mode)
    output_path = Path(__file__).resolve().parent / "data" / "vrsbench_train_subset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Saved {len(samples)} authentic VRSBench samples to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VRSBench dataset")
    parser.add_argument("--mode", default="smoke", choices=["smoke", "small", "medium", "full"])
    args = parser.parse_args()
    prepare_vrsbench_data(mode=args.mode)
