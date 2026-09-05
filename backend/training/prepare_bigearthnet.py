"""
Dataset Preparation for BigEarthNet Remote-Sensing VLM Adaptation
SIH Problem Statement 26167 | Team Vyomix

Prepares authentic remote-sensing instruction pairs and metadata derived from
the official BigEarthNet.txt benchmark (BIFOLD / TU Berlin / DLR).
Zero fake synthetic noise rasters. Strict sample validation.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

from training.data_adapters.bigearthnet_adapter import BigEarthNetAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satquery.prep_ben")


def prepare_bigearthnet_data(mode: str = "smoke") -> str:
    """Prepares and saves authentic BigEarthNet instruction samples."""
    logger.info(f"Preparing BigEarthNet instruction dataset in '{mode}' mode...")
    adapter = BigEarthNetAdapter()
    samples = adapter.parse_instruction_subset(mode=mode)
    output_path = Path(__file__).resolve().parent / "data" / "bigearthnet_train_subset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Saved {len(samples)} authentic BigEarthNet samples to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BigEarthNet dataset")
    parser.add_argument("--mode", default="smoke", choices=["smoke", "small", "medium", "full"])
    args = parser.parse_args()
    prepare_bigearthnet_data(mode=args.mode)
