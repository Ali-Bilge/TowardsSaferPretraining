#!/usr/bin/env python3
"""Sample datasets for analysis (100K samples each to stay in budget)."""

from datasets import load_dataset  # type: ignore
import json
from pathlib import Path
from tqdm import tqdm  # type: ignore

def sample_dataset(dataset_name, dataset_config, output_name, n_samples):
    """Generic helper to sample from a dataset."""
    print(f"Sampling {n_samples} from {output_name}...")

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    # Save
    output_file = Path(f"datasets/samples/{output_name}_{n_samples//1000}k.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for i, item in enumerate(tqdm(dataset, total=n_samples)):
            if i >= n_samples:
                break
            sample = {
                "url": item.get("url", f"{output_name}_sample_{i}"),
                "text": item["text"][:4096]  # Truncate to 4K chars
            }
            f.write(json.dumps(sample) + "\n")

    print(f"Saved to {output_file}")

def sample_c4(n_samples=100000):
    """Sample from C4 dataset."""
    sample_dataset("allenai/c4", "en", "c4", n_samples)

def sample_fineweb(n_samples=100000):
    """Sample from FineWeb dataset."""
    sample_dataset("HuggingFaceFW/fineweb", "CC-MAIN-2024-10", "fineweb", n_samples)

if __name__ == "__main__":
    sample_c4(100000)
    sample_fineweb(100000)
    print("\nDataset sampling complete!")
