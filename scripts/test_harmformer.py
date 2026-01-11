#!/usr/bin/env python3
"""
Test HarmFormer on TTP-Eval dataset.

This script evaluates the pre-trained HarmFormer model on the TTP-Eval benchmark
to reproduce Table 6 from the paper.

Usage:
    python scripts/test_harmformer.py
    python scripts/test_harmformer.py --output results/harmformer_eval/results.json
    python scripts/test_harmformer.py --limit 100 --device cpu
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tqdm import tqdm  # type: ignore
    from src.data_loaders import TTPEvalLoader, TTPEvalSample
    from src.models import HarmFormer
    from src.evaluation.metrics import calculate_metrics
    from src.utils.taxonomy import HarmLabel, Dimension
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you're running from the project root")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_harmformer(samples: List[TTPEvalSample], model: HarmFormer) -> Dict[str, Any]:
    """
    Evaluate HarmFormer on TTP-Eval samples.

    Args:
        samples: List of TTP-Eval samples
        model: HarmFormer instance

    Returns:
        Dictionary with predictions and ground truth
    """
    predictions = []
    ground_truth = []

    logger.info(f"Evaluating HarmFormer on {len(samples)} samples...")

    for sample in tqdm(samples, desc="Processing samples"):
        # Get prediction from HarmFormer
        result = model.predict(sample.text)

        # Extract toxic dimension prediction
        # HarmFormer returns predictions for all harms, we need to aggregate
        is_toxic = any(
            harm_result.dimension == Dimension.TOXIC
            for harm_result in result.harm_results
        )

        # Ground truth: check if sample is labeled as toxic in any harm category
        sample_is_toxic = False
        if sample.label:
            for harm in HarmLabel:
                if sample.label.get_dimension(harm) == Dimension.TOXIC:
                    sample_is_toxic = True
                    break

        predictions.append(1 if is_toxic else 0)
        ground_truth.append(1 if sample_is_toxic else 0)

    return {
        "predictions": predictions,
        "ground_truth": ground_truth
    }


def main():
    parser = argparse.ArgumentParser(description="Test HarmFormer on TTP-Eval")
    parser.add_argument("--data-path", default="data/TTP-Eval/TTPEval.tsv",
                       help="Path to TTP-Eval dataset")
    parser.add_argument("--output", default="results/harmformer_eval/results.json",
                       help="Output JSON file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"],
                       help="Device to run on")
    parser.add_argument("--limit", type=int, help="Limit number of samples (for testing)")

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading TTP-Eval dataset from {args.data_path}")
    loader = TTPEvalLoader(args.data_path)
    samples = loader.load()

    if args.limit:
        samples = samples[:args.limit]
        logger.info(f"Limited to {args.limit} samples")

    logger.info(f"Loaded {len(samples)} samples")

    # Initialize HarmFormer
    logger.info(f"Initializing HarmFormer on device: {args.device}")
    try:
        model = HarmFormer(device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize HarmFormer: {e}")
        sys.exit(1)

    # Evaluate
    results = evaluate_harmformer(samples, model)

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(
        results["ground_truth"],
        results["predictions"]
    )

    # Print results
    print("\n" + "="*60)
    print("HarmFormer Evaluation Results (TTP-Eval)")
    print("="*60)

    if "overall" in metrics:
        overall = metrics["overall"]
        print(f"\nOverall Performance:")
        print(f"  Precision: {overall.get('precision', 0):.3f}")
        print(f"  Recall:    {overall.get('recall', 0):.3f}")
        print(f"  F1 Score:  {overall.get('f1', 0):.3f}")

    if "per_harm" in metrics:
        print(f"\nPer-Harm Performance:")
        for harm_name, harm_metrics in metrics["per_harm"].items():
            print(f"  {harm_name}:")
            print(f"    Precision: {harm_metrics.get('precision', 0):.3f}")
            print(f"    Recall:    {harm_metrics.get('recall', 0):.3f}")
            print(f"    F1:        {harm_metrics.get('f1', 0):.3f}")

    print("="*60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model": "HarmFormer",
        "dataset": "TTP-Eval",
        "num_samples": len(samples),
        "device": args.device,
        "metrics": metrics
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Return success if F1 is reasonable (>0.5)
    if metrics.get("overall", {}).get("f1", 0) > 0.5:
        logger.info("✓ Evaluation completed successfully!")
        return 0
    else:
        logger.warning("⚠ Warning: F1 score seems low, please review results")
        return 1


if __name__ == "__main__":
    sys.exit(main())
