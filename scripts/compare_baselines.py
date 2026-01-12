"""
Compare baseline classifiers (Perspective API, Llama Guard) against TTP and HarmFormer.

Reproduces Table 7 from the paper.

Usage:
    # Compare all baselines on TTP-Eval
    python scripts/compare_baselines.py \
      --data-path data/TTP-Eval/TTPEval.tsv \
      --perspective-key YOUR_API_KEY \
      --limit 100 \
      --output baseline_comparison.json

    # Compare specific baselines
    python scripts/compare_baselines.py \
      --baselines perspective harmformer \
      --perspective-key YOUR_API_KEY \
      --limit 100 \
      --output results.json

    # Test without API (HarmFormer and Llama Guard only)
    python scripts/compare_baselines.py \
      --baselines harmformer llama_guard \
      --limit 10 \
      --output test.json
"""

# For local development, install the package in editable mode: pip install -e .
# Alternatively, set PYTHONPATH to include the src directory.

import sys
from pathlib import Path
import argparse
import json
import logging
import csv
import os
from typing import List, Dict, Any

try:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from tqdm import tqdm  # type: ignore

    from src.data_loaders import TTPEvalLoader, TTPEvalSample
    from src.evaluation.metrics import calculate_metrics
    from src.models import HarmFormer
    from src.baselines import PerspectiveAPI, LlamaGuard
    from src.utils.codecarbon import maybe_track_emissions
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you're running from the project root or install the package with: pip install -e .")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_classifier(
    classifier,
    classifier_name: str,
    samples: List[TTPEvalSample],
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a classifier on TTP-Eval samples.

    Args:
        classifier: Classifier with predict() method
        classifier_name: Name for display
        samples: TTP-Eval samples
        show_progress: Show progress bar

    Returns:
        Dictionary with metrics and predictions
    """
    logger.info(f"Evaluating {classifier_name}...")

    # Get predictions
    predictions = []
    ground_truth = []

    if show_progress:
        iterator = tqdm(samples, desc=f"Evaluating {classifier_name}")
    else:
        iterator = samples

    failed_count = 0
    for sample in iterator:
        try:
            predicted_label = classifier.predict(sample.body)
            predictions.append(predicted_label)
            ground_truth.append(sample.get_harm_label())
        except Exception as e:
            logger.error(f"Error predicting for sample {sample.url}: {e}")
            failed_count += 1
            # Skip this sample entirely to avoid skewing metrics
            continue

    if failed_count > 0:
        logger.warning(f"{failed_count} samples failed prediction and were excluded from evaluation")

    # Calculate metrics
    metrics = calculate_metrics(
        ground_truth=ground_truth,
        predictions=predictions,
        dimension="toxic"  # Focus on toxic detection
    )

    return {
        "classifier": classifier_name,
        "total_samples": len(samples),
        "failed_samples": failed_count,
        "evaluated_samples": len(predictions),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline classifiers on TTP-Eval"
    )
    parser.add_argument(
        "--data-path",
        default="data/TTP-Eval/TTPEval.tsv",
        help="Path to TTP-Eval dataset"
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=["perspective", "llama_guard", "harmformer", "ttp"],
        default=["perspective", "llama_guard", "harmformer"],
        help="Baselines to compare"
    )
    parser.add_argument(
        "--perspective-key",
        help="Perspective API key (required if using perspective)"
    )
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key (required if using ttp)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file"
    )
    parser.add_argument(
        "--device",
        help="Device for models (cuda/cpu/mps)"
    )

    args = parser.parse_args()

    # Validate API keys
    if "perspective" in args.baselines and not args.perspective_key:
        parser.error("--perspective-key required when using perspective baseline")
    if "ttp" in args.baselines and not args.openai_key:
        parser.error("--openai-key required when using ttp baseline")

    # Load dataset
    logger.info(f"Loading TTP-Eval from {args.data_path}...")
    try:
        loader = TTPEvalLoader(args.data_path)
        samples = loader.load()
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {args.data_path}. Error: {e}")
        sys.exit(1)
    except (UnicodeDecodeError, csv.Error, ValueError) as e:
        logger.error(f"Failed to parse TSV dataset file: {args.data_path}. Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error loading dataset from {args.data_path}: {e}")
        sys.exit(1)

    if args.limit:
        samples = samples[:args.limit]

    logger.info(f"Loaded {len(samples)} samples")

    # Initialize classifiers
    classifiers = {}

    if "perspective" in args.baselines:
        logger.info("Initializing Perspective API...")
        classifiers["Perspective API"] = PerspectiveAPI(
            api_key=args.perspective_key
        )

    if "llama_guard" in args.baselines:
        logger.info("Initializing Llama Guard...")
        try:
            classifiers["Llama Guard"] = LlamaGuard(device=args.device)
        except Exception as e:
            # Common cause: gated repo access not granted (even if token is present).
            logger.error(f"Failed to initialize Llama Guard; skipping it. Error: {e}")

    if "harmformer" in args.baselines:
        logger.info("Initializing HarmFormer...")
        classifiers["HarmFormer"] = HarmFormer(device=args.device)

    if "ttp" in args.baselines:
        logger.info("Initializing TTP (GPT-4)...")
        from src.evaluation import TTPEvaluator  # type: ignore
        classifiers["TTP (GPT-4)"] = TTPEvaluator(api_key=args.openai_key)

    # Evaluate each classifier
    results = []
    for name, classifier in classifiers.items():
        with maybe_track_emissions(run_name=f"baseline_{name.replace(' ', '_').lower()}"):
            result = evaluate_classifier(
                classifier=classifier,
                classifier_name=name,
                samples=samples,
                show_progress=True
            )
        results.append(result)

        # Print summary
        print(f"\n{'='*70}")
        print(f"{name} Results")
        print(f"{'='*70}")
        print(f"Total Samples: {result['total_samples']}")
        print("\nOverall Metrics:")
        metrics = result['metrics']['overall']
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print("\nPer-Harm Metrics:")
        for harm in sorted(result['metrics']['per_harm'].keys()):
            harm_metrics = result['metrics']['per_harm'][harm]
            print(f"  {harm}: P={harm_metrics['precision']:.4f}, "
                  f"R={harm_metrics['recall']:.4f}, "
                  f"F1={harm_metrics['f1']:.4f}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("Baseline Comparison (Toxic Detection)")
    print(f"{'='*70}")
    print(f"{'Classifier':<20} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print(f"{'-'*70}")

    for result in results:
        metrics = result['metrics']['overall']
        print(f"{result['classifier']:<20} "
              f"{metrics['precision']:>12.4f} "
              f"{metrics['recall']:>12.4f} "
              f"{metrics['f1']:>12.4f}")

    # Save results
    output_data = {
        "evaluation_config": {
            "dataset": args.data_path,
            "total_samples": len(samples),
            "baselines": list(classifiers.keys()),
        },
        "results": results,
    }

    # Create parent directory for output file
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write output data with error handling
    try:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
    except OSError as e:
        logger.error(f"Failed to write results to {args.output}: {e}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Results saved to {args.output}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
