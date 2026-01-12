"""
Script to evaluate TTP on TTP-Eval dataset.

Usage:
    python scripts/evaluate_ttp.py --api-key sk-... --output results.json
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv  # type: ignore
    # Prefer .env, but fall back to example.env (repo ships example.env, .env may be blocked by policies).
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if not env_path.exists():
        env_path = project_root / "example.env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # Silently skip when python-dotenv is not installed
    pass
except (OSError, UnicodeDecodeError) as e:
    # Warn about file/IO errors when loading .env file
    warnings.warn(f"Failed to load .env file: {e}", UserWarning)

from src.data_loaders import TTPEvalLoader
from src.clients import OpenAITTPClient
from src.benchmarks import calculate_metrics, print_metrics
from src.clients.ttp_openai import TTPResult
from src.utils.taxonomy import HarmLabel
from src.utils.codecarbon import maybe_track_emissions


def main():
    parser = argparse.ArgumentParser(description="Evaluate TTP on TTP-Eval dataset")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--data-path", default="data/TTP-Eval/TTPEval.tsv", help="Path to TTP-Eval dataset")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--limit", type=int, help="Limit number of samples to evaluate")
    parser.add_argument("--dimension", default="toxic", choices=["toxic", "topical", "all"],
                       help="Which dimension to evaluate")

    args = parser.parse_args()

    # Validate API key
    if not (args.api_key or os.environ.get("OPENAI_API_KEY")):
        parser.error("OpenAI API key is required (provide --api-key or set OPENAI_API_KEY environment variable)")

    # Validate data path
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        sys.exit(1)

    print(f"Loading TTP-Eval dataset from {args.data_path}...")
    try:
        loader = TTPEvalLoader(args.data_path)
        samples = loader.load()
    except Exception as e:
        print(f"Error loading TTP-Eval dataset: {e}")
        sys.exit(1)

    if args.limit:
        samples = samples[:args.limit]

    print(f"Evaluating {len(samples)} samples with {args.model}...")

    evaluator = OpenAITTPClient(api_key=args.api_key, model=args.model)

    # Evaluate
    results = []
    with maybe_track_emissions(run_name="evaluate_ttp"):
        with tqdm(total=len(samples), desc="Evaluating samples", unit="sample") as pbar:
            for i, sample in enumerate(samples):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = evaluator.evaluate(sample.url, sample.body)
                        results.append(result)
                        pbar.update(1)
                        pbar.set_postfix({
                            'sample': f'{i+1}/{len(samples)}',
                            'status': 'success',
                            'attempts': attempt + 1
                        })
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            pbar.set_postfix({
                                'sample': f'{i+1}/{len(samples)}',
                                'status': 'retrying',
                                'attempts': attempt + 1
                            })
                            print(f"Warning: Evaluation failed for sample {i} (attempt {attempt + 1}/{max_retries}): {e}")
                            print(f"Sample identifier: {getattr(sample, 'url', 'unknown')}")
                            print("Retrying in 5 seconds...")
                            time.sleep(5)
                        else:
                            pbar.set_postfix({
                                'sample': f'{i+1}/{len(samples)}',
                                'status': 'failed',
                                'attempts': attempt + 1
                            })
                            print(f"Error: Evaluation failed for sample {i} after {max_retries} attempts: {e}")
                            print(f"Sample identifier: {getattr(sample, 'url', 'unknown')}")
                            # Append a failure marker or partial result (fail-open label, keep error)
                            failure_result = TTPResult(
                                url=getattr(sample, "url", "unknown"),
                                body=getattr(sample, "body", ""),
                                predicted_label=HarmLabel(),
                                reasoning="Evaluation failed after retries",
                                error=str(e),
                            )
                            results.append(failure_result)
                            pbar.update(1)
                            print("Continuing with next sample...")

    # Calculate metrics
    predictions = [r.predicted_label for r in results]
    ground_truth = [s.get_harm_label() for s in samples]

    metrics = calculate_metrics(predictions, ground_truth, dimension=args.dimension)
    print_metrics(metrics)

    # Print API stats
    stats = evaluator.get_stats()
    print("\nAPI Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Failed requests: {stats['failed_requests']}")
    print(f"  Total tokens: {stats['total_tokens']}")

    # Save results
    if args.output:
        output_data = {
            "metrics": metrics,
            "stats": stats,
            "results": [
                {
                    "url": r.url,
                    "predicted": r.predicted_label.to_dict(),
                    "ground_truth": gt.to_dict(),
                    "reasoning": r.reasoning,
                    "error": r.error
                }
                for r, gt in zip(results, ground_truth)
            ]
        }

        try:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")
        except Exception as e:
            print(f"Error writing results to {args.output}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
