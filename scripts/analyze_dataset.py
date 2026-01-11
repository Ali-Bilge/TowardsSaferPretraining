"""
Analyze web-scale datasets for harm prevalence.

Usage:
    # From text file with URLs
    python scripts/analyze_dataset.py --dataset "Common Crawl" --urls urls.txt --output results.json

    # From JSONL file with text content
    python scripts/analyze_dataset.py --dataset "C4" --jsonl data.jsonl --text-field text --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
except Exception:
    pass

from src.analysis import DatasetAnalyzer


def load_samples_from_urls(url_file: str, limit: Optional[int] = None) -> list[tuple[str, str]]:
    """Load samples from URL file (placeholder - would need web scraping)."""
    print("Note: URL loading not implemented. Please provide JSONL with text content.")
    return []


def load_samples_from_jsonl(
    jsonl_file: str,
    text_field: str = "text",
    url_field: str = "url",
    limit: Optional[int] = None
) -> list[tuple[str, str]]:
    """Load samples from JSONL file."""
    samples = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            if limit and i > limit:
                break

            try:
                data = json.loads(line)
            except (json.JSONDecodeError, ValueError) as e:
                truncated_line = line.strip()[:100] + ("..." if len(line.strip()) > 100 else "")
                print(f"Warning: Failed to parse JSON at line {i}: {e}")
                print(f"Offending line: {truncated_line}")
                continue

            url = data.get(url_field, f"sample_{i}")
            text = data.get(text_field, "")

            if text:
                samples.append((url, text))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset for harm prevalence")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., 'Common Crawl')")

    # Model selection
    parser.add_argument("--use-ttp", action="store_true",
                       help="Use TTP (GPT-4) instead of HarmFormer (requires API key, slower, costs money)")
    parser.add_argument("--api-key", help="OpenAI API key for TTP (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o", help="Model to use for TTP")
    parser.add_argument("--device", help="Device for HarmFormer (cuda/cpu/mps)")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--urls", help="Text file with URLs (one per line)")
    input_group.add_argument("--jsonl", help="JSONL file with samples")

    # JSONL options
    parser.add_argument("--text-field", default="text", help="Field name for text in JSONL")
    parser.add_argument("--url-field", default="url", help="Field name for URL in JSONL")

    # Limits and output
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    # Load samples
    if args.urls:
        print(f"Loading samples from URL file: {args.urls}")
        samples = load_samples_from_urls(args.urls, limit=args.limit)
    else:
        print(f"Loading samples from JSONL: {args.jsonl}")
        samples = load_samples_from_jsonl(
            args.jsonl,
            text_field=args.text_field,
            url_field=args.url_field,
            limit=args.limit
        )

    if not samples:
        print("Error: No samples loaded")
        return 1

    print(f"Loaded {len(samples)} samples")

    # Analyze
    if args.use_ttp:
        print("Using TTP (GPT-4) for analysis...")
        analyzer = DatasetAnalyzer(
            use_harmformer=False,
            api_key=args.api_key,
            model=args.model
        )
    else:
        print("Using HarmFormer for analysis...")
        analyzer = DatasetAnalyzer(
            use_harmformer=True,
            device=args.device
        )

    result = analyzer.analyze_samples(args.dataset, samples)

    # Print results
    analyzer.print_results(result)

    # Print stats
    stats = analyzer.get_stats()
    if args.use_ttp:
        print("\nAPI Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Failed requests: {stats['failed_requests']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Estimated cost: ${stats['estimated_cost_usd']:.2f}")
    else:
        print(f"\nModel: {stats['model']}")
        print(f"Device: {stats['device']}")

    # Save results
    output_data = {
        "analysis": result.to_dict(),
        "stats": stats,
        "samples": [
            {
                "url": r.url,
                "predicted": r.predicted_label.to_dict(),
                "reasoning": r.reasoning,
                "error": r.error
            }
            for r in result.ttp_results
        ]
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
