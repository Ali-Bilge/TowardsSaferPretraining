"""
Helper script to prepare sample JSONL files for analysis.

This is a placeholder showing the expected format. In practice, you would:
1. Download Common Crawl/C4/FineWeb using their respective tools
2. Sample N random documents
3. Convert to JSONL format expected by analyze_dataset.py

Example output format:
{"url": "https://example.com", "text": "Web page content..."}
{"url": "https://example2.com", "text": "More content..."}
"""

import argparse
import json


def create_sample_jsonl(output_file: str, num_samples: int = 100, progress: bool = False) -> None:
    """Create a sample JSONL file for testing."""
    # Input validation
    if not (isinstance(num_samples, int) and not isinstance(num_samples, bool)) or num_samples <= 0:
        raise ValueError(f"num_samples must be an integer greater than 0, got {num_samples}")

    if not output_file or not isinstance(output_file, str):
        raise ValueError("output_file must be a non-empty string")

    print(f"Creating sample JSONL with {num_samples} dummy entries...")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(num_samples):
                if progress and (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{num_samples} samples...")

                sample = {
                    "url": f"https://example.com/page{i}",
                    "text": f"This is sample web page content number {i}. "
                           f"Add your actual web page text here for analysis."
                }
                f.write(json.dumps(sample) + '\n')
    except OSError as e:
        print(f"Error writing to file '{output_file}': {e}")
        raise

    if progress:
        print(f"Completed processing {num_samples} samples.")

    print(f"Created {output_file}")
    print("\nUsage:")
    print(f"  python scripts/analyze_dataset.py --dataset 'Test' --jsonl {output_file} --output results.json")


def main():
    parser = argparse.ArgumentParser(description="Prepare sample JSONL file")
    parser.add_argument("--output", default="samples.jsonl", help="Output JSONL file")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")

    args = parser.parse_args()

    # Enable progress reporting for large sample counts
    progress = args.num_samples > 10000

    create_sample_jsonl(args.output, args.num_samples, progress=progress)


if __name__ == "__main__":
    main()
