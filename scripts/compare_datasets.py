"""
Compare analysis results from multiple datasets.

Usage:
    python scripts/compare_datasets.py results/*.json --output comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to Python path for robust package-relative importing
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.analysis import DatasetAnalyzer, AnalysisResult


def load_analysis_result(filepath: str) -> AnalysisResult:
    """Load AnalysisResult from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Analysis result file not found: {filepath}")
    except PermissionError:
        raise ValueError(f"Permission denied reading analysis result file: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in analysis result file {filepath}: {e}")

    # Validate required top-level structure
    if "analysis" not in data:
        raise ValueError(f"Missing required 'analysis' key in {filepath}")

    analysis_data = data["analysis"]

    # Validate required fields
    required_fields = ["dataset_name", "total_samples", "overall", "harm_counts"]
    for field in required_fields:
        if field not in analysis_data:
            raise ValueError(f"Missing required field '{field}' in analysis data from {filepath}")

    # Validate overall structure
    overall = analysis_data["overall"]
    required_overall_fields = ["toxic", "topical", "safe"]
    for field in required_overall_fields:
        if field not in overall:
            raise ValueError(f"Missing required field 'overall.{field}' in analysis data from {filepath}")

    result = AnalysisResult(
        dataset_name=analysis_data["dataset_name"],
        total_samples=analysis_data["total_samples"],
        toxic_count=analysis_data["overall"]["toxic"],
        topical_count=analysis_data["overall"]["topical"],
        safe_count=analysis_data["overall"]["safe"],
        harm_counts=analysis_data["harm_counts"]
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare dataset analysis results")
    parser.add_argument("results", nargs="+", help="Analysis result JSON files")
    parser.add_argument("--output", help="Output comparison JSON file")

    args = parser.parse_args()

    # Load all results
    results = []
    failed_files = []
    successful_files = []
    for filepath in args.results:
        print(f"Loading {filepath}...")
        try:
            result = load_analysis_result(filepath)
            results.append(result)
            successful_files.append(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            failed_files.append(filepath)

    # Report loading results
    if failed_files:
        print(f"\nFailed to load {len(failed_files)} file(s):")
        for filepath in failed_files:
            print(f"  - {filepath}")
    if successful_files:
        print(f"\nSuccessfully loaded {len(successful_files)} file(s):")
        for filepath in successful_files:
            print(f"  - {filepath}")
    if failed_files:
        print(f"\nWarning: {len(failed_files)} file(s) failed to load. Continuing with {len(successful_files)} successful file(s).")

    # Guard against empty results
    if not results:
        print("\nError: No analysis results could be loaded. Cannot perform comparison.")
        print("Please check that your input files contain valid analysis data.")
        return 1

    # Compare
    comparison = DatasetAnalyzer.compare_datasets(results)

    # Print comparison
    DatasetAnalyzer.print_comparison(comparison)

    # Save if output specified
    write_failed = False
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison saved to {args.output}")
        except (OSError, IOError) as e:
            print(f"Error writing to output file {args.output}: {e}")
            write_failed = True
        except (TypeError, ValueError) as e:
            print(f"Error serializing comparison data to JSON for {args.output}: {e}")
            print("Attempting fallback serialization with default=str...")
            try:
                with open(args.output, 'w') as f:
                    json.dump(comparison, f, default=str, indent=2)
                print(f"Comparison saved to {args.output} using fallback serialization")
            except (TypeError, ValueError, OSError, IOError) as fallback_e:
                print(f"Fallback serialization also failed for {args.output}: {fallback_e}")
                print("Consider serializing problematic objects before writing the comparison.")
                write_failed = True

    return 1 if failed_files or write_failed else 0


if __name__ == "__main__":
    sys.exit(main())
