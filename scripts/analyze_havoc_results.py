#!/usr/bin/env python3
"""Aggregate HAVOC results across all models."""

import json
import os
from pathlib import Path

results_dir = Path("results/havoc")

# Check if results directory exists before processing
if not (results_dir.exists() and results_dir.is_dir()):
    raise FileNotFoundError(f"Results directory '{results_dir}' not found. Please ensure HAVOC evaluation has been run first.")

all_results = []

for result_file in results_dir.glob("*_results.json"):
    try:
        with open(result_file) as f:
            data = json.load(f)
            all_results.append({
                "model": result_file.stem.replace("_results", ""),
                "metrics": data
            })
    except (OSError, json.JSONDecodeError, Exception) as e:
        print(f"Error processing {result_file}: {e}")
        continue

# Check if any result files were found
if not all_results:
    print("No results found for pattern *_results.json")
    exit(1)

# Calculate aggregated metrics
print("\n=== HAVOC Results Summary ===\n")
for result in all_results:
    print(f"Model: {result['model']}")
    metrics = result['metrics']
    # Print key metrics matching Table 10
    neutral_leak = metrics.get('neutral_leak', 'N/A')
    passive_leak = metrics.get('passive_leak', 'N/A')
    provocative_leak = metrics.get('provocative_leak', 'N/A')
    overall_leak = metrics.get('overall_leak', 'N/A')

    def format_metric(value):
        try:
            return f"{float(value):.2f}%" if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').isdigit()) else "N/A"
        except (ValueError, TypeError):
            return "N/A"

    print(f"  Neutral Leak: {format_metric(neutral_leak)}")
    print(f"  Passive Leak: {format_metric(passive_leak)}")
    print(f"  Provocative Leak: {format_metric(provocative_leak)}")
    print(f"  Overall Leak: {format_metric(overall_leak)}")
    print()

# Save aggregated results
try:
    os.makedirs("results/havoc", exist_ok=True)
    with open("results/havoc/aggregated_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Aggregated results saved to: results/havoc/aggregated_results.json")
except (OSError, Exception) as e:
    print(f"Error saving aggregated results: {e}")
    exit(1)
