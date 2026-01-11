"""
Evaluate LLMs on HAVOC benchmark.

Usage:
    # Using HuggingFace Transformers (for HPC clusters)
    python scripts/evaluate_havoc.py \
      --model meta-llama/Llama-2-7b-hf \
      --backend transformers \
      --device cuda \
      --limit 100 \
      --output llama2_7b_results.json

    # Or with smaller model
    python scripts/evaluate_havoc.py \
      --model google/gemma-2-2b \
      --backend transformers \
      --device cuda \
      --limit 100 \
      --output gemma_2b_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
except Exception:
    pass

try:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_loaders import HAVOCLoader
    from src.evaluation import HAVOCEvaluator
    from src.evaluation.llm_generator import create_generator
except ImportError as e:
    attempted_path = Path(__file__).parent.parent
    error_msg = (
        f"Failed to import required modules from src package. "
        f"Attempted to add path: {attempted_path}. "
        f"Please verify the repository layout and ensure the src package exists. "
        f"Original error: {e}"
    )
    print(error_msg)
    sys.exit(1)


def _serialize_samples(results, sample_limit):
    """
    Serialize evaluation results to a list of dictionaries.

    Args:
        results: List of evaluation result objects
        sample_limit: Maximum number of samples to serialize

    Returns:
        List of serialized sample dictionaries
    """
    return [
        {
            "prefix": r.sample.prefix,
            "generated": r.generated_text,
            "leaked": r.leaked,
            "leakage_type": r.leakage_type,
            "response_label": r.response_label.to_dict() if r.response_label else {},
            "error": r.error
        }
        for r in results[:sample_limit]
    ]


def _parse_args():
    """
    Parse command-line arguments for HAVOC evaluation.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate LLM on HAVOC benchmark")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--backend", default="transformers",
                       choices=["transformers"],
                       help="Generation backend (HuggingFace Transformers)")
    parser.add_argument("--data-path", default="data/HAVOC/havoc.tsv",
                       help="Path to HAVOC dataset")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--device", help="Device for HarmFormer/Transformers")
    parser.add_argument(
        "--judge",
        default="ttp",
        choices=["ttp", "harmformer"],
        help="Judge used to label leakage. Paper-faithful Tier-1 uses 'ttp'."
    )
    parser.add_argument("--openai-key", help="OpenAI API key (required if --judge ttp)")
    parser.add_argument("--ttp-model", default="gpt-4o", help="OpenAI model for TTP judge")
    parser.add_argument("--ttp-prompt-path", default="prompts/TTP/TTP.txt", help="Path to TTP prompt")
    # Limit samples saved to output to prevent large file sizes and memory usage
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=100,
        help="Maximum number of samples to save in output JSON (limits file size and memory usage)"
    )

    # Generation settings (Tier-1 defaults: greedy, 200 tokens)
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling generation (NOT paper-faithful; paper uses greedy decoding)"
    )

    args = parser.parse_args()
    if args.judge == "ttp" and not (args.openai_key or os.environ.get("OPENAI_API_KEY")):
        parser.error("--openai-key is required when --judge ttp (or set OPENAI_API_KEY env var)")

    return args


def _run_evaluation(evaluator, generator, model_name, samples, args):
    """
    Run the evaluation with interrupt handling and partial result saving.

    Args:
        evaluator: Initialized evaluator instance
        generator: Initialized generator instance
        model_name: Name of the model being evaluated
        samples: List of samples to evaluate
        args: Parsed command-line arguments

    Returns:
        Evaluation result object

    Raises:
        KeyboardInterrupt: If evaluation is interrupted by user
        SystemExit: If evaluation fails with an error
    """
    print(f"Evaluating {model_name} on HAVOC...")
    result = None
    try:
        result = evaluator.evaluate_samples(
            model_name=model_name,
            samples=samples,
            generate_fn=generator
        )

        # Print results
        evaluator.print_results(result)
        return result
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user. Saving partial results...")
        if result:
            evaluator.print_results(result)
            save_partial_results(result, args)
        raise
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if result:
            print("Saving partial results...")
            evaluator.print_results(result)
            save_partial_results(result, args)
        sys.exit(1)


def _initialize_evaluator(args):
    """
    Initialize the HAVOC evaluator with error handling.

    Args:
        args: Parsed command-line arguments

    Returns:
        Initialized evaluator instance

    Raises:
        SystemExit: If evaluator initialization fails
    """
    if args.judge == "harmformer":
        print("Using HarmFormer as judge (fast, not paper-faithful)...")
        try:
            evaluator = HAVOCEvaluator(judge="harmformer", device=args.device)
        except Exception as e:
            print(f"Error creating HarmFormer evaluator: {e}")
            sys.exit(1)
    else:
        # Validate TTP prompt path
        if not Path(args.ttp_prompt_path).exists():
            print(f"Error: TTP prompt path does not exist: {args.ttp_prompt_path}")
            sys.exit(1)

        print(f"Using TTP ({args.ttp_model}) as judge (paper-faithful Tier-1)...")
        try:
            api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
            evaluator = HAVOCEvaluator(
                judge="ttp",
                api_key=api_key,
                ttp_model=args.ttp_model,
                ttp_prompt_path=args.ttp_prompt_path,
            )
        except Exception as e:
            print(f"Error creating TTP evaluator: {e}")
            sys.exit(1)

    return evaluator


def _initialize_generator(args):
    """
    Initialize the text generator with error handling.

    Args:
        args: Parsed command-line arguments

    Returns:
        Initialized generator instance

    Raises:
        SystemExit: If generator initialization fails
    """
    print(f"Initializing {args.backend} generator with model: {args.model}...")
    try:
        generator = create_generator(
            backend=args.backend,
            model_name=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing generator with backend {args.backend} and model {args.model}: {e}")
        sys.exit(1)

    return generator


def _load_data(data_path, limit):
    """
    Load and validate HAVOC dataset samples.

    Args:
        data_path: Path to the HAVOC dataset file
        limit: Optional limit on number of samples to load

    Returns:
        List of loaded samples

    Raises:
        SystemExit: If data loading fails or samples are invalid
    """
    # Validate data path exists
    if not Path(data_path).exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)

    print(f"Loading HAVOC dataset from {data_path}...")
    try:
        loader = HAVOCLoader(data_path)
        samples = loader.load()
    except Exception as e:
        print(f"Error loading HAVOC dataset: {e}")
        sys.exit(1)

    # Validate samples were loaded
    if not samples or not isinstance(samples, (list, tuple)):
        print("Error: No valid samples loaded from dataset")
        sys.exit(1)

    if limit:
        samples = samples[:limit]

    print(f"Loaded {len(samples)} samples")
    return samples


def save_partial_results(result, args):
    """
    Save partial evaluation results to a temporary file.

    Args:
        result: Evaluation result object containing results
        args: Parsed command-line arguments
    """
    partial_output = {
        "model": args.model,
        "backend": args.backend,
        "samples": _serialize_samples(result.results, args.sample_limit)
    }
    partial_path = f"{args.output}.partial"
    try:
        with open(partial_path, 'w') as f:
            json.dump(partial_output, f, indent=2)
        print(f"Partial results saved to {partial_path}")
    except Exception as save_error:
        print(f"Failed to save partial results: {save_error}")


def main():
    """
    Runs the HavOC evaluation workflow.

    Parses command-line arguments, loads HAVOC dataset, initializes text generator and evaluator,
    runs evaluation on samples using specified judge (TTP or HarmFormer), handles interruptions
    with partial result saving, and outputs complete results to JSON file.

    Takes no arguments and returns exit code.
    """
    args = _parse_args()
    samples = _load_data(args.data_path, args.limit)
    generator = _initialize_generator(args)
    evaluator = _initialize_evaluator(args)
    result = _run_evaluation(evaluator, generator, args.model, samples, args)

    # Save results
    output_data = {
        "evaluation": result.to_dict(),
        "config": {
            "model": args.model,
            "backend": args.backend,
            "judge": args.judge,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
            "total_samples": len(samples)
        },
        "samples": _serialize_samples(result.results, args.sample_limit)
    }

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write output with error handling
    try:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
    except (IOError, OSError, TypeError, ValueError) as e:
        print(f"Error writing results to {args.output}: {e}")
        sys.exit(1)

    print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
