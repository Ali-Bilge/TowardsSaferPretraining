#!/usr/bin/env python3
"""
Unified evaluator for TTP-Eval.

This replaces multiple overlapping scripts by supporting multiple "setups" in one place:
- openai_ttp (TTP prompt + OpenAI Chat Completions)
- gemini_ttp (TTP prompt + Gemini API)
- perspective (Perspective Comment Analyzer; paper-faithful Table 4 behavior by default)
- harmformer (local HF model)
- llama_guard (local HF model; defaults to Llama Guard 3)
- local_ttp (TTP prompt + local Transformers CausalLM; for Gemma/R1 rows)

Output schema is consistent across setups and is meant for report generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loaders import TTPEvalLoader, TTPEvalSample  # noqa: E402
from src.benchmarks.metrics import calculate_metrics  # noqa: E402
from src.clients import PerspectiveAPI, LlamaGuard, TransformersTTPClient, OpenRouterTTPClient  # noqa: E402
from src.clients.ttp_openai import OpenAITTPClient  # noqa: E402
from src.clients.ttp_gemini import GeminiTTPEvaluator  # noqa: E402
from src.models import HarmFormer  # noqa: E402
from src.utils.codecarbon import maybe_track_emissions  # noqa: E402
from src.utils.taxonomy import HarmLabel  # noqa: E402


def _evaluate_setup(name: str, clf, samples: List[TTPEvalSample], dimension: str) -> Dict[str, Any]:
    preds: List[HarmLabel] = []
    gts: List[HarmLabel] = []
    failed = 0
    for s in samples:
        try:
            preds.append(clf.predict(s.body))
            gts.append(s.get_harm_label())
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"[{name}] sample failed: {e}", file=sys.stderr)
            continue
    metrics = calculate_metrics(predictions=preds, ground_truth=gts, dimension=dimension)
    return {
        "setup": name,
        "total_samples": len(samples),
        "failed_samples": failed,
        "evaluated_samples": len(preds),
        "metrics": metrics,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Unified evaluator for TTP-Eval")
    p.add_argument("--data-path", default="data/TTP-Eval/TTPEval.tsv")
    p.add_argument("--limit", type=int)
    p.add_argument("--dimension", default="toxic", choices=["toxic", "topical", "all"])
    p.add_argument("--output", required=True, help="Output JSON file")

    p.add_argument(
        "--setups",
        nargs="+",
        # Default to local-only setups to avoid API costs/rate limits unless explicitly requested.
        default=["harmformer"],
        choices=["perspective", "openai_ttp", "openrouter_ttp", "gemini_ttp", "harmformer", "llama_guard", "local_ttp"],
        help="Which setups to run",
    )

    # Perspective
    p.add_argument("--perspective-key", help="Perspective API key (or set PERSPECTIVE_API_KEY)")
    p.add_argument("--perspective-threshold", type=float, default=0.4)
    p.add_argument("--perspective-chunk-chars", type=int, default=500)

    # OpenAI
    p.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY)")
    p.add_argument("--openai-model", default="gpt-4o")

    # OpenRouter (OpenAI-compatible)
    p.add_argument("--openrouter-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    p.add_argument("--openrouter-model", default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o"))
    p.add_argument("--openrouter-referer", default=os.environ.get("OPENROUTER_REFERER"), help="Optional HTTP-Referer header")
    p.add_argument("--openrouter-title", default=os.environ.get("OPENROUTER_TITLE"), help="Optional X-Title header")

    # Gemini
    p.add_argument("--gemini-key", help="Gemini API key (or set GEMINI_API_KEY)")
    p.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))

    # Local models
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--local-model", action="append", default=[], help="HF model id for local_ttp (repeatable)")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    p.add_argument("--quantization", default="none", choices=["none", "8bit", "4bit"])

    args = p.parse_args()

    samples = TTPEvalLoader(args.data_path).load()
    if args.limit:
        samples = samples[: args.limit]

    setups: List[Tuple[str, Any]] = []

    if "perspective" in args.setups:
        perspective_key = args.perspective_key or os.environ.get("PERSPECTIVE_API_KEY")
        if not perspective_key and os.environ.get("ENABLE_PERSPECTIVE_WITH_GEMINI_KEY") == "1":
            perspective_key = os.environ.get("GEMINI_API_KEY")
        if not perspective_key:
            raise SystemExit("Perspective setup selected but no key provided.")
        setups.append(
            (
                "Perspective API",
                PerspectiveAPI(
                    api_key=perspective_key,
                    mode="paper_table4",
                    paper_threshold=args.perspective_threshold,
                    paper_chunk_chars=args.perspective_chunk_chars,
                ),
            )
        )

    if "harmformer" in args.setups:
        setups.append(("HarmFormer", HarmFormer(device=args.device)))

    if "llama_guard" in args.setups:
        setups.append(("Llama Guard", LlamaGuard(device=args.device)))

    if "openai_ttp" in args.setups:
        key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("openai_ttp selected but no OPENAI_API_KEY/--openai-key provided.")
        setups.append((f"TTP ({args.openai_model})", OpenAITTPClient(api_key=key, model=args.openai_model)))

    if "openrouter_ttp" in args.setups:
        key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("openrouter_ttp selected but no OPENROUTER_API_KEY/--openrouter-key provided.")
        setups.append(
            (
                f"TTP (OpenRouter: {args.openrouter_model})",
                OpenRouterTTPClient(
                    api_key=key,
                    model=args.openrouter_model,
                    referer=args.openrouter_referer,
                    title=args.openrouter_title,
                ),
            )
        )

    if "gemini_ttp" in args.setups:
        key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("gemini_ttp selected but no GEMINI_API_KEY/--gemini-key provided.")
        setups.append((f"TTP (Gemini: {args.gemini_model})", GeminiTTPEvaluator(api_key=key, model=args.gemini_model)))

    if "local_ttp" in args.setups:
        if not args.local_model:
            raise SystemExit("local_ttp selected but no --local-model provided.")
        for mid in args.local_model:
            setups.append(
                (
                    f"TTP (Local: {mid})",
                    TransformersTTPClient(
                        mid,
                        device=args.device,
                        dtype=args.dtype,
                        quantization=args.quantization,
                    ),
                )
            )

    results: List[Dict[str, Any]] = []
    for name, clf in setups:
        with maybe_track_emissions(run_name=f"ttp_eval_{name.replace(' ', '_').lower()}"):
            results.append(_evaluate_setup(name, clf, samples, dimension=args.dimension))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_config": {
            "dataset": args.data_path,
            "total_samples": len(samples),
            "dimension": args.dimension,
            "setups": [n for n, _ in setups],
            "device": args.device,
            "openai_model": args.openai_model,
            "openrouter_model": args.openrouter_model,
            "gemini_model": args.gemini_model,
            "perspective_threshold": args.perspective_threshold,
            "perspective_chunk_chars": args.perspective_chunk_chars,
            "local_models": args.local_model,
            "dtype": args.dtype,
            "quantization": args.quantization,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

