#!/usr/bin/env python3
"""
Estimate toxicity prevalence on large text datasets (Table 8-style).

The paper reports prevalence using 1,000,000 samples. In this repo we default to
10,000 samples to fit small budgets. Set --limit 1000000 for paper-faithful runs.

Input is user-provided (this repo does not ship CommonCrawl/C4/FineWeb dumps).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clients import PerspectiveAPI, LlamaGuard, OpenRouterTTPClient  # noqa: E402
from src.clients.ttp_openai import OpenAITTPClient  # noqa: E402
from src.models import HarmFormer  # noqa: E402
from src.utils.taxonomy import Dimension, HarmLabel  # noqa: E402


@dataclass(frozen=True)
class _Counts:
    overall_toxic: int = 0
    H: int = 0
    IH: int = 0
    SE: int = 0
    IL: int = 0
    SI: int = 0


def _iter_texts(path: Path, *, fmt: str, text_field: str) -> Iterable[str]:
    if fmt == "txt":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                t = (line or "").strip()
                if t:
                    yield t
        return

    if fmt == "jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get(text_field)
                if isinstance(t, str) and t.strip():
                    yield t.strip()
        return

    if fmt == "tsv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                t = (row.get(text_field) or "").strip()
                if t:
                    yield t
        return

    raise ValueError(f"Unknown format: {fmt}")


def _reservoir_sample(texts: Iterable[str], *, k: int, rng: random.Random) -> List[str]:
    sample: List[str] = []
    seen = 0
    for t in texts:
        if not t:
            continue
        if len(sample) < k:
            sample.append(t)
        else:
            j = rng.randrange(seen + 1)
            if j < k:
                sample[j] = t
        seen += 1
    return sample


def _load_samples(path: Path, *, fmt: str, text_field: str, limit: int, method: str, seed: int) -> List[str]:
    it = _iter_texts(path, fmt=fmt, text_field=text_field)
    if limit <= 0:
        return []

    if method == "first":
        out: List[str] = []
        for t in it:
            out.append(t)
            if len(out) >= limit:
                break
        return out

    if method == "reservoir":
        rng = random.Random(seed)
        return _reservoir_sample(it, k=limit, rng=rng)

    raise ValueError(f"Unknown sampling method: {method}")


def _make_classifier(args):
    if args.setup == "harmformer":
        return HarmFormer(device=args.device, batch_size=args.batch_size)

    if args.setup == "llama_guard":
        return LlamaGuard(device=args.device)

    if args.setup == "perspective":
        key = args.perspective_key or os.environ.get("PERSPECTIVE_API_KEY")
        if not key and os.environ.get("ENABLE_PERSPECTIVE_WITH_GEMINI_KEY") == "1":
            key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("perspective selected but no PERSPECTIVE_API_KEY/--perspective-key provided.")
        return PerspectiveAPI(
            api_key=key,
            mode="paper_table4",
            paper_threshold=args.perspective_threshold,
            paper_chunk_chars=args.perspective_chunk_chars,
        )

    if args.setup == "openai_ttp":
        key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("openai_ttp selected but no OPENAI_API_KEY/--openai-key provided.")
        return OpenAITTPClient(api_key=key, model=args.openai_model)

    if args.setup == "openrouter_ttp":
        key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("openrouter_ttp selected but no OPENROUTER_API_KEY/--openrouter-key provided.")
        return OpenRouterTTPClient(
            api_key=key,
            model=args.openrouter_model,
            referer=args.openrouter_referer,
            title=args.openrouter_title,
        )

    raise ValueError(f"Unknown setup: {args.setup}")


def _accumulate_counts(label: HarmLabel, counts: _Counts) -> _Counts:
    return _Counts(
        overall_toxic=counts.overall_toxic + (1 if label.is_toxic() else 0),
        H=counts.H + (1 if label.hate_violence == Dimension.TOXIC else 0),
        IH=counts.IH + (1 if label.ideological == Dimension.TOXIC else 0),
        SE=counts.SE + (1 if label.sexual == Dimension.TOXIC else 0),
        IL=counts.IL + (1 if label.illegal == Dimension.TOXIC else 0),
        SI=counts.SI + (1 if label.self_inflicted == Dimension.TOXIC else 0),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Estimate toxicity prevalence (Table 8-style)")
    p.add_argument("--input-path", required=True, help="Path to dataset file (txt/jsonl/tsv)")
    p.add_argument("--input-format", default="jsonl", choices=["jsonl", "txt", "tsv"])
    p.add_argument(
        "--text-field",
        default="text",
        help="Field name for jsonl/tsv inputs (ignored for txt). Example: Body",
    )

    # Paper uses 1,000,000; default to 10,000 for small budgets.
    p.add_argument("--limit", type=int, default=10_000, help="Number of samples to evaluate")
    p.add_argument("--sample-method", default="reservoir", choices=["reservoir", "first"])
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--setup",
        default="harmformer",
        choices=["harmformer", "llama_guard", "perspective", "openai_ttp", "openrouter_ttp"],
    )
    p.add_argument("--output", required=True, help="Output JSON file")

    # Common local model settings
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--batch-size", type=int, default=16)

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
    p.add_argument("--openrouter-referer", default=os.environ.get("OPENROUTER_REFERER"))
    p.add_argument("--openrouter-title", default=os.environ.get("OPENROUTER_TITLE"))

    args = p.parse_args()

    in_path = Path(args.input_path)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    texts = _load_samples(
        in_path,
        fmt=args.input_format,
        text_field=args.text_field,
        limit=int(args.limit),
        method=args.sample_method,
        seed=int(args.seed),
    )
    if not texts:
        raise SystemExit("No samples loaded (check --input-format/--text-field and input file contents).")

    clf = _make_classifier(args)

    # Prefer batch prediction when supported.
    labels: List[HarmLabel]
    if hasattr(clf, "predict_batch") and callable(getattr(clf, "predict_batch")):
        labels = clf.predict_batch(texts, show_progress=True)
    else:
        labels = [clf.predict(t) for t in texts]

    counts = _Counts()
    for lab in labels:
        counts = _accumulate_counts(lab, counts)

    n = len(labels)
    payload: Dict[str, Any] = {
        "config": {
            "input_path": str(in_path),
            "input_format": args.input_format,
            "text_field": args.text_field,
            "limit": int(args.limit),
            "sample_method": args.sample_method,
            "seed": int(args.seed),
            "setup": args.setup,
        },
        "counts": {
            "n": n,
            "overall_toxic": counts.overall_toxic,
            "H": counts.H,
            "IH": counts.IH,
            "SE": counts.SE,
            "IL": counts.IL,
            "SI": counts.SI,
        },
        "prevalence": {
            "overall_toxic": counts.overall_toxic / n,
            "H": counts.H / n,
            "IH": counts.IH / n,
            "SE": counts.SE / n,
            "IL": counts.IL / n,
            "SI": counts.SI / n,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

