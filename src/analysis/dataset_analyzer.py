"""
Dataset analyzer for calculating harm prevalence in web-scale datasets.

Applies TTP to sampled web pages and generates statistics.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..evaluation import TTPEvaluator, TTPResult
from ..utils.taxonomy import Dimension

logger = logging.getLogger(__name__)

try:
    from ..models import HarmFormer
    HARMFORMER_AVAILABLE = True
except ImportError:
    HARMFORMER_AVAILABLE = False
    logger.warning("HarmFormer not available. Install torch and transformers to use it.")

# Harm codes used across comparison methods
_HARM_CODES = ["H", "IH", "SE", "IL", "SI"]


@dataclass
class AnalysisResult:
    """Results from dataset analysis."""
    dataset_name: str
    total_samples: int

    # Overall counts
    toxic_count: int = 0
    topical_count: int = 0
    safe_count: int = 0

    # Per-harm counts: {harm_code: {dimension: count}}
    harm_counts: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "H": {"toxic": 0, "topical": 0, "safe": 0},
        "IH": {"toxic": 0, "topical": 0, "safe": 0},
        "SE": {"toxic": 0, "topical": 0, "safe": 0},
        "IL": {"toxic": 0, "topical": 0, "safe": 0},
        "SI": {"toxic": 0, "topical": 0, "safe": 0},
    })

    # Evaluation results
    ttp_results: List[TTPResult] = field(default_factory=list)

    def get_percentages(self) -> Dict[str, Any]:
        """Calculate percentages for all metrics."""
        if self.total_samples == 0:
            return {}

        overall = {
            "toxic": (self.toxic_count / self.total_samples) * 100,
            "topical": (self.topical_count / self.total_samples) * 100,
            "safe": (self.safe_count / self.total_samples) * 100,
        }

        per_harm = {}
        for harm_code, counts in self.harm_counts.items():
            per_harm[harm_code] = {
                dim: (count / self.total_samples) * 100
                for dim, count in counts.items()
            }

        return {
            "overall": overall,
            "per_harm": per_harm,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "overall": {
                "toxic": self.toxic_count,
                "topical": self.topical_count,
                "safe": self.safe_count,
            },
            "harm_counts": self.harm_counts,
            "percentages": self.get_percentages(),
        }


class DatasetAnalyzer:
    """
    Analyzer for web-scale datasets.

    Example:
        analyzer = DatasetAnalyzer(api_key="sk-...")
        result = analyzer.analyze_samples("Common Crawl", samples)
        print(result.get_percentages())
    """

    _HARM_ATTRS = {
        "H": "hate_violence",
        "IH": "ideological",
        "SE": "sexual",
        "IL": "illegal",
        "SI": "self_inflicted"
    }

    def __init__(
        self,
        use_harmformer: bool = True,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        prompt_path: str = "prompts/TTP/TTP.txt",
        device: Optional[str] = None
    ):
        """
        Initialize analyzer.

        Args:
            use_harmformer: Use HarmFormer model (fast, free) instead of TTP (slow, paid)
            api_key: OpenAI API key (if using TTP)
            model: Model to use (if using TTP)
            prompt_path: Path to TTP prompt (if using TTP)
            device: Device for HarmFormer (cuda/cpu/mps)
        """
        self.use_harmformer = use_harmformer

        if use_harmformer:
            if not HARMFORMER_AVAILABLE:
                raise RuntimeError("HarmFormer requires torch and transformers. Install with: pip install torch transformers")
            self.harmformer = HarmFormer(device=device)
            self.evaluator = None
        else:
            # Validate api_key before creating TTPEvaluator
            if not api_key:
                raise ValueError("OpenAI API key is required when use_harmformer=False. Please provide a valid api_key parameter.")
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("OpenAI API key must be a non-empty string when use_harmformer=False.")
            if not api_key.startswith("sk-"):
                raise ValueError("OpenAI API key must start with 'sk-' when use_harmformer=False.")

            self.evaluator = TTPEvaluator(
                api_key=api_key,
                model=model,
                prompt_path=prompt_path
            )
            self.harmformer = None

    def analyze_samples(
        self,
        dataset_name: str,
        samples: List[Tuple[str, str]],
        show_progress: bool = True
    ) -> AnalysisResult:
        """
        Analyze a list of web page samples.

        Args:
            dataset_name: Name of dataset (e.g., "Common Crawl")
            samples: List of (url, body) tuples
            show_progress: Show progress bar

        Returns:
            AnalysisResult with statistics
        """
        logger.info(f"Analyzing {len(samples)} samples from {dataset_name} using {'HarmFormer' if self.use_harmformer else 'TTP'}...")

        # Evaluate all samples
        if self.use_harmformer:
            # Use HarmFormer (fast, local)
            texts = [body for _, body in samples]
            predicted_labels = self.harmformer.predict_batch(texts, show_progress=show_progress)

            # Convert to TTPResult format for compatibility
            ttp_results = [
                TTPResult(
                    url=url,
                    body=body,
                    predicted_label=label
                )
                for (url, body), label in zip(samples, predicted_labels)
            ]
        else:
            # Use TTP (slow, requires API key)
            ttp_results = self.evaluator.evaluate_batch(samples, show_progress=show_progress)

        # Initialize result
        result = AnalysisResult(
            dataset_name=dataset_name,
            total_samples=len(samples),
            ttp_results=ttp_results
        )

        # Calculate statistics
        for ttp_result in ttp_results:
            label = ttp_result.predicted_label

            # Overall counts
            if label.is_toxic():
                result.toxic_count += 1
            elif label.is_topical():
                result.topical_count += 1
            else:
                result.safe_count += 1

            # Per-harm counts
            for harm_code, attr_name in self._HARM_ATTRS.items():
                dim = getattr(label, attr_name)

                if dim == Dimension.TOXIC:
                    result.harm_counts[harm_code]["toxic"] += 1
                elif dim == Dimension.TOPICAL:
                    result.harm_counts[harm_code]["topical"] += 1
                else:
                    result.harm_counts[harm_code]["safe"] += 1

        logger.info(f"Analysis complete: {result.toxic_count} toxic, {result.topical_count} topical, {result.safe_count} safe")

        return result

    @staticmethod
    def compare_datasets(
        results: List[AnalysisResult]
    ) -> Dict[str, Any]:
        """
        Compare multiple dataset analysis results.

        Args:
            results: List of AnalysisResult objects

        Returns:
            Comparison dictionary
        """
        comparison = {
            "datasets": [r.dataset_name for r in results],
            "overall": {},
            "per_harm": {}
        }

        # Overall comparison
        for result in results:
            percentages = result.get_percentages()
            if result.total_samples == 0:
                comparison["overall"][result.dataset_name] = 0  # or None, safe default
            else:
                comparison["overall"][result.dataset_name] = percentages.get("overall", 0)

            # Per-harm comparison
            for harm_code in _HARM_CODES:
                if harm_code not in comparison["per_harm"]:
                    comparison["per_harm"][harm_code] = {}

                if result.total_samples == 0:
                    comparison["per_harm"][harm_code][result.dataset_name] = 0  # or None, safe default
                else:
                    per_harm_data = percentages.get("per_harm", {})
                    comparison["per_harm"][harm_code][result.dataset_name] = per_harm_data.get(harm_code, 0)

        return comparison

    def print_results(self, result: AnalysisResult):
        """Pretty print analysis results."""
        print(f"\n{'='*70}")
        print(f"Dataset Analysis: {result.dataset_name}")
        print(f"Total Samples: {result.total_samples}")
        print(f"{'='*70}")

        # Overall
        percentages = result.get_percentages()
        print("\nOverall Distribution:")
        if result.total_samples == 0:
            print(f"  Toxic:   {result.toxic_count:5d} (0.00%)")
            print(f"  Topical: {result.topical_count:5d} (0.00%)")
            print(f"  Safe:    {result.safe_count:5d} (0.00%)")
        else:
            overall = percentages.get('overall', {'toxic': 0, 'topical': 0, 'safe': 0})
            print(f"  Toxic:   {result.toxic_count:5d} ({overall.get('toxic', 0):.2f}%)")
            print(f"  Topical: {result.topical_count:5d} ({overall.get('topical', 0):.2f}%)")
            print(f"  Safe:    {result.safe_count:5d} ({overall.get('safe', 0):.2f}%)")

        # Per-harm
        print("\nPer-Harm Distribution:")
        print(f"{'Harm':<6} {'Toxic':<15} {'Topical':<15} {'Safe':<15}")
        print("-"*60)

        for harm_code in _HARM_CODES:
            counts = result.harm_counts[harm_code]
            pcts = percentages["per_harm"][harm_code]

            print(f"{harm_code:<6} "
                  f"{counts['toxic']:4d} ({pcts['toxic']:5.2f}%)  "
                  f"{counts['topical']:4d} ({pcts['topical']:5.2f}%)  "
                  f"{counts['safe']:4d} ({pcts['safe']:5.2f}%)")

        print(f"{'='*70}\n")

    @staticmethod
    def print_comparison(comparison: Dict[str, Any]):
        """Pretty print dataset comparison."""
        print(f"\n{'='*70}")
        print("Dataset Comparison")
        print(f"Datasets: {', '.join(comparison['datasets'])}")
        print(f"{'='*70}")

        # Overall comparison
        print("\nOverall Toxic Percentages:")
        for dataset in comparison["datasets"]:
            toxic_pct = comparison["overall"][dataset]["toxic"]
            print(f"  {dataset:<20}: {toxic_pct:.2f}%")

        print("\nOverall Topical Percentages:")
        for dataset in comparison["datasets"]:
            topical_pct = comparison["overall"][dataset]["topical"]
            print(f"  {dataset:<20}: {topical_pct:.2f}%")

        # Per-harm toxic comparison
        print("\nPer-Harm Toxic Percentages:")
        print(f"{'Harm':<6}", end="")
        for dataset in comparison["datasets"]:
            print(f"{dataset:<15}", end="")
        print()
        print("-"*70)

        for harm_code in _HARM_CODES:
            print(f"{harm_code:<6}", end="")
            for dataset in comparison["datasets"]:
                toxic_pct = comparison["per_harm"][harm_code][dataset]["toxic"]
                print(f"{toxic_pct:5.2f}%        ", end="")
            print()

        print(f"{'='*70}\n")

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        if self.use_harmformer:
            return {"model": "HarmFormer", "device": self.harmformer.device}
        else:
            return self.evaluator.get_stats()
