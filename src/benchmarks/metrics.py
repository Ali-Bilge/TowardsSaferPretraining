"""Metrics calculation for TTP evaluation."""

from typing import List, Dict, Any

from ..utils.taxonomy import HarmLabel, Dimension, HarmCategory


def calculate_metrics(
    predictions: List[HarmLabel],
    ground_truth: List[HarmLabel],
    dimension: str = "toxic",
) -> Dict[str, Any]:
    """Calculate precision/recall/F1 for TTP predictions."""
    if dimension not in ["toxic", "topical", "all"]:
        raise ValueError("Invalid dimension. Must be one of: 'toxic', 'topical', 'all'")
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(ground_truth)} ground truth")

    overall_tp, overall_fp, overall_fn = 0, 0, 0

    short_name_mapping = HarmCategory.get_short_name_mapping()
    short_to_attr = {short_name: category.name.lower() for category, short_name in short_name_mapping.items()}
    harm_metrics = {short_name: {"tp": 0, "fp": 0, "fn": 0} for short_name in short_to_attr.keys()}

    for pred, truth in zip(predictions, ground_truth):
        for attr_name in short_to_attr.values():
            if not hasattr(pred, attr_name):
                raise AttributeError(f"Prediction object missing required attribute '{attr_name}'")
            if not hasattr(truth, attr_name):
                raise AttributeError(f"Ground truth object missing required attribute '{attr_name}'")

        if dimension == "toxic":
            pred_positive = pred.is_toxic()
            truth_positive = truth.is_toxic()
        elif dimension == "topical":
            pred_positive = pred.is_topical()
            truth_positive = truth.is_topical()
        else:
            pred_positive = not pred.is_safe()
            truth_positive = not truth.is_safe()

        if pred_positive and truth_positive:
            overall_tp += 1
        elif pred_positive and not truth_positive:
            overall_fp += 1
        elif not pred_positive and truth_positive:
            overall_fn += 1

        for harm_code, attr_name in short_to_attr.items():
            pred_dim = getattr(pred, attr_name)
            truth_dim = getattr(truth, attr_name)

            if dimension == "toxic":
                pred_pos = pred_dim == Dimension.TOXIC
                truth_pos = truth_dim == Dimension.TOXIC
            elif dimension == "topical":
                pred_pos = pred_dim == Dimension.TOPICAL
                truth_pos = truth_dim == Dimension.TOPICAL
            else:
                pred_pos = pred_dim != Dimension.SAFE
                truth_pos = truth_dim != Dimension.SAFE

            if pred_pos and truth_pos:
                harm_metrics[harm_code]["tp"] += 1
            elif pred_pos and not truth_pos:
                harm_metrics[harm_code]["fp"] += 1
            elif not pred_pos and truth_pos:
                harm_metrics[harm_code]["fn"] += 1

    def calc_scores(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    overall_scores = calc_scores(overall_tp, overall_fp, overall_fn)
    per_harm_scores = {
        harm_code: calc_scores(counts["tp"], counts["fp"], counts["fn"])
        for harm_code, counts in harm_metrics.items()
    }

    return {
        "overall": overall_scores,
        "per_harm": per_harm_scores,
        "dimension": dimension,
        "total_samples": len(predictions),
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluation Metrics (Dimension: {metrics['dimension']})")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"{'='*60}")

    print("\nOverall:")
    overall = metrics["overall"]
    print(f"  Precision: {overall['precision']:.3f}")
    print(f"  Recall:    {overall['recall']:.3f}")
    print(f"  F1:        {overall['f1']:.3f}")

    print("\nPer-Harm:")
    for harm, scores in metrics["per_harm"].items():
        print(f"  {harm:3s}: P={scores['precision']:.3f}, R={scores['recall']:.3f}, F1={scores['f1']:.3f}")

    print(f"{'='*60}\n")

