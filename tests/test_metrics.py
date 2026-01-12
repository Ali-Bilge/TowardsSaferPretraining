"""Tests for evaluation metrics."""

import pytest
from src.benchmarks.metrics import calculate_metrics
from src.utils.taxonomy import HarmLabel, Dimension


class TestMetrics:
    """Tests for metrics calculation."""

    @pytest.fixture
    def perfect_predictions(self):
        """Perfect predictions matching ground truth."""
        ground_truth = [
            HarmLabel(hate_violence=Dimension.TOXIC),
            HarmLabel(sexual=Dimension.TOXIC),
            HarmLabel(),  # Safe
        ]
        predictions = ground_truth.copy()
        return predictions, ground_truth

    @pytest.fixture
    def partial_predictions(self):
        """Partial predictions with some errors."""
        ground_truth = [
            HarmLabel(hate_violence=Dimension.TOXIC),
            HarmLabel(sexual=Dimension.TOXIC),
            HarmLabel(),  # Safe
        ]
        predictions = [
            HarmLabel(hate_violence=Dimension.TOXIC),  # Correct
            HarmLabel(sexual=Dimension.TOPICAL),       # Partial (wrong level)
            HarmLabel(),                                # Correct
        ]
        return predictions, ground_truth

    def test_perfect_predictions(self, perfect_predictions):
        """Test metrics with perfect predictions."""
        predictions, ground_truth = perfect_predictions
        metrics = calculate_metrics(predictions, ground_truth, dimension="toxic")

        assert metrics['overall']['precision'] == 1.0
        assert metrics['overall']['recall'] == 1.0
        assert metrics['overall']['f1'] == 1.0

    def test_partial_predictions(self, partial_predictions):
        """Test metrics with partial predictions."""
        predictions, ground_truth = partial_predictions
        metrics = calculate_metrics(predictions, ground_truth, dimension="toxic")

        # One correct toxic (H), one missed (SE classified as topical), one correct safe
        assert metrics['overall']['precision'] == 1.0  # No false positives
        assert metrics['overall']['recall'] == 0.5     # Missed 1 of 2 toxic
        expected_f1 = 2 * 1.0 * 0.5 / (1.0 + 0.5)
        assert metrics['overall']['f1'] == pytest.approx(expected_f1)

    def test_per_harm_metrics(self, partial_predictions):
        """Test per-harm metrics calculation."""
        predictions, ground_truth = partial_predictions
        metrics = calculate_metrics(predictions, ground_truth, dimension="toxic")

        assert 'per_harm' in metrics
        assert 'H' in metrics['per_harm']
        assert 'SE' in metrics['per_harm']

        # H should be perfect
        assert metrics['per_harm']['H']['precision'] == 1.0
        assert metrics['per_harm']['H']['recall'] == 1.0

        # SE should have 0 recall (predicted topical instead of toxic)
        assert metrics['per_harm']['SE']['recall'] == 0.0
        # SE should have 0 precision when no SE predictions exist at toxic level
        assert metrics['per_harm']['SE']['precision'] == 0.0

    def test_dimension_modes(self):
        """Test different dimension modes."""
        # Create ground truth and predictions with different Dimension values across categories
        # so that metrics differ between modes
        ground_truth = [
            HarmLabel(hate_violence=Dimension.TOXIC),      # Toxic sample
            HarmLabel(sexual=Dimension.TOPICAL),           # Topical sample
            HarmLabel(illegal=Dimension.TOXIC),            # Toxic sample
            HarmLabel(ideological=Dimension.TOPICAL),      # Topical sample
        ]
        predictions = [
            HarmLabel(hate_violence=Dimension.TOXIC),      # Correct toxic
            HarmLabel(sexual=Dimension.TOPICAL),           # Correct topical
            HarmLabel(illegal=Dimension.SAFE),             # Missed toxic (predicted safe)
            HarmLabel(ideological=Dimension.TOXIC),        # Wrong (predicted toxic instead of topical)
        ]

        # Test toxic mode
        metrics_toxic = calculate_metrics(predictions, ground_truth, dimension="toxic")
        assert metrics_toxic['dimension'] == "toxic"

        # Test topical mode
        metrics_topical = calculate_metrics(predictions, ground_truth, dimension="topical")
        assert metrics_topical['dimension'] == "topical"

        # Test all mode
        metrics_all = calculate_metrics(predictions, ground_truth, dimension="all")
        assert metrics_all['dimension'] == "all"

        # Assert that metric values differ between modes
        # Toxic mode: TP=1, FP=1, FN=1 → precision=0.5, recall=0.5, f1=0.5
        assert metrics_toxic['overall']['precision'] == 0.5
        assert metrics_toxic['overall']['recall'] == 0.5
        assert metrics_toxic['overall']['f1'] == 0.5

        # Topical mode: TP=1, FP=0, FN=1 → precision=1.0, recall=0.5, f1≈0.667
        assert metrics_topical['overall']['precision'] == 1.0
        assert metrics_topical['overall']['recall'] == 0.5
        assert metrics_topical['overall']['f1'] == pytest.approx(2/3)

        # All mode: TP=3, FP=0, FN=1 → precision=1.0, recall=0.75, f1≈0.857
        assert metrics_all['overall']['precision'] == 1.0
        assert metrics_all['overall']['recall'] == 0.75
        assert metrics_all['overall']['f1'] == pytest.approx(6/7)

    def test_mismatched_lengths(self):
        """Test error on mismatched prediction/ground truth lengths."""
        predictions = [HarmLabel()]
        ground_truth = [HarmLabel(), HarmLabel()]

        with pytest.raises(ValueError, match="Mismatch"):
            calculate_metrics(predictions, ground_truth)

    def test_empty_predictions(self):
        """Test with empty predictions."""
        predictions = []
        ground_truth = []

        metrics = calculate_metrics(predictions, ground_truth, dimension="toxic")
        assert metrics['total_samples'] == 0
        assert metrics['overall']['precision'] == 0.0
        assert metrics['overall']['recall'] == 0.0
        assert metrics['overall']['f1'] == 0.0

    def test_all_safe_predictions(self):
        """Test with all safe predictions."""
        predictions = [HarmLabel(), HarmLabel(), HarmLabel()]
        ground_truth = [HarmLabel(), HarmLabel(), HarmLabel()]

        metrics = calculate_metrics(predictions, ground_truth, dimension="toxic")
        assert metrics['total_samples'] == 3
        assert metrics['overall']['precision'] == 0.0
        assert metrics['overall']['recall'] == 0.0
        assert metrics['overall']['f1'] == 0.0

