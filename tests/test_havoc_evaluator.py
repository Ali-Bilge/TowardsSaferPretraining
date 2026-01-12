"""Tests for HAVOC evaluator."""

import pytest  # type: ignore
import os
from pathlib import Path
from src.benchmarks import HAVOCBenchmark as HAVOCEvaluator
from src.benchmarks.havoc import HAVOCEvaluationResult
from src.data_loaders import HAVOCLoader


class TestHAVOCEvaluator:
    """Tests for HAVOC benchmark evaluator."""

    @pytest.fixture
    def samples(self):
        """Load small sample of HAVOC data."""
        # Check environment variable, fall back to default
        havoc_path_str = os.environ.get("HAVOC_PATH", "data/HAVOC/havoc.tsv")
        havoc_path = Path(havoc_path_str)
        if not havoc_path.exists():
            pytest.skip(f"HAVOC dataset not found at {havoc_path}")

        loader = HAVOCLoader(str(havoc_path))
        return loader.load()[:5]  # Just 5 samples for testing

    @pytest.mark.requires_model
    def test_evaluator_initialization(self):
        """Test HAVOC evaluator initialization."""
        evaluator = HAVOCEvaluator(judge="harmformer")
        assert evaluator.judge == "harmformer"

    def test_leakage_percentages(self):
        """Test leakage percentage calculation."""
        result = HAVOCEvaluationResult(
            model_name="test",
            total_samples=10,
            neutral_total=5,
            neutral_leaked=1,
            passive_total=3,
            passive_leaked=2,
            provocative_total=2,
            provocative_leaked=2
        )

        percentages = result.get_leakage_percentages()

        assert percentages['neutral'] == 20.0  # 1/5
        assert percentages['passive'] == pytest.approx(66.67, rel=0.01)  # 2/3
        assert percentages['provocative'] == 100.0  # 2/2
        assert percentages['overall'] == 50.0  # 5/10

    def test_leakage_percentages_zero_total_samples(self):
        """Test leakage percentages with zero total_samples."""
        result = HAVOCEvaluationResult(
            model_name="test",
            total_samples=0,  # Edge case: zero total samples
            neutral_total=0,
            neutral_leaked=0,
            passive_total=0,
            passive_leaked=0,
            provocative_total=0,
            provocative_leaked=0
        )

        percentages = result.get_leakage_percentages()

        # All percentages should be 0.0 due to division by zero protection
        assert percentages['neutral'] == 0.0  # 0/0
        assert percentages['passive'] == 0.0  # 0/0
        assert percentages['provocative'] == 0.0  # 0/0
        # Overall should be 0.0 due to division by zero protection
        assert percentages['overall'] == 0.0  # 0/0 handled safely

    def test_leakage_percentages_zero_subcategory_totals(self):
        """Test leakage percentages with all subcategory totals zero."""
        result = HAVOCEvaluationResult(
            model_name="test",
            total_samples=0,  # Edge case: all subcategory totals zero
            neutral_total=0,
            neutral_leaked=0,
            passive_total=0,
            passive_leaked=0,
            provocative_total=0,
            provocative_leaked=0
        )

        percentages = result.get_leakage_percentages()

        # All subcategory percentages should be 0.0 due to division by zero protection
        assert percentages['neutral'] == 0.0
        assert percentages['passive'] == 0.0
        assert percentages['provocative'] == 0.0
        # Overall should also be 0.0 since no leaked samples
        assert percentages['overall'] == 0.0  # 0/0 (division by zero protection)

    def test_leakage_percentages_leaked_exceeds_total(self):
        """Test leakage percentages when leaked counts exceed their totals."""
        result = HAVOCEvaluationResult(
            model_name="test",
            total_samples=10,
            neutral_total=5,
            neutral_leaked=7,  # Edge case: leaked exceeds total
            passive_total=3,
            passive_leaked=5,  # Edge case: leaked exceeds total
            provocative_total=2,
            provocative_leaked=4  # Edge case: leaked exceeds total
        )

        percentages = result.get_leakage_percentages()

        # Percentages should exceed 100% since method doesn't validate
        assert percentages['neutral'] == 140.0  # 7/5
        assert percentages['passive'] == pytest.approx(166.67, rel=0.01)  # 5/3
        assert percentages['provocative'] == 200.0  # 4/2
        assert percentages['overall'] == 160.0  # 16/10

    def test_result_to_dict(self):
        """Test conversion of result to dictionary."""
        result = HAVOCEvaluationResult(
            model_name="test-model",
            total_samples=10,
            neutral_total=5,
            neutral_leaked=1,
            passive_total=3,
            passive_leaked=2,
            provocative_total=2,
            provocative_leaked=1
        )

        result_dict = result.to_dict()

        # Basic fields
        assert result_dict['model_name'] == "test-model"
        assert result_dict['total_samples'] == 10

        # Leakage structure
        assert result_dict['leakage']['neutral']['total'] == 5
        assert result_dict['leakage']['neutral']['leaked'] == 1
        assert result_dict['leakage']['passive']['total'] == 3
        assert result_dict['leakage']['passive']['leaked'] == 2
        assert result_dict['leakage']['provocative']['total'] == 2
        assert result_dict['leakage']['provocative']['leaked'] == 1

        # Leakage percentages - computed values
        assert result_dict['leakage_percentages']['neutral'] == 20.0  # 1/5 * 100
        assert result_dict['leakage_percentages']['passive'] == pytest.approx(66.67, rel=0.01)  # 2/3 * 100
        assert result_dict['leakage_percentages']['provocative'] == 50.0  # 1/2 * 100
        assert result_dict['leakage_percentages']['overall'] == 40.0  # 4/10 * 100

        # Harm leakage structure and values
        expected_harm_leakage = {
            "H": {"neutral": 0, "passive": 0, "provocative": 0},
            "IH": {"neutral": 0, "passive": 0, "provocative": 0},
            "SE": {"neutral": 0, "passive": 0, "provocative": 0},
            "IL": {"neutral": 0, "passive": 0, "provocative": 0},
            "SI": {"neutral": 0, "passive": 0, "provocative": 0},
        }
        assert result_dict['harm_leakage'] == expected_harm_leakage

    @pytest.mark.requires_dataset
    @pytest.mark.requires_api_key
    def test_havoc_evaluator_integration(self, samples):
        """Integration test for HAVOC evaluator with real samples."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        # Mock generate function that returns varied outputs based on prefix content
        # to exercise different leakage classifications and error paths
        call_count = 0

        def mock_generate_fn(prefix: str) -> str:
            nonlocal call_count
            call_count += 1

            # Branch based on prefix content to exercise different leakage types
            prefix_lower = prefix.lower()

            # Test error path - raise exception on first call
            if call_count == 1:
                raise RuntimeError("Simulated generation failure")

            # Return harmful content for sensitive prefixes to test provocative leakage
            if any(word in prefix_lower for word in ["kill", "murder", "harm", "violence", "attack", "gun", "bomb"]):
                return "I think violence is the only solution. Let's organize an attack immediately."

            # Return topical but potentially harmful content for certain prefixes to test passive leakage
            if any(word in prefix_lower for word in ["war", "conflict", "death", "abuse", "bullying", "harassment"]):
                return "This is a serious issue that needs attention. People should be aware of these dangers."

            # Return neutral content for most other cases
            return "This is a harmless and neutral response to the input text."

        # Initialize evaluator with TTP judge
        evaluator = HAVOCEvaluator(judge="ttp", api_key=api_key)

        # Evaluate samples
        result = evaluator.evaluate_samples(
            model_name="test-model",
            samples=samples,
            generate_fn=mock_generate_fn,
            show_progress=False
        )

        # Basic assertions
        assert result.model_name == "test-model"
        assert result.total_samples == len(samples)
        assert len(result.results) == len(samples)

        # Check that samples were processed with varied outcomes
        error_results = []
        successful_results = []

        for leak_result in result.results:
            assert leak_result.sample is not None
            assert leak_result.generated_text is not None
            assert leak_result.full_generation is not None
            assert leak_result.leakage_type in ["neutral", "passive", "provocative"]

            if leak_result.error:
                error_results.append(leak_result)
            else:
                successful_results.append(leak_result)

        # Verify we have at least one error result (from the first call that raises exception)
        assert len(error_results) >= 1, "Expected at least one error result from simulated failure"
        assert len(successful_results) >= 1, "Expected at least some successful results"

        # Verify error handling - first result should have error
        first_result = result.results[0]
        assert first_result.error is not None, "First result should have error from simulated failure"
        assert "Simulated generation failure" in first_result.error

        # Check that totals include both successful and failed samples
        total_counted = (result.neutral_total + result.passive_total + result.provocative_total)
        # Total samples should equal successful evaluations (errors are not counted in totals)
        assert total_counted == result.total_samples
        # Error count should match our error results
        assert result.error_count == len(error_results)
