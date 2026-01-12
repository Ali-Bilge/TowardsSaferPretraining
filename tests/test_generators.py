"""Tests for LLM generators."""

import pytest
from src.benchmarks.generators import create_generator


class TestGeneratorFactory:
    """Tests for generator factory."""

    def test_create_invalid_backend(self):
        """Test creating generator with invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_generator("invalid_backend", "test-model")


class TestTransformersGenerator:
    """Tests for Transformers generator (requires model download)."""

    @pytest.mark.requires_model
    def test_transformers_initialization(self):
        """Test Transformers generator initialization."""
        from src.benchmarks.generators import TransformersGenerator

        generator = TransformersGenerator(
            model_name="gpt2",  # Small model for testing
            max_new_tokens=10,
            temperature=0.7
        )
        assert generator.max_new_tokens == 10
        assert generator.temperature == 0.7
        assert generator.model.config.name_or_path is not None
        assert "gpt2" in generator.model.config.name_or_path
        assert generator.do_sample is True  # enabled automatically when temperature > 0
        assert generator.low_cpu_mem_usage is False  # default value
        assert generator.trust_remote_code is False  # default value