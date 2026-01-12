import os

import pytest


pytestmark = [pytest.mark.slow]


if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Integration smoke test. Set RUN_INTEGRATION_TESTS=1 to enable.",
        allow_module_level=True,
    )


@pytest.fixture
def transformers_generator():
    """Fixture that provides a transformers generator and cleans up after use."""
    from src.benchmarks.generators import create_generator

    gen = create_generator("transformers", "gpt2")
    yield gen
    gen.cleanup()


def test_core_components_importable():
    # Arrange / Act
    import src  # noqa: F401
    from src.data_loaders import HAVOCLoader, TTPEvalLoader  # noqa: F401
    from src.benchmarks import HAVOCBenchmark  # noqa: F401
    from src.clients import TTPEvaluator  # noqa: F401


@pytest.mark.requires_model
def test_transformers_generator_smoke(transformers_generator):
    # Arrange
    gen = transformers_generator

    # Act
    out = gen.generate("Hello there!", max_tokens=5, temperature=0.0)

    # Assert
    assert out.startswith("Hello there!")

