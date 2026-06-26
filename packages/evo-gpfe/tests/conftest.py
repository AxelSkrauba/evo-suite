"""Shared pytest fixtures for the evo-gpfe test-suite."""

from __future__ import annotations

import pytest
from sklearn.datasets import load_iris, make_regression

from evo_gpfe import GPConfig


@pytest.fixture
def classification_data():
    """Small classification dataset (iris) as a DataFrame plus target."""
    return load_iris(return_X_y=True, as_frame=True)


@pytest.fixture
def regression_data():
    """Small synthetic regression dataset as numpy arrays."""
    return make_regression(n_samples=80, n_features=6, n_informative=4, noise=0.1, random_state=0)


@pytest.fixture
def tiny_config():
    """A tiny configuration that keeps tests fast."""
    return GPConfig(
        population_size=30,
        n_generations=6,
        n_features_to_generate=2,
        random_seed=0,
        verbose=False,
    )


@pytest.fixture(autouse=True)
def _matplotlib_agg():
    """Force a non-interactive matplotlib backend if matplotlib is present."""
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        pass
    yield
