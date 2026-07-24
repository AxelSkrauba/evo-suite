"""Shared pytest fixtures for the evo-ens test-suite."""

from __future__ import annotations

import pytest
from sklearn.datasets import load_iris, make_classification, make_regression

from evo_ens import EvoEnsConfig


@pytest.fixture
def classification_data():
    """Small classification dataset (iris) as a DataFrame plus target."""
    return load_iris(return_X_y=True, as_frame=True)


@pytest.fixture
def binary_classification_data():
    """Small synthetic binary classification dataset as numpy arrays."""
    return make_classification(n_samples=120, n_features=8, n_informative=5, random_state=0)


@pytest.fixture
def regression_data():
    """Small synthetic regression dataset as numpy arrays."""
    return make_regression(n_samples=100, n_features=6, n_informative=4, noise=0.1, random_state=0)


@pytest.fixture
def tiny_config():
    """A tiny single-objective configuration that keeps tests fast."""
    return EvoEnsConfig(
        population_size=12,
        n_generations=5,
        min_models=1,
        cv_folds=3,
        random_seed=0,
        verbose=False,
        early_stopping_rounds=None,
    )


@pytest.fixture
def tiny_nsga2_config():
    """A tiny multiobjective configuration that keeps tests fast."""
    return EvoEnsConfig(
        mode="multiobjective",
        population_size=12,
        n_generations=5,
        min_models=1,
        cv_folds=3,
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
