"""Shared pytest fixtures for the evo_gafs test-suite."""

from __future__ import annotations

import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from evo_gafs import GAConfig


@pytest.fixture
def classification_data():
    """Small classification dataset (iris) as a DataFrame plus target."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    return X, y


@pytest.fixture
def regression_data():
    """Small synthetic regression dataset as numpy arrays."""
    X, y = make_regression(n_samples=80, n_features=8, n_informative=4, noise=0.1, random_state=0)
    return X, y


@pytest.fixture
def clf():
    """A fast classifier used as the wrapper estimator."""
    return DecisionTreeClassifier(random_state=0)


@pytest.fixture
def reg():
    """A fast regressor used as the wrapper estimator."""
    return DecisionTreeRegressor(random_state=0)


@pytest.fixture
def tiny_config():
    """A tiny configuration that keeps tests fast."""
    return GAConfig(
        population_size=10,
        n_generations=4,
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
