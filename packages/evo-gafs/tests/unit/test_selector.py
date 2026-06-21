"""Unit tests for the GAFeatureSelector estimator surface."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from evo_gafs import GAConfig, GAFeatureSelector


def test_fit_sets_fitted_attributes(classification_data, clf, tiny_config):
    X, y = classification_data
    selector = GAFeatureSelector(clf, tiny_config).fit(X, y)
    assert selector.n_features_in_ == X.shape[1]
    assert selector.support_.shape == (X.shape[1],)
    assert selector.support_.dtype == bool
    assert selector.result_.n_selected >= 1


def test_transform_matches_support(classification_data, clf, tiny_config):
    X, y = classification_data
    selector = GAFeatureSelector(clf, tiny_config).fit(X, y)
    Xt = selector.transform(X)
    assert Xt.shape[1] == selector.support_.sum()


def test_get_support_indices(classification_data, clf, tiny_config):
    X, y = classification_data
    selector = GAFeatureSelector(clf, tiny_config).fit(X, y)
    idx = selector.get_support(indices=True)
    assert len(idx) == selector.result_.n_selected


def test_fit_does_not_mutate_config(classification_data, clf):
    X, y = classification_data
    config = GAConfig(population_size=8, n_generations=3, verbose=False)
    assert config.mutation_indpb is None
    GAFeatureSelector(clf, config).fit(X, y)
    assert config.mutation_indpb is None  # untouched


def test_feature_names_in_set_for_dataframe(classification_data, clf, tiny_config):
    X, y = classification_data
    selector = GAFeatureSelector(clf, tiny_config).fit(X, y)
    assert list(selector.feature_names_in_) == list(X.columns)


def test_numpy_input_has_no_feature_names_in(regression_data, reg, tiny_config):
    X, y = regression_data
    selector = GAFeatureSelector(reg, tiny_config, task_type="regression").fit(X, y)
    assert not hasattr(selector, "feature_names_in_")


def test_transform_before_fit_raises(classification_data, clf, tiny_config):
    X, _ = classification_data
    selector = GAFeatureSelector(clf, tiny_config)
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        selector.transform(X)


def test_nan_input_raises(clf, tiny_config):
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        GAFeatureSelector(clf, tiny_config).fit(X, y)


def test_get_params_and_clone(clf, tiny_config):
    selector = GAFeatureSelector(clf, tiny_config, scoring="accuracy")
    params = selector.get_params()
    assert params["scoring"] == "accuracy"
    cloned = clone(selector)
    assert isinstance(cloned, GAFeatureSelector)


def test_summary_before_fit_raises(clf, tiny_config):
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        GAFeatureSelector(clf, tiny_config).summary()
