"""Integration tests for single-objective classification."""

from __future__ import annotations

import numpy as np

from gafs import GAConfig, GAFeatureSelector


def test_classification_end_to_end(classification_data, clf, tiny_config):
    X, y = classification_data
    selector = GAFeatureSelector(clf, tiny_config).fit(X, y)
    result = selector.result_
    assert 1 <= result.n_selected <= X.shape[1]
    assert 0.0 <= result.best_cv_score <= 1.0
    assert 0.0 <= result.compression_ratio < 1.0
    assert len(result.history) <= tiny_config.n_generations


def test_reproducible_with_seed(classification_data, clf):
    X, y = classification_data
    config = GAConfig(population_size=12, n_generations=5, random_seed=7, verbose=False)
    a = GAFeatureSelector(clf, config).fit(X, y).support_
    b = GAFeatureSelector(clf, config).fit(X, y).support_
    assert np.array_equal(a, b)


def test_alpha_zero_favours_compression(classification_data, clf):
    X, y = classification_data
    high_alpha = GAConfig(
        population_size=14, n_generations=8, alpha=1.0, random_seed=0, verbose=False
    )
    low_alpha = GAConfig(
        population_size=14, n_generations=8, alpha=0.1, random_seed=0, verbose=False
    )
    n_high = GAFeatureSelector(clf, high_alpha).fit(X, y).result_.n_selected
    n_low = GAFeatureSelector(clf, low_alpha).fit(X, y).result_.n_selected
    assert n_low <= n_high


def test_string_labels_are_encoded(clf, tiny_config):
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data
    y = np.array(["setosa", "versicolor", "virginica"])[data.target]
    selector = GAFeatureSelector(clf, tiny_config, task_type="classification").fit(X, y)
    assert selector.result_.n_selected >= 1
