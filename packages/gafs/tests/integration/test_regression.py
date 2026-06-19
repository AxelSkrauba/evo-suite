"""Integration tests for regression."""

from __future__ import annotations

from gafs import GAConfig, GAFeatureSelector


def test_regression_end_to_end(regression_data, reg, tiny_config):
    X, y = regression_data
    selector = GAFeatureSelector(reg, tiny_config, task_type="regression").fit(X, y)
    result = selector.result_
    assert 1 <= result.n_selected <= X.shape[1]
    assert selector.transform(X).shape[1] == result.n_selected


def test_auto_task_inference_regression(regression_data, reg):
    X, y = regression_data
    config = GAConfig(population_size=10, n_generations=4, verbose=False)
    # task_type defaults to 'auto'; continuous target -> regression
    selector = GAFeatureSelector(reg, config).fit(X, y)
    assert selector.result_.n_selected >= 1


def test_custom_scoring(regression_data, reg, tiny_config):
    X, y = regression_data
    selector = GAFeatureSelector(
        reg, tiny_config, scoring="neg_mean_squared_error", task_type="regression"
    ).fit(X, y)
    assert selector.result_.n_selected >= 1
