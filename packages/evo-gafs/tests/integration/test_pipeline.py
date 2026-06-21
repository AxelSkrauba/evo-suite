"""Integration tests for scikit-learn pipeline interoperability and plotting."""

from __future__ import annotations

import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evo_gafs import BenchmarkRunner, GAFeatureSelector, GAPlotter


def test_works_inside_pipeline(classification_data, clf, tiny_config):
    X, y = classification_data
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", GAFeatureSelector(clf, tiny_config)),
            ("clf", SVC()),
        ]
    )
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.5


def test_benchmark_runner(classification_data, regression_data, clf, reg, tiny_config):
    X_clf, y_clf = classification_data
    X_reg, y_reg = regression_data
    runner = BenchmarkRunner()
    runner.add_dataset("iris", X_clf, y_clf, task_type="classification")
    runner.add_dataset("synth", X_reg, y_reg, task_type="regression")
    results = runner.run(clf, config=tiny_config, estimator_regression=reg, verbose=False)
    assert len(results) == 2
    df = runner.report()
    assert len(df) == 2


def test_plot_helpers_return_figures(classification_data, clf, tiny_config):
    pytest.importorskip("matplotlib")
    X, y = classification_data
    selector = GAFeatureSelector(clf, tiny_config).fit(X, y)
    fig = GAPlotter.plot_evolution(selector.result_)
    assert fig is not None
    fig2 = GAPlotter.plot_selected_features(selector.result_, list(X.columns))
    assert fig2 is not None


def test_plot_pareto_requires_multiobjective(classification_data, clf, tiny_config):
    pytest.importorskip("matplotlib")
    X, y = classification_data
    selector = GAFeatureSelector(clf, tiny_config).fit(X, y)
    with pytest.raises(ValueError):
        GAPlotter.plot_pareto_front(selector.result_)
