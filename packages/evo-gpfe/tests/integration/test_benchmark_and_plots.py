"""Integration tests for GPBenchmarkRunner and GPPlotter."""

from __future__ import annotations

import pytest

from evo_gpfe import GPBenchmarkRunner, GPFeatureEngineer, GPPlotter


def test_benchmark_runner(classification_data, regression_data, tiny_config):
    X_clf, y_clf = classification_data
    X_reg, y_reg = regression_data

    runner = GPBenchmarkRunner()
    runner.add_dataset("iris", X_clf, y_clf, task_type="classification")
    runner.add_dataset("synth", X_reg, y_reg, task_type="regression")

    results = runner.run(config=tiny_config, verbose=False)
    assert len(results) == 2
    for r in results:
        assert r["n_features_generated"] == tiny_config.n_features_to_generate
        assert len(r["expressions"]) == tiny_config.n_features_to_generate

    df = runner.report()
    assert len(df) == 2


def test_benchmark_runner_no_results_report():
    runner = GPBenchmarkRunner()
    df = runner.report()
    assert df.empty


def test_plot_feature_quality(classification_data, tiny_config):
    pytest.importorskip("matplotlib")
    X, y = classification_data
    engineer = GPFeatureEngineer(config=tiny_config).fit(X, y)
    fig = GPPlotter.plot_feature_quality(engineer.result_)
    assert fig is not None


def test_plot_benchmark_comparison(classification_data, tiny_config):
    pytest.importorskip("matplotlib")
    X, y = classification_data
    runner = GPBenchmarkRunner()
    runner.add_dataset("iris", X, y, task_type="classification")
    results = runner.run(config=tiny_config, verbose=False)
    fig = GPPlotter.plot_benchmark_comparison(results)
    assert fig is not None


def test_plot_mi_comparison(classification_data, tiny_config):
    pytest.importorskip("matplotlib")
    X, y = classification_data
    engineer = GPFeatureEngineer(config=tiny_config).fit(X, y)
    fig = GPPlotter.plot_mi_comparison(engineer.result_, X.to_numpy(), y.to_numpy())
    assert fig is not None
