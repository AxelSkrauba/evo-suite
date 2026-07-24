"""Integration tests for EvoEnsBenchmarkRunner and EvoEnsPlotter."""

from __future__ import annotations

import pytest

from evo_ens import EvoEnsBenchmarkRunner, EvoEnsPlotter


class TestBenchmarkRunner:
    def test_run_classification_and_regression(
        self, binary_classification_data, regression_data, tiny_config
    ):
        Xc, yc = binary_classification_data
        Xr, yr = regression_data
        runner = EvoEnsBenchmarkRunner()
        runner.add_dataset("clf-ds", Xc, yc, task_type="classification")
        runner.add_dataset("reg-ds", Xr, yr, task_type="regression")

        results = runner.run(config=tiny_config, verbose=False)
        assert len(results) == 2
        assert results[0]["task_type"] == "classification"
        assert results[1]["task_type"] == "regression"

        report = runner.report()
        assert len(report) == 2
        assert "Dataset" in report.columns

    def test_invalid_task_type_raises(self):
        runner = EvoEnsBenchmarkRunner()
        with pytest.raises(ValueError):
            runner.add_dataset("bad", [[0, 1]], [0], task_type="bogus")

    def test_report_without_run_is_empty(self):
        runner = EvoEnsBenchmarkRunner()
        assert runner.report().empty

    def test_run_does_not_mutate_base_config(self, binary_classification_data, tiny_config):
        Xc, yc = binary_classification_data
        runner = EvoEnsBenchmarkRunner()
        runner.add_dataset("clf-ds", Xc, yc, task_type="classification")
        before = tiny_config.to_dict()
        runner.run(config=tiny_config, verbose=False)
        assert tiny_config.to_dict() == before


class TestPlotter:
    def test_plot_evolution(self, binary_classification_data, tiny_config):
        from evo_ens import EvoEnsembleClassifier

        X, y = binary_classification_data
        clf = EvoEnsembleClassifier(config=tiny_config)
        clf.fit(X, y)
        fig = EvoEnsPlotter.plot_evolution(clf.result_)
        assert fig is not None

    def test_plot_ensemble_composition(self, binary_classification_data, tiny_config):
        from evo_ens import EvoEnsembleClassifier

        X, y = binary_classification_data
        clf = EvoEnsembleClassifier(config=tiny_config)
        clf.fit(X, y)
        fig = EvoEnsPlotter.plot_ensemble_composition(clf.result_)
        assert fig is not None

    def test_plot_pareto_front(self, binary_classification_data, tiny_nsga2_config):
        from evo_ens import EvoEnsembleClassifier

        X, y = binary_classification_data
        clf = EvoEnsembleClassifier(config=tiny_nsga2_config)
        clf.fit(X, y)
        fig = EvoEnsPlotter.plot_pareto_front(clf.result_)
        assert fig is not None

    def test_plot_pareto_front_raises_without_front(self, binary_classification_data, tiny_config):
        from evo_ens import EvoEnsembleClassifier

        X, y = binary_classification_data
        clf = EvoEnsembleClassifier(config=tiny_config)
        clf.fit(X, y)
        with pytest.raises(ValueError):
            EvoEnsPlotter.plot_pareto_front(clf.result_)

    def test_plot_benchmark(self, binary_classification_data, tiny_config):
        runner = EvoEnsBenchmarkRunner()
        Xc, yc = binary_classification_data
        runner.add_dataset("clf-ds", Xc, yc, task_type="classification")
        results = runner.run(config=tiny_config, verbose=False)
        fig = EvoEnsPlotter.plot_benchmark(results)
        assert fig is not None
