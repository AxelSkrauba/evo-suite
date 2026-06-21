"""Integration tests for multi-objective (NSGA-II) selection."""

from __future__ import annotations

from evo_gafs import GAConfig, GAFeatureSelector


def test_multiobjective_produces_pareto_front(classification_data, clf):
    X, y = classification_data
    config = GAConfig(
        mode="multiobjective",
        population_size=16,
        n_generations=6,
        random_seed=0,
        verbose=False,
    )
    selector = GAFeatureSelector(clf, config).fit(X, y)
    front = selector.result_.pareto_front
    assert front is not None
    assert len(front) >= 1
    for point in front:
        assert set(point) == {"mask", "cv_score", "compression", "n_features"}
        assert point["n_features"] == sum(point["mask"])


def test_single_objective_has_no_pareto_front(classification_data, clf, tiny_config):
    X, y = classification_data
    selector = GAFeatureSelector(clf, tiny_config).fit(X, y)
    assert selector.result_.pareto_front is None
