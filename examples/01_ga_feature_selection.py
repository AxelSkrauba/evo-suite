"""Example: GA feature selection with `gafs`.

Run with the workspace environment:

    uv run python examples/01_ga_feature_selection.py

It demonstrates the three core workflows:

1. Single-objective (weighted) selection with the `alpha` trade-off.
2. Multi-objective (NSGA-II) selection with a Pareto front.
3. Multi-dataset benchmarking.
"""

from __future__ import annotations

from gafs import BenchmarkRunner, GAConfig, GAFeatureSelector
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.tree import DecisionTreeClassifier


def single_objective() -> None:
    print("\n=== 1. Single-objective selection (alpha=0.8) ===")
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    selector = GAFeatureSelector(
        estimator=DecisionTreeClassifier(random_state=42),
        config=GAConfig(population_size=30, n_generations=20, alpha=0.8, verbose=False),
    )
    selector.fit(X, y)
    print(selector.summary())
    print("Reduced shape:", selector.transform(X).shape)


def multi_objective() -> None:
    print("\n=== 2. Multi-objective selection (NSGA-II) ===")
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    selector = GAFeatureSelector(
        estimator=DecisionTreeClassifier(random_state=42),
        config=GAConfig(
            mode="multiobjective",
            population_size=40,
            n_generations=25,
            verbose=False,
        ),
    )
    selector.fit(X, y)
    print(f"Pareto front has {len(selector.result_.pareto_front)} solutions:")
    for point in sorted(selector.result_.pareto_front, key=lambda p: p["n_features"]):
        print(f"  n_features={point['n_features']:2d}  cv_score={point['cv_score']:.4f}")


def benchmark() -> None:
    print("\n=== 3. Multi-dataset benchmark ===")
    runner = BenchmarkRunner()
    runner.add_dataset("breast_cancer", *load_breast_cancer(return_X_y=True, as_frame=True))
    runner.add_dataset("wine", *load_wine(return_X_y=True, as_frame=True))
    runner.run(
        estimator=DecisionTreeClassifier(random_state=42),
        config=GAConfig(population_size=20, n_generations=15, verbose=False),
        verbose=False,
    )
    runner.report()


if __name__ == "__main__":
    single_objective()
    multi_objective()
    benchmark()
