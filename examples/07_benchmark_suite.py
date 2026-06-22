"""07 · Benchmarking across datasets.

`BenchmarkRunner` compares, per dataset, the model's score using all features
vs the evo-gafs-selected subset, plus compression and timing — handy for
reporting and for sanity-checking the selector on several problems at once.

Run:
    uv run python examples/07_benchmark_suite.py
"""

from __future__ import annotations

from evo_gafs import BenchmarkRunner, GAConfig
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    runner = BenchmarkRunner()
    runner.add_dataset(
        "iris", *load_iris(return_X_y=True, as_frame=True), task_type="classification"
    )
    runner.add_dataset(
        "wine", *load_wine(return_X_y=True, as_frame=True), task_type="classification"
    )
    runner.add_dataset(
        "breast_cancer",
        *load_breast_cancer(return_X_y=True, as_frame=True),
        task_type="classification",
    )

    runner.run(
        estimator=DecisionTreeClassifier(random_state=42),
        config=GAConfig(population_size=20, n_generations=15, random_seed=42, verbose=False),
        verbose=False,
    )
    runner.report()


if __name__ == "__main__":
    main()
