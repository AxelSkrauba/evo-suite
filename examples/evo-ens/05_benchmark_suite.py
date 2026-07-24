"""05 - Benchmark suite -- compare EvoEnsemble against the best single model
across several datasets.

Run:
    uv run python examples/evo-ens/05_benchmark_suite.py
"""

from __future__ import annotations

from evo_ens import EvoEnsBenchmarkRunner, EvoEnsConfig
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine


def main() -> None:
    runner = EvoEnsBenchmarkRunner()

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    runner.add_dataset("Breast Cancer", X, y, task_type="classification")

    X, y = load_wine(return_X_y=True, as_frame=True)
    runner.add_dataset("Wine", X, y, task_type="classification")

    X, y = load_iris(return_X_y=True, as_frame=True)
    runner.add_dataset("Iris", X, y, task_type="classification")

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    runner.add_dataset("Diabetes", X, y, task_type="regression")

    runner.run(
        config=EvoEnsConfig(population_size=40, n_generations=30, random_seed=42, verbose=False),
        verbose=False,
    )
    print(runner.report().to_string(index=False))


if __name__ == "__main__":
    main()
