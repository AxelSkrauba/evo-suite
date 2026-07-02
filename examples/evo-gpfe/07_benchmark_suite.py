"""07 · Benchmarking across datasets.

`GPBenchmarkRunner` compares, per dataset, the baseline CV score (original
features) against the GP-augmented and GP-only scores — handy for reporting
and for sanity-checking the engineer on several problems at once.

Run:
    uv run python examples/evo-gpfe/07_benchmark_suite.py
"""

from __future__ import annotations

from evo_gpfe import GPBenchmarkRunner, GPConfig
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine


def main() -> None:
    runner = GPBenchmarkRunner()
    runner.add_dataset(
        "wine", *load_wine(return_X_y=True, as_frame=True), task_type="classification"
    )
    runner.add_dataset(
        "breast_cancer",
        *load_breast_cancer(return_X_y=True, as_frame=True),
        task_type="classification",
    )
    runner.add_dataset(
        "diabetes", *load_diabetes(return_X_y=True, as_frame=True), task_type="regression"
    )

    runner.run(
        config=GPConfig(
            population_size=60, n_generations=15, n_features_to_generate=3, verbose=False
        ),
        verbose=False,
    )
    runner.report()


if __name__ == "__main__":
    main()
