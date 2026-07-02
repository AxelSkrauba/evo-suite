"""02 · Fitness metrics — relevance vs. cost trade-off.

`fitness_metric` controls how a candidate tree's relevance is scored:

- `mutual_info` (default): general-purpose, captures non-linear dependence.
- `correlation` / `spearman`: cheap association measures.
- `model_score`: directly optimises the downstream estimator's CV score —
  the most informative signal, but by far the most expensive (it trains a
  model for every candidate tree).

This example evolves the same number of features under each metric and
reports the resulting downstream CV score and wall-clock time.

Run:
    uv run python examples/evo-gpfe/02_fitness_metrics.py
"""

from __future__ import annotations

import time

from evo_gpfe import GPConfig, GPFeatureEngineer
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge


def main() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    print(f"{'metric':<14} | {'augmented CV (R^2)':>18} | {'time (s)':>9}")
    print("-" * 48)
    for metric in ("correlation", "mutual_info", "model_score"):
        start = time.time()
        engineer = GPFeatureEngineer(
            estimator=Ridge(alpha=1.0),
            config=GPConfig(
                population_size=60,
                n_generations=15,
                n_features_to_generate=3,
                fitness_metric=metric,
                random_seed=0,
                verbose=False,
            ),
        )
        engineer.fit(X, y)
        elapsed = time.time() - start
        print(f"{metric:<14} | {engineer.result_.augmented_cv_score:>18.4f} | {elapsed:>9.1f}")

    print(
        "\nmutual_info is a good general-purpose default; model_score directly"
        " targets downstream performance at a higher computational cost."
    )


if __name__ == "__main__":
    main()
