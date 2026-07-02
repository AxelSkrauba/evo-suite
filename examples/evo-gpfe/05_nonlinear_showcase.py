"""05 · Where GP feature engineering shines — interactions and non-linearity.

A linear model cannot represent feature *interactions* or non-linear terms
directly. Here the target combines a product interaction, a quadratic term, a
square root and a linear term — plus one irrelevant feature:

    y = 3*x0*x1 + 2*x2^2 + sqrt(|x3|) + 1.5*x4 + noise      (x5 is noise-only)

A plain Ridge regression on the raw features cannot capture the product or the
quadratic/sqrt terms, so its CV R^2 is poor (even negative). GP feature
engineering can discover expressions that approximate this structure — often
recovering the interaction almost exactly — closing most of the gap.

Run:
    uv run python examples/evo-gpfe/05_nonlinear_showcase.py
"""

from __future__ import annotations

import numpy as np
from evo_gpfe import GPConfig, GPFeatureEngineer
from sklearn.linear_model import Ridge


def main() -> None:
    rng = np.random.default_rng(42)
    n_samples = 400
    X = rng.uniform(-3, 3, size=(n_samples, 6))  # x5 is irrelevant noise
    y = (
        3 * X[:, 0] * X[:, 1]
        + 2 * X[:, 2] ** 2
        + np.sqrt(np.abs(X[:, 3]))
        + 1.5 * X[:, 4]
        + rng.normal(scale=1.0, size=n_samples)
    )

    engineer = GPFeatureEngineer(
        estimator=Ridge(alpha=1.0),
        config=GPConfig(
            population_size=150,
            n_generations=30,
            n_features_to_generate=4,
            max_tree_height=5,
            random_seed=42,
            verbose=False,
        ),
    )
    engineer.fit(X, y)
    result = engineer.result_

    print(f"Baseline CV R^2 (linear model, raw features): {result.baseline_cv_score:.4f}")
    print(f"Augmented CV R^2 (+ GP features):              {result.augmented_cv_score:.4f}")
    print(f"Improvement:                                   {result.score_improvement:+.4f}")
    print("\nDiscovered expressions:")
    for gf in result.generated_features:
        print(f"  gp_{gf.feature_index}: {gf.expression}  (MI={gf.mi_score:.3f})")
    print("\nNote how the interaction term x0*x1 is often recovered almost exactly.")


if __name__ == "__main__":
    main()
