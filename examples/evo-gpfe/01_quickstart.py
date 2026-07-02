"""01 · Quickstart — construct new features and inspect the expressions.

The smallest end-to-end use: evolve a handful of symbolic features for a
regression model and see what was discovered.

Run:
    uv run python examples/evo-gpfe/01_quickstart.py
"""

from __future__ import annotations

from evo_gpfe import GPConfig, GPFeatureEngineer
from sklearn.datasets import load_diabetes


def main() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    engineer = GPFeatureEngineer(
        config=GPConfig(
            population_size=80,
            n_generations=20,
            n_features_to_generate=3,
            random_seed=42,
            verbose=False,
        )
    )
    engineer.fit(X, y)

    print(engineer.summary())
    print("\nOutput feature names:", list(engineer.get_feature_names_out()))
    print("Transformed shape:", engineer.transform(X).shape)


if __name__ == "__main__":
    main()
