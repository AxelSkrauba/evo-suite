"""06 · Regression — selecting features for a regressor.

evo-gafs supports regression out of the box: pass a regressor and a regression
scoring metric (task type is auto-detected from a continuous target).

Run:
    uv run python examples/06_regression.py
"""

from __future__ import annotations

from evo_gafs import GAConfig, GAFeatureSelector
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


def main() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    selector = GAFeatureSelector(
        estimator=RandomForestRegressor(n_estimators=50, random_state=42),
        scoring="r2",
        config=GAConfig(
            population_size=20, n_generations=15, alpha=0.85, random_seed=42, verbose=False
        ),
    )
    selector.fit(X, y)

    r = selector.result_
    print("Task                : regression (auto-detected)")
    print(f"Features            : {X.shape[1]} -> {r.n_selected}")
    print(f"Best CV R^2         : {r.best_cv_score:.4f}")
    print(f"Selected            : {r.selected_feature_names}")


if __name__ == "__main__":
    main()
