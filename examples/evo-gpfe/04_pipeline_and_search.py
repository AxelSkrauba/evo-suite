"""04 · scikit-learn interoperability — Pipeline + GridSearchCV.

GPFeatureEngineer is a first-class scikit-learn transformer: it slots into a
Pipeline and is tunable with GridSearchCV (which relies on correct
get_params/set_params and clone).

Run:
    uv run python examples/evo-gpfe/04_pipeline_and_search.py
"""

from __future__ import annotations

from evo_gpfe import GPConfig, GPFeatureEngineer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    pipe = Pipeline(
        [
            ("gp", GPFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ]
    )

    base = {"population_size": 40, "n_generations": 10, "random_seed": 42, "verbose": False}
    param_grid = {
        "gp__config": [
            GPConfig(n_features_to_generate=2, **base),
            GPConfig(n_features_to_generate=4, **base),
        ]
    }

    search = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy")
    search.fit(X, y)

    best_cfg = search.best_params_["gp__config"]
    print(f"Best n_features_to_generate : {best_cfg.n_features_to_generate}")
    print(f"Best CV accuracy            : {search.best_score_:.4f}")


if __name__ == "__main__":
    main()
