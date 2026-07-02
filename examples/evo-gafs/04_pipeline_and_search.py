"""04 · scikit-learn interoperability — Pipeline + GridSearchCV.

evo-gafs is a first-class scikit-learn estimator: it slots into a Pipeline and
is tunable with GridSearchCV (which relies on correct get_params/set_params and
clone). Here we grid over two GAConfig presets to pick the better alpha.

Run:
    uv run python examples/04_pipeline_and_search.py
"""

from __future__ import annotations

from evo_gafs import GAConfig, GAFeatureSelector
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "selector",
                GAFeatureSelector(estimator=DecisionTreeClassifier(random_state=42)),
            ),
            ("clf", SVC()),
        ]
    )

    base = {"population_size": 12, "n_generations": 6, "random_seed": 42, "verbose": False}
    param_grid = {
        "selector__config": [
            GAConfig(alpha=0.9, **base),  # accuracy-leaning
            GAConfig(alpha=0.6, **base),  # compression-leaning
        ]
    }

    search = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy")
    search.fit(X, y)

    best_cfg = search.best_params_["selector__config"]
    best_selector = search.best_estimator_.named_steps["selector"]
    print(f"Best alpha            : {best_cfg.alpha}")
    print(f"Best CV accuracy      : {search.best_score_:.4f}")
    print(f"Features kept (best)  : {best_selector.result_.n_selected} / {X.shape[1]}")


if __name__ == "__main__":
    main()
