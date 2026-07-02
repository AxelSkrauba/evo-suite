"""06 · The evo-suite combo — GP construction, then GA selection.

`evo-gpfe` and `evo-gafs` are complementary: GP *constructs* new candidate
features, and GA *selects* the best subset from the union of original and
generated features. Chaining them lets the downstream model benefit from both
richer features and a compact, edge-friendly subset.

This example requires both packages (`pip install evo-gpfe evo-gafs`); the
combination is optional — neither package depends on the other.

Run:
    uv run python examples/evo-gpfe/06_gp_then_ga_pipeline.py
"""

from __future__ import annotations

from evo_gpfe import GPConfig, GPFeatureEngineer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

try:
    from evo_gafs import GAConfig, GAFeatureSelector
except ImportError as exc:  # pragma: no cover - optional combo dependency
    raise SystemExit("This example needs both packages: `pip install evo-gpfe evo-gafs`.") from exc


def _cv_score(estimator, X, y) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return float(cross_val_score(estimator, X, y, cv=cv, scoring="accuracy").mean())


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    model = DecisionTreeClassifier(random_state=42)

    baseline = _cv_score(model, X, y)

    # 1. GP constructs new candidate features (augmenting the originals).
    engineer = GPFeatureEngineer(
        config=GPConfig(
            population_size=60,
            n_generations=15,
            n_features_to_generate=4,
            random_seed=42,
            verbose=False,
        )
    )
    X_augmented = engineer.fit_transform(X, y)
    augmented_names = list(engineer.get_feature_names_out())

    # 2. GA selects the best subset from the augmented feature set.
    selector = GAFeatureSelector(
        estimator=DecisionTreeClassifier(random_state=42),
        config=GAConfig(population_size=30, n_generations=20, alpha=0.8, verbose=False),
        feature_names=augmented_names,
    )
    selector.fit(X_augmented, y)
    X_final = selector.transform(X_augmented)
    final_score = _cv_score(model, X_final, y)

    print(f"{'stage':<28} | {'n_features':>10} | {'cv_accuracy':>11}")
    print("-" * 56)
    print(f"{'original':<28} | {X.shape[1]:>10d} | {baseline:>11.4f}")
    print(f"{'+ GP (augmented)':<28} | {X_augmented.shape[1]:>10d} | {'-':>11}")
    print(f"{'+ GP -> GA (selected)':<28} | {X_final.shape[1]:>10d} | {final_score:>11.4f}")
    print("\nSelected features:", selector.result_.selected_feature_names)


if __name__ == "__main__":
    main()
