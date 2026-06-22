"""05 · Where the wrapper shines — evo-gafs vs all-features vs a filter.

Wrapper selection optimises the subset *for the chosen model*. This compares,
at a matched subset size:

    1. All features (baseline)
    2. SelectKBest (a fast univariate filter)
    3. evo-gafs (the genetic wrapper)

The wrapper typically matches or beats the filter at the same number of
features, because it accounts for feature interactions the filter ignores.

Run:
    uv run python examples/05_compare_baselines.py
"""

from __future__ import annotations

import numpy as np
from evo_gafs import GAConfig, GAFeatureSelector
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def _cv_score(estimator, X, y) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return float(np.mean(cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")))


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    model = DecisionTreeClassifier(random_state=42)

    # 1. All features
    full = _cv_score(model, X, y)

    # 2. evo-gafs wrapper selection
    selector = GAFeatureSelector(
        estimator=DecisionTreeClassifier(random_state=42),
        config=GAConfig(population_size=25, n_generations=20, random_seed=42, verbose=False),
    )
    selector.fit(X, y)
    k = selector.result_.n_selected
    ga = _cv_score(model, selector.transform(X), y)

    # 3. Univariate filter at the SAME subset size k
    kbest = SelectKBest(f_classif, k=k).fit(X, y)
    flt = _cv_score(model, kbest.transform(X), y)

    print(f"{'method':<22} | {'n_features':>10} | {'cv_accuracy':>11}")
    print("-" * 50)
    print(f"{'all features':<22} | {X.shape[1]:>10d} | {full:>11.4f}")
    print(f"{'SelectKBest (filter)':<22} | {k:>10d} | {flt:>11.4f}")
    print(f"{'evo-gafs (wrapper)':<22} | {k:>10d} | {ga:>11.4f}")


if __name__ == "__main__":
    main()
