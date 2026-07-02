"""01 · Quickstart — fit / transform / inspect.

The smallest end-to-end use: select features for a classifier on the
breast-cancer dataset and inspect what was kept.

Run:
    uv run python examples/01_quickstart.py
"""

from __future__ import annotations

from evo_gafs import GAConfig, GAFeatureSelector
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    selector = GAFeatureSelector(
        estimator=DecisionTreeClassifier(random_state=42),
        config=GAConfig(population_size=20, n_generations=15, random_seed=42, verbose=False),
    )
    selector.fit(X, y)

    print(selector.summary())
    print("\nSelected feature names:", selector.result_.selected_feature_names)
    print("Support mask (indices):", selector.get_support(indices=True).tolist())
    print("Transformed shape:", selector.transform(X).shape)


if __name__ == "__main__":
    main()
