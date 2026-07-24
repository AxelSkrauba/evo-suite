"""01 - Quickstart -- build a classification ensemble and inspect it.

The smallest end-to-end use: let the Genetic Algorithm pick which default
candidate models to combine (and how to weight them) for a classification
problem.

Run:
    uv run python examples/evo-ens/01_quickstart.py
"""

from __future__ import annotations

from evo_ens import EvoEnsConfig, EvoEnsembleClassifier
from sklearn.datasets import load_breast_cancer


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    clf = EvoEnsembleClassifier(
        config=EvoEnsConfig(
            population_size=60,
            n_generations=40,
            diversity_beta=0.10,
            random_seed=42,
            verbose=False,
        )
    )
    clf.fit(X, y)

    print(clf.result_.summary())
    print("\nEnsemble composition:")
    print(clf.get_ensemble_info().to_string(index=False))


if __name__ == "__main__":
    main()
