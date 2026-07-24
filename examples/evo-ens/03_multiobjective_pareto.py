"""03 - Multiobjective (NSGA-II) -- explore the score/compression Pareto front.

'multiobjective' mode returns every non-dominated (score, compression)
solution instead of a single ensemble, useful for choosing a smaller
ensemble for edge deployment without re-running the search.

Run:
    uv run python examples/evo-ens/03_multiobjective_pareto.py
"""

from __future__ import annotations

from evo_ens import EvoEnsConfig, EvoEnsembleClassifier
from sklearn.datasets import load_wine


def main() -> None:
    X, y = load_wine(return_X_y=True, as_frame=True)

    clf = EvoEnsembleClassifier(
        config=EvoEnsConfig(
            mode="multiobjective",
            population_size=60,
            n_generations=40,
            random_seed=42,
            verbose=False,
        )
    )
    clf.fit(X, y)

    print(f"Pareto front: {len(clf.result_.pareto_front)} solutions\n")
    for sol in sorted(clf.result_.pareto_front, key=lambda s: s.n_models):
        print(
            f"  n_models={sol.n_models:2d} | score={sol.score:.4f} | compression={sol.compression:.0%}"
        )

    print("\nSelected (highest-score) solution:")
    print(clf.result_.summary())


if __name__ == "__main__":
    main()
