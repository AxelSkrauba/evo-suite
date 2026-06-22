"""02 · The alpha trade-off — accuracy vs compression (edge deployment).

`alpha` is evo-gafs's headline knob:

    fitness = alpha * cv_score + (1 - alpha) * compression

Sweeping it shows the explicit performance/size trade-off that matters when
deploying to memory- or compute-constrained targets (edge, IoT, embedded).
Lower alpha favours smaller feature subsets.

Run:
    uv run python examples/02_alpha_tradeoff.py
"""

from __future__ import annotations

from evo_gafs import GAConfig, GAFeatureSelector
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    n_total = X.shape[1]

    print(f"{'alpha':>6} | {'n_features':>10} | {'compression':>11} | {'cv_score':>8}")
    print("-" * 46)
    for alpha in (1.0, 0.9, 0.8, 0.6, 0.4):
        selector = GAFeatureSelector(
            estimator=DecisionTreeClassifier(random_state=42),
            config=GAConfig(
                population_size=25,
                n_generations=20,
                alpha=alpha,
                random_seed=42,
                verbose=False,
            ),
        )
        selector.fit(X, y)
        r = selector.result_
        print(
            f"{alpha:>6.2f} | {r.n_selected:>10d} | "
            f"{r.compression_ratio:>10.1%} | {r.best_cv_score:>8.4f}"
        )

    print(f"\n(total available features: {n_total})")
    print("Lower alpha trades a little accuracy for substantially fewer features.")


if __name__ == "__main__":
    main()
