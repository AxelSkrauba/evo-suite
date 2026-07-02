"""03 · Multi-objective selection — the NSGA-II Pareto front.

Instead of committing to a single alpha, multi-objective mode optimises
performance and compression simultaneously and returns the whole Pareto front,
letting a decision-maker pick a trade-off after the fact.

If matplotlib is available, the front is also saved to a PNG next to this file.

Run:
    uv run python examples/03_multiobjective_pareto.py
"""

from __future__ import annotations

from pathlib import Path

from evo_gafs import GAConfig, GAFeatureSelector
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    selector = GAFeatureSelector(
        estimator=DecisionTreeClassifier(random_state=42),
        config=GAConfig(
            mode="multiobjective",
            population_size=40,
            n_generations=25,
            random_seed=42,
            verbose=False,
        ),
    )
    selector.fit(X, y)

    front = sorted(selector.result_.pareto_front, key=lambda p: p["n_features"])
    print(f"Pareto front ({len(front)} non-dominated solutions):\n")
    print(f"{'n_features':>10} | {'cv_score':>8} | {'compression':>11}")
    print("-" * 36)
    seen = set()
    for p in front:
        if p["n_features"] in seen:
            continue
        seen.add(p["n_features"])
        print(f"{p['n_features']:>10d} | {p['cv_score']:>8.4f} | {p['compression']:>10.1%}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        from evo_gafs import GAPlotter

        fig = GAPlotter.plot_pareto_front(selector.result_)
        out = Path(__file__).with_name("03_pareto_front.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\nSaved Pareto plot to {out.name}")
    except ImportError:
        print("\n(install evo-gafs[viz] to also render the Pareto plot)")


if __name__ == "__main__":
    main()
