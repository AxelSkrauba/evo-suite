"""03 · Redundancy penalty and parsimony — controlling diversity and size.

The sequential hall-of-fame strategy penalises new features that correlate too
much with already-generated ones (`redundancy_beta`) and favours smaller trees
(`parsimony_coeff`). This example shows both knobs in action: a low
`redundancy_beta` tends to produce near-duplicate features, while a higher
value spreads the generated features across the input space.

Run:
    uv run python examples/evo-gpfe/03_redundancy_and_parsimony.py
"""

from __future__ import annotations

import numpy as np
from evo_gpfe import GPConfig, GPFeatureEngineer
from sklearn.datasets import load_breast_cancer


def _max_pairwise_correlation(values: list) -> float:
    if len(values) < 2:
        return 0.0
    corr = np.corrcoef(np.column_stack(values), rowvar=False)
    np.fill_diagonal(corr, 0.0)
    return float(np.max(np.abs(corr)))


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    print(f"{'redundancy_beta':>16} | {'max pairwise |corr|':>20} | {'mean n_nodes':>12}")
    print("-" * 56)
    for beta in (0.0, 0.3, 0.7):
        engineer = GPFeatureEngineer(
            config=GPConfig(
                population_size=60,
                n_generations=15,
                n_features_to_generate=4,
                redundancy_beta=beta,
                random_seed=0,
                verbose=False,
            )
        )
        engineer.fit(X, y)
        values = [gf.values for gf in engineer.result_.generated_features]
        sizes = [gf.n_nodes for gf in engineer.result_.generated_features]
        print(
            f"{beta:>16.1f} | {_max_pairwise_correlation(values):>20.3f} | {np.mean(sizes):>12.1f}"
        )

    print(f"\n{'parsimony_coeff':>16} | {'mean n_nodes':>12}")
    print("-" * 32)
    for coeff in (0.0, 0.01, 0.05):
        engineer = GPFeatureEngineer(
            config=GPConfig(
                population_size=60,
                n_generations=15,
                n_features_to_generate=3,
                parsimony_coeff=coeff,
                random_seed=0,
                verbose=False,
            )
        )
        engineer.fit(X, y)
        sizes = [gf.n_nodes for gf in engineer.result_.generated_features]
        print(f"{coeff:>16.2f} | {np.mean(sizes):>12.1f}")


if __name__ == "__main__":
    main()
