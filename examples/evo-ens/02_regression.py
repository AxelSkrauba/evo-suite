"""02 - Regression -- build a weighted-average regression ensemble.

Shows EvoEnsembleRegressor on a regression dataset, using the absolute
Pearson correlation between members' predictions as the diversity penalty.

Run:
    uv run python examples/evo-ens/02_regression.py
"""

from __future__ import annotations

from evo_ens import EvoEnsConfig, EvoEnsembleRegressor
from sklearn.datasets import fetch_california_housing


def main() -> None:
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X, y = X.sample(n=2000, random_state=0), y.sample(n=2000, random_state=0)

    reg = EvoEnsembleRegressor(
        config=EvoEnsConfig(
            population_size=60,
            n_generations=40,
            scoring="r2",
            diversity_beta=0.15,
            random_seed=42,
            verbose=False,
        )
    )
    reg.fit(X, y)

    print(reg.result_.summary())


if __name__ == "__main__":
    main()
