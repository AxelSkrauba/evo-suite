# evo-ens - Evolutionary Ensemble Builder

[![PyPI](https://img.shields.io/pypi/v/evo-ens.svg)](https://pypi.org/project/evo-ens/)
[![Python versions](https://img.shields.io/pypi/pyversions/evo-ens.svg)](https://pypi.org/project/evo-ens/)
[![Docs](https://readthedocs.org/projects/evo-suite/badge/?version=latest)](https://evo-suite.readthedocs.io/en/latest/)
[![CI](https://github.com/AxelSkrauba/evo-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/AxelSkrauba/evo-suite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

A **scikit-learn-compatible** ensemble constructor for tabular data, powered
by [DEAP](https://github.com/DEAP/deap). `evo-ens` uses a Genetic Algorithm
to decide which candidate models to combine and how to weight them, trading
off predictive score against prediction diversity, complementing
[`evo-gafs`](../evo-gafs) (feature selection) and [`evo-gpfe`](../evo-gpfe)
(feature engineering) further upstream in the pipeline.

Part of the [`evo-suite`](../../README.md) family (import name: `evo_ens`).
Documentation: <https://evo-suite.readthedocs.io/>

## Why evo-ens?

| Capability | evo-ens |
|------------|---------|
| **Diversity-aware fitness**: score minus a Q-statistic / Pearson-correlation penalty | Yes |
| **Out-of-fold pre-computation**: candidates cross-validated once, evolution stays cheap | Yes |
| Native scikit-learn `EvoEnsembleClassifier` / `EvoEnsembleRegressor`, usable in a `Pipeline` | Yes |
| **Multi-objective NSGA-II** mode exposing the full score/compression Pareto front | Yes |
| `min_models` / `max_models` constraints for edge-deployment budgets | Yes |
| Built-in multi-dataset `EvoEnsBenchmarkRunner` | Yes |

## Installation

```bash
pip install evo-ens            # core
pip install "evo-ens[viz]"     # + matplotlib for the plotting helpers
```

## Quickstart

```python
from sklearn.datasets import load_breast_cancer
from evo_ens import EvoEnsConfig, EvoEnsembleClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

clf = EvoEnsembleClassifier(
    config=EvoEnsConfig(population_size=60, n_generations=40, diversity_beta=0.10, verbose=False)
)
clf.fit(X, y)

print(clf.result_.summary())
print(clf.get_ensemble_info())
```

`EvoEnsembleRegressor` follows the same API for regression targets, scoring
with R² by default.

## Documentation & examples

- **Full documentation** (user guide + API reference): <https://evo-suite.readthedocs.io/>
- **Runnable examples**: the repository's [`examples/evo-ens/`](../../examples/evo-ens) directory.

## Citation

```bibtex
@software{evo_ens,
  author    = {Skrauba, Axel},
  title     = {evo-ens: Evolutionary Ensemble Builder for tabular data},
  year      = {2026},
  version   = {0.1.0},
  url       = {https://github.com/AxelSkrauba/evo-suite}
}
```

## License

[MIT](../../LICENSE)
