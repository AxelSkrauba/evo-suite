# evo-gpfe - Genetic Programming Feature Engineer

[![PyPI](https://img.shields.io/pypi/v/evo-gpfe.svg)](https://pypi.org/project/evo-gpfe/)
[![Python versions](https://img.shields.io/pypi/pyversions/evo-gpfe.svg)](https://pypi.org/project/evo-gpfe/)
[![Docs](https://readthedocs.org/projects/evo-suite/badge/?version=latest)](https://evo-suite.readthedocs.io/en/latest/)
[![CI](https://github.com/AxelSkrauba/evo-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/AxelSkrauba/evo-suite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

A **scikit-learn-compatible** symbolic feature constructor for tabular data,
powered by [DEAP](https://github.com/DEAP/deap). `evo-gpfe` evolves expression
trees that *combine* the original features into new, informative ones,
complementing [`evo-gafs`](../evo-gafs) (which *selects* among existing
features).

Part of the [`evo-suite`](../../README.md) family (import name: `evo_gpfe`).
Documentation: <https://evo-suite.readthedocs.io/>

## Why evo-gpfe?

| Capability | evo-gpfe |
|------------|----------|
| **Sequential hall-of-fame** strategy: relevance minus redundancy minus parsimony | Yes |
| **Protected primitives** (safe division, log, sqrt, ...) so any random tree evaluates | Yes |
| Multiple relevance metrics: `mutual_info`, `correlation`, `spearman`, `model_score` | Yes |
| **Anti-bloat**: static tree-height limit and a parsimony penalty | Yes |
| Native scikit-learn `fit`/`transform`/`get_feature_names_out`, usable in a `Pipeline` | Yes |
| Built-in multi-dataset `GPBenchmarkRunner` | Yes |

## Installation

```bash
pip install evo-gpfe            # core
pip install "evo-gpfe[viz]"     # + matplotlib for the plotting helpers
```

## Quickstart

```python
from sklearn.datasets import load_diabetes
from evo_gpfe import GPFeatureEngineer, GPConfig

X, y = load_diabetes(return_X_y=True, as_frame=True)

engineer = GPFeatureEngineer(
    config=GPConfig(population_size=80, n_generations=20, n_features_to_generate=3, verbose=False)
)
engineer.fit(X, y)

print(engineer.summary())
X_augmented = engineer.transform(X)
print("Output features:", list(engineer.get_feature_names_out()))
```

### The GP-then-GA combo

`evo-gpfe` and `evo-gafs` are independent but complementary: construct new
features with GP, then select the best subset (original + generated) with GA.

```python
from evo_gafs import GAFeatureSelector, GAConfig

X_aug = engineer.fit_transform(X, y)
selector = GAFeatureSelector(estimator=..., feature_names=list(engineer.get_feature_names_out()))
selector.fit(X_aug, y)
```

See `06_gp_then_ga_pipeline.py` in the repository's `examples/evo-gpfe/`
directory for the full example.

## Documentation & examples

- **Full documentation** (user guide + API reference): <https://evo-suite.readthedocs.io/>
- **Runnable examples**: the repository's [`examples/evo-gpfe/`](../../examples/evo-gpfe) directory.

## Citation

```bibtex
@software{evo_gpfe,
  author    = {Skrauba, Axel},
  title     = {evo-gpfe: Genetic Programming Feature Engineer for tabular data},
  year      = {2026},
  version   = {0.1.0},
  url       = {https://github.com/AxelSkrauba/evo-suite}
}
```

## License

[MIT](../../LICENSE)
