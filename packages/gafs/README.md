# gafs — Genetic Algorithm Feature Selector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

A **scikit-learn-compatible** wrapper feature selector for tabular data, powered
by [DEAP](https://github.com/DEAP/deap). `gafs` searches for the subset of
features that maximises a cross-validated score of your model, and lets you
explicitly trade raw performance for a smaller feature set — useful for edge
deployment.

Part of the [`evo-suite`](../../README.md) monorepo.

## Why gafs?

| Capability | gafs |
|------------|------|
| Single-objective **weighted** fitness with a configurable `alpha` (performance ↔ compression) | ✓ |
| **Multi-objective** NSGA-II with an accessible Pareto front | ✓ |
| **Repair operator** guaranteeing a minimum number of features | ✓ |
| Evaluation **cache** to skip repeated genomes | ✓ |
| Native scikit-learn `fit`/`transform`/`get_support`, usable in a `Pipeline` | ✓ |
| Built-in multi-dataset `BenchmarkRunner` | ✓ |

## Installation

```bash
pip install gafs            # core
pip install gafs[viz]       # + matplotlib for the plotting helpers
```

## Quickstart

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from gafs import GAFeatureSelector, GAConfig

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

selector = GAFeatureSelector(
    estimator=DecisionTreeClassifier(random_state=42),
    config=GAConfig(population_size=30, n_generations=20, alpha=0.8, verbose=False),
)
selector.fit(X, y)

print(selector.summary())
X_reduced = selector.transform(X)
print("Selected:", selector.get_support(indices=True))
```

### Multi-objective (Pareto front)

```python
config = GAConfig(mode="multiobjective", population_size=40, n_generations=30, verbose=False)
selector = GAFeatureSelector(estimator=DecisionTreeClassifier(random_state=42), config=config)
selector.fit(X, y)

for point in selector.result_.pareto_front:
    print(point["n_features"], point["cv_score"])
```

### In a scikit-learn pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", GAFeatureSelector(estimator=DecisionTreeClassifier(), config=config)),
    ("clf", SVC()),
])
pipe.fit(X, y)
```

## The `alpha` trade-off (single-objective)

```
fitness = alpha * cv_score + (1 - alpha) * compression
compression = 1 - n_selected / n_total
```

- `alpha = 1.0` → pure wrapper (performance only)
- `alpha ≈ 0.7` → balanced, good default for edge deployment

## Citation

```bibtex
@software{gafs,
  author    = {Skrauba, Axel},
  title     = {gafs: Genetic Algorithm Feature Selector for tabular data},
  year      = {2026},
  version   = {0.1.0},
  url       = {https://github.com/AxelSkrauba/evo-suite}
}
```

## License

[MIT](../../LICENSE)
