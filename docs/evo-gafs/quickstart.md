# Quickstart

Select the best feature subset for a model in a few lines.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from evo_gafs import GAConfig, GAFeatureSelector

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

selector = GAFeatureSelector(
    estimator=DecisionTreeClassifier(random_state=42),
    config=GAConfig(population_size=30, n_generations=20, alpha=0.8, verbose=False),
)
selector.fit(X, y)

print(selector.summary())
X_reduced = selector.transform(X)
print("Selected indices:", selector.get_support(indices=True))
```

`GAFeatureSelector` is a standard scikit-learn transformer:

- `fit(X, y)` runs the genetic search.
- `transform(X)` projects `X` onto the selected features.
- `get_support(indices=...)` returns the boolean mask or the selected indices.
- the full outcome is available on `selector.result_` (a
  {class}`~evo_gafs.SelectionResult`).

## What just happened?

`evo-gafs` is a **wrapper** method: each candidate feature subset is scored by
cross-validating your `estimator` on it. A genetic algorithm evolves a
population of binary masks towards the best subset, where "best" combines model
performance and feature compression according to `alpha`
(see [Concepts](guide/concepts.md)).

## Next steps

- Tune the accuracy/size trade-off: [Configuration](guide/configuration.md).
- Explore the full trade-off curve: [Multi-objective](guide/multiobjective.md).
- Plug it into a pipeline: [Pipelines](guide/pipeline.md).
- Browse runnable scripts in the repository's `examples/` directory.
