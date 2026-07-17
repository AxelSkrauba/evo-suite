# Quickstart

Construct new, informative features for a model in a few lines.

```python
from sklearn.datasets import load_diabetes
from evo_gpfe import GPConfig, GPFeatureEngineer

X, y = load_diabetes(return_X_y=True, as_frame=True)

engineer = GPFeatureEngineer(
    config=GPConfig(population_size=80, n_generations=20, n_features_to_generate=3, verbose=False)
)
engineer.fit(X, y)

print(engineer.summary())
X_augmented = engineer.transform(X)
print("Output features:", list(engineer.get_feature_names_out()))
```

`GPFeatureEngineer` is a standard scikit-learn transformer:

- `fit(X, y)` evolves the expression trees (a target is required: GP optimises
  relevance to ``y``).
- `transform(X)` applies the learned trees, returning the generated features
  (concatenated to the originals when `augment_original=True`, the default).
- `get_feature_names_out()` returns the output column names, sklearn-style.
- the full outcome is available on `engineer.result_` (a
  {class}`~evo_gpfe.GPEngineeringResult`), including the discovered symbolic
  expressions and the CV-score improvement.

## What just happened?

Each candidate feature is an **expression tree** built from a fixed set of
protected mathematical primitives (`add`, `mul`, `protected_div`, `log`,
`sqrt`, ...). A genetic program evolves a population of trees to maximise
relevance to `y`, while a **sequential hall-of-fame** strategy penalises
redundancy with already-generated features and rewards smaller (more
parsimonious) trees — see [Concepts](guide/concepts.md).

## Next steps

- Understand the hall-of-fame strategy and the fitness formula:
  [Concepts](guide/concepts.md).
- Tune primitives, redundancy and parsimony:
  [Configuration](guide/configuration.md).
- Compare across datasets: [Benchmarking](guide/benchmark.md).
- Combine with `evo-gafs` (construct, then select): see the
  `06_gp_then_ga_pipeline.py` example in the repository's `examples/evo-gpfe/`
  directory.
