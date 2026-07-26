# Quickstart

Build a diversity-aware ensemble in a few lines.

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

`EvoEnsembleClassifier` (and its regression counterpart,
`EvoEnsembleRegressor`) are standard scikit-learn estimators:

- `fit(X, y)` pre-computes out-of-fold predictions for every candidate model,
  then runs the Genetic Algorithm to select which candidates to combine and
  how much weight to give each.
- `predict(X)` / `predict_proba(X)` (classification only) return the
  weighted-combination prediction.
- `score(X, y)` uses accuracy (classification) or R² (regression), the
  scikit-learn default for each mixin.
- the full outcome is available on `clf.result_` (an
  {class}`~evo_ens.EvoEnsResult`), including per-member weights, out-of-fold
  scores and the generation-by-generation history.
- `get_ensemble_info()` returns a `pandas.DataFrame` summary of the selected
  members.

## What just happened?

An individual is a vector of `2 * n_candidates` genes: the first half decide
which candidates are active, the second half their (pre-normalization)
weight. The Genetic Algorithm searches this space for the combination that
maximizes score while penalizing correlated errors between members (the
Q-statistic for classification, absolute Pearson correlation for
regression) — see [Concepts](guide/concepts.md).

## Next steps

- Understand the diversity penalty and the two optimization modes:
  [Concepts](guide/concepts.md).
- Tune the Genetic Algorithm and the candidate pool:
  [Configuration](guide/configuration.md).
- Compare across datasets: [Benchmarking](guide/benchmark.md).
- Use your own candidate pool and slot the ensemble into a `Pipeline`: see
  `04_custom_candidates_and_pipeline.py` in the repository's
  `examples/evo-ens/` directory.
