# Benchmarking

{class}`~evo_ens.EvoEnsBenchmarkRunner` runs `EvoEnsembleClassifier`/
`EvoEnsembleRegressor` over several datasets and collects a comparable
summary — improvement over the best single candidate, diversity score,
compression ratio and wall-clock time.

```python
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from evo_ens import EvoEnsBenchmarkRunner, EvoEnsConfig

runner = EvoEnsBenchmarkRunner()

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
runner.add_dataset("Breast Cancer", X, y, task_type="classification")

X, y = load_wine(return_X_y=True, as_frame=True)
runner.add_dataset("Wine", X, y, task_type="classification")

X, y = load_diabetes(return_X_y=True, as_frame=True)
runner.add_dataset("Diabetes", X, y, task_type="regression")

runner.run(config=EvoEnsConfig(population_size=60, n_generations=40, verbose=False))
print(runner.report())
```

`add_dataset(name, X, y, task_type=..., candidate_estimators=None)` selects
`EvoEnsembleClassifier` or `EvoEnsembleRegressor` per dataset based on
`task_type`; `run()` returns the raw per-dataset dictionaries (including the
fitted estimator and its `result_`), and `report()` renders them as a
`pandas.DataFrame`. See `05_benchmark_suite.py` for a runnable, multi-dataset
example, and {class}`~evo_ens.EvoEnsPlotter` (`plot_benchmark`,
`plot_evolution`, `plot_ensemble_composition`, `plot_pareto_front`) for the
corresponding visualizations (requires the `evo-ens[viz]` extra).
