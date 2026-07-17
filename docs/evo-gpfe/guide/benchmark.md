# Benchmarking

{class}`~evo_gpfe.GPBenchmarkRunner` compares, per dataset:

- the baseline CV score with the **original** features,
- the CV score with original **+ generated** features (augmented), and
- the CV score with **only** the generated features,

alongside the discovered expressions and the maximum mutual information among
original vs. generated features.

```python
from sklearn.datasets import load_breast_cancer, load_wine
from evo_gpfe import GPBenchmarkRunner, GPConfig

runner = GPBenchmarkRunner()
runner.add_dataset("wine", *load_wine(return_X_y=True, as_frame=True), task_type="classification")
runner.add_dataset(
    "breast_cancer",
    *load_breast_cancer(return_X_y=True, as_frame=True),
    task_type="classification",
)

results = runner.run(
    config=GPConfig(population_size=60, n_generations=15, n_features_to_generate=3, verbose=False),
    verbose=False,
)
df = runner.report()   # also returns a pandas DataFrame
```

Pass `estimator=` to use a specific downstream model instead of the per-task
default (`LogisticRegression` / `Ridge`).
