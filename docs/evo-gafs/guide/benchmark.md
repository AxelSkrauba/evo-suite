# Benchmarking

{class}`~evo_gafs.BenchmarkRunner` compares, per dataset, the model's score
using **all** features against the **evo-gafs-selected** subset, alongside the
compression ratio and timing. It is useful both for sanity-checking the selector
across problems and for reporting.

```python
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.tree import DecisionTreeClassifier
from evo_gafs import BenchmarkRunner, GAConfig

runner = BenchmarkRunner()
runner.add_dataset("wine", *load_wine(return_X_y=True, as_frame=True), task_type="classification")
runner.add_dataset(
    "breast_cancer",
    *load_breast_cancer(return_X_y=True, as_frame=True),
    task_type="classification",
)

results = runner.run(
    estimator=DecisionTreeClassifier(random_state=42),
    config=GAConfig(population_size=20, n_generations=15, verbose=False),
    verbose=False,
)
df = runner.report()   # also returns a pandas DataFrame
```

Each result records the baseline score, the selected-subset score, the score
delta, compression, evaluations and wall-clock time. Use
`estimator_regression=` to supply a separate model for regression datasets in a
mixed benchmark.
