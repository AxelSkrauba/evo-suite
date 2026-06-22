# Multi-objective selection (NSGA-II)

Single-objective mode collapses performance and compression into one number via
`alpha`. Multi-objective mode keeps them separate and optimises both at once
with **NSGA-II**, returning the full **Pareto front** of non-dominated
solutions.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from evo_gafs import GAConfig, GAFeatureSelector

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

selector = GAFeatureSelector(
    estimator=DecisionTreeClassifier(random_state=42),
    config=GAConfig(mode="multiobjective", population_size=40, n_generations=30, verbose=False),
)
selector.fit(X, y)

for point in sorted(selector.result_.pareto_front, key=lambda p: p["n_features"]):
    print(point["n_features"], round(point["cv_score"], 4))
```

Each entry of `result_.pareto_front` is a dict with `mask`, `cv_score`,
`compression` and `n_features`. Because the front contains *non-dominated*
trade-offs, you can choose the smallest subset that meets your accuracy target,
or the most accurate subset under a size budget.

## Visualising the front

With the `viz` extra installed:

```python
from evo_gafs import GAPlotter

fig = GAPlotter.plot_pareto_front(selector.result_)
fig.savefig("pareto.png", dpi=120, bbox_inches="tight")
```

## When to use which mode

- **Single-objective** when you already know the trade-off you want (set `alpha`).
- **Multi-objective** when you want to *discover* the trade-off curve and decide
  afterwards, or when reporting the full performance/size frontier matters
  (e.g. in research).
