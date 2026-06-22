# Examples

Runnable, self-contained examples for the `evo-suite` packages. Each script uses
a small configuration so it finishes quickly, and is executed in CI to stay
valid.

## evo-gafs

| Example | Highlights |
|---------|------------|
| [`01_quickstart.py`](01_quickstart.py) | Minimal `fit` / `transform` / `get_support` |
| [`02_alpha_tradeoff.py`](02_alpha_tradeoff.py) | The `alpha` accuracy↔compression knob (edge deployment) |
| [`03_multiobjective_pareto.py`](03_multiobjective_pareto.py) | NSGA-II Pareto front (+ optional plot) |
| [`04_pipeline_and_search.py`](04_pipeline_and_search.py) | `Pipeline` + `GridSearchCV` interoperability |
| [`05_compare_baselines.py`](05_compare_baselines.py) | Wrapper vs all-features vs univariate filter |
| [`06_regression.py`](06_regression.py) | Regression task with auto-detected task type |
| [`07_benchmark_suite.py`](07_benchmark_suite.py) | `BenchmarkRunner` across several datasets |

Run any example from the repository root with the workspace environment:

```bash
uv run python examples/01_quickstart.py
```
