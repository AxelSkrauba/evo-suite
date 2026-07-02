# Examples

Runnable, self-contained examples for the `evo-suite` packages, organised by
package. Each script uses a small configuration so it finishes quickly, and is
executed in CI to stay valid.

## evo-gafs

| Example | Highlights |
|---------|------------|
| [`01_quickstart.py`](evo-gafs/01_quickstart.py) | Minimal `fit` / `transform` / `get_support` |
| [`02_alpha_tradeoff.py`](evo-gafs/02_alpha_tradeoff.py) | The `alpha` accuracy↔compression knob (edge deployment) |
| [`03_multiobjective_pareto.py`](evo-gafs/03_multiobjective_pareto.py) | NSGA-II Pareto front (+ optional plot) |
| [`04_pipeline_and_search.py`](evo-gafs/04_pipeline_and_search.py) | `Pipeline` + `GridSearchCV` interoperability |
| [`05_compare_baselines.py`](evo-gafs/05_compare_baselines.py) | Wrapper vs all-features vs univariate filter |
| [`06_regression.py`](evo-gafs/06_regression.py) | Regression task with auto-detected task type |
| [`07_benchmark_suite.py`](evo-gafs/07_benchmark_suite.py) | `BenchmarkRunner` across several datasets |

## evo-gpfe

| Example | Highlights |
|---------|------------|
| [`01_quickstart.py`](evo-gpfe/01_quickstart.py) | Minimal `fit` / `transform`, inspect discovered expressions |
| [`02_fitness_metrics.py`](evo-gpfe/02_fitness_metrics.py) | `mutual_info` vs `correlation` vs `model_score`: relevance vs cost |
| [`03_redundancy_and_parsimony.py`](evo-gpfe/03_redundancy_and_parsimony.py) | `redundancy_beta` and `parsimony_coeff` in action |
| [`04_pipeline_and_search.py`](evo-gpfe/04_pipeline_and_search.py) | `Pipeline` + `GridSearchCV` interoperability |
| [`05_nonlinear_showcase.py`](evo-gpfe/05_nonlinear_showcase.py) | Genuinely non-linear target (`make_friedman1`) — where GP shines |
| [`06_gp_then_ga_pipeline.py`](evo-gpfe/06_gp_then_ga_pipeline.py) | The evo-suite combo: GP constructs, GA selects |
| [`07_benchmark_suite.py`](evo-gpfe/07_benchmark_suite.py) | `GPBenchmarkRunner` across several datasets |

Run any example from the repository root with the workspace environment:

```bash
uv run python examples/evo-gafs/01_quickstart.py
uv run python examples/evo-gpfe/01_quickstart.py
```
