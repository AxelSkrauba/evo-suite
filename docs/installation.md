# Installation

Both packages require Python ≥ 3.9 and are installed independently.

```bash
pip install evo-gafs     # Genetic Algorithm Feature Selector
pip install evo-gpfe     # Genetic Programming Feature Engineer
```

Optional plotting helpers (matplotlib) are available via the `viz` extra on
either package:

```bash
pip install "evo-gafs[viz]"
pip install "evo-gpfe[viz]"
```

## Dependencies

Both packages depend only on widely used scientific Python libraries:

| Package | Core dependencies |
|---------|--------------------|
| `evo-gafs` | [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/) (≥ 1.6), [DEAP](https://github.com/DEAP/deap) |
| `evo-gpfe` | same as `evo-gafs`, plus [scipy](https://scipy.org/) (used for correlation-based fitness metrics) |

Neither package depends on the other; combining them (see the
`06_gp_then_ga_pipeline.py` example) is opt-in.

## Development install

The project is a [uv](https://docs.astral.sh/uv/) workspace. From a checkout:

```bash
uv sync                 # both packages + dev tooling
uv sync --group docs    # also install the documentation toolchain
```
