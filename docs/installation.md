# Installation

`evo-gafs` requires Python ≥ 3.9.

```bash
pip install evo-gafs
```

Optional plotting helpers (matplotlib) are available via the `viz` extra:

```bash
pip install "evo-gafs[viz]"
```

## Dependencies

The core package depends only on widely used scientific Python libraries:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/) (≥ 1.6)
- [DEAP](https://github.com/DEAP/deap)

## Development install

The project is a [uv](https://docs.astral.sh/uv/) workspace. From a checkout:

```bash
uv sync                 # core + dev tooling
uv sync --group docs    # also install the documentation toolchain
```
