# evo-gpfe — Genetic Programming Feature Engineer

> � **In development.** Part of the [`evo-suite`](../../README.md) family
> (import name: `evo_gpfe`).

A **scikit-learn-compatible** symbolic feature constructor for tabular data,
powered by [DEAP](https://github.com/DEAP/deap). `evo-gpfe` evolves expression
trees that *combine* the original features into new, informative ones —
complementing [`evo-gafs`](../evo-gafs), which *selects* among existing features.

It uses a **Hall-of-Fame sequential** strategy that rewards relevance to the
target (mutual information) while penalising redundancy with already-generated
features and tree complexity (parsimony), producing diverse, interpretable
expressions such as `log(x2) * (x5 + x1^2)`.

## Installation

```bash
pip install evo-gpfe            # core
pip install "evo-gpfe[viz]"     # + matplotlib for the plotting helpers
```

📖 **Documentation:** <https://evo-suite.readthedocs.io/>

## License

[MIT](../../LICENSE)
