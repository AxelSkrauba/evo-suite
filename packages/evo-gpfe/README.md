# evo-gpfe — Genetic Programming Feature Engineer

> 📋 **Planned.** This package is reserved within the `evo-suite` monorepo and
> is not yet implemented.

`evo-gpfe` will use **Genetic Programming** to *construct* new symbolic features
(transformations of the existing ones) for tabular data, complementing
[`evo-gafs`](../evo-gafs) which *selects* among existing features.

Like `evo-gafs`, it will expose a scikit-learn-compatible transformer
(`fit`/`transform`, import name `evo_gpfe`) and integrate into the shared
documentation and CI of the [`evo-suite`](../../README.md) project. It will be
published independently to PyPI (`pip install evo-gpfe`) and is expected to add
`scipy` as its only extra core dependency.
