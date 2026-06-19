# gpfe — Genetic Programming Feature Engineer

> 📋 **Planned.** This package is reserved within the `evo-suite` monorepo and
> is not yet implemented.

`gpfe` will use **Genetic Programming** to *construct* new symbolic features
(transformations of the existing ones) for tabular data, complementing
[`gafs`](../gafs) which *selects* among existing features.

Like `gafs`, it will expose a scikit-learn-compatible transformer
(`fit`/`transform`) and integrate into the shared documentation and CI of the
[`evo-suite`](../../README.md) project. It will be published independently to
PyPI (`pip install gpfe`) and is expected to add `scipy` as its only extra core
dependency.
