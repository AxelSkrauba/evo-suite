# evo-suite

**Evolutionary computation for tabular data engineering.**

`evo-suite` is a monorepo hosting a family of independent, `scikit-learn`-compatible
packages that apply evolutionary computation to the data-preprocessing stage of a
machine-learning pipeline.

| Distribution | Import | Technique | Role | Status |
|--------------|--------|-----------|------|--------|
| [`evo-gafs`](packages/evo-gafs) | `evo_gafs` | Genetic Algorithm | **Feature selection** (choose the best subset of existing features) | 🚧 In development |
| [`evo-gpfe`](packages/evo-gpfe) | `evo_gpfe` | Genetic Programming | **Feature engineering** (build new symbolic features) | 📋 Planned |

Packages follow a consistent family pattern — distribution `evo-<module>`, import
`evo_<module>` — so future additions (`evo-hpo`, `evo-nas`, …) slot in naturally.
Each package is published to PyPI independently (`pip install evo-gafs`), has
minimal dependencies, and can be used on its own. They share this repository, its
CI, its documentation and its conventions.

## Why a monorepo?

The packages are **conceptually related but technically independent**: both apply
evolutionary computation to tabular data, yet neither imports the other and each
has its own release cadence. A monorepo with separately published packages gives
us the best of both worlds:

- **Single-repo maintenance** — one CI matrix, one issue tracker, one set of
  conventions and shared documentation.
- **Per-package release independence** — a fix in one package never forces a
  release of the other; each is versioned and published on its own schedule.
- **Minimal dependency footprints** — users install only what they need
  (`evo-gafs` does not pull in `evo-gpfe`'s extra dependencies, and vice versa).
- **Room to grow** — new evolutionary tools can join as additional packages
  without re-architecting anything.

This mirrors the structure used by established scientific Python projects that
ship multiple coordinated-but-separate packages from a single repository.

## Compatibility matrix

| `evo-gafs` | `evo-gpfe` | Status |
|------------|------------|--------|
| 0.1.x      | —          | ✓ Supported |

## Development

This repo uses [`uv`](https://docs.astral.sh/uv/) workspaces. A single shared
virtual environment covers every package.

```bash
# Create the environment and install all workspace packages + dev tooling
uv sync

# Run the test suite (a specific package)
uv run pytest packages/evo-gafs

# Lint, format-check and type-check
uv run ruff check .
uv run ruff format --check .
uv run mypy packages/evo-gafs/src
```

## Releasing

Packages are published to PyPI **independently** from this repo. Continuous
integration (`.github/workflows/ci.yml`) runs lint, type-checks, tests and a
packaging check on every push and pull request. Continuous delivery
(`.github/workflows/publish.yml`) builds and publishes a package via PyPI
Trusted Publishing when a package-scoped tag (e.g. `evo-gafs-v0.1.0`) is pushed.

## License

[MIT](LICENSE) © 2026 Axel Skrauba and the evo-suite contributors.
