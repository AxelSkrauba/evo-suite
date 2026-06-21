# Contributing to evo-suite

Thanks for your interest in contributing! This document covers the shared
conventions for every package in the monorepo.

## Development environment

We use [`uv`](https://docs.astral.sh/uv/) to manage a single shared virtual
environment for the whole workspace.

```bash
# One-time: install uv (https://docs.astral.sh/uv/getting-started/installation/)

# Create/sync the environment with every workspace package and dev tooling
uv sync

# Activate (optional — `uv run` works without activation)
#   Windows (PowerShell): .venv\Scripts\Activate.ps1
#   Linux/macOS:          source .venv/bin/activate
```

## Workflow

1. Create a feature branch.
2. Make your change with tests.
3. Run the full quality gate locally (see below).
4. Open a pull request.

## Quality gate

Every change must pass, from the repo root:

```bash
uv run ruff check .                  # lint
uv run ruff format --check .         # formatting
uv run mypy packages/evo-gafs/src    # type checking
uv run pytest packages/evo-gafs      # tests (target: >= 80% coverage)
```

## Conventions

- **Layout:** every package uses the `src/` layout
  (`packages/<dist-name>/src/<import_name>/`, e.g. `packages/evo-gafs/src/evo_gafs/`).
- **Naming:** the package family follows `evo-<module>` (distribution) and
  `evo_<module>` (import).
- **Docstrings:** NumPy style (compatible with `numpydoc` / Sphinx).
- **Public API:** exported explicitly via each package's top-level `__init__.py`.
- **Versioning:** semantic versioning, per-package.
- **scikit-learn compatibility:** estimators inherit from `BaseEstimator` and the
  appropriate mixin, follow the fit/transform contract, store fitted attributes
  with a trailing underscore, and never mutate constructor parameters in `fit`.

## Commit messages

Focus on *why*, not *what*. Keep the subject concise.
