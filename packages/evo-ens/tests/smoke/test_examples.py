"""Smoke test: every evo-ens example runs end-to-end without error.

The examples live at the repository root (`examples/evo-ens/`), outside the
package. They are only present in a source checkout (not in the published
distribution), so the test is skipped when the directory is absent.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "evo-ens"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py")) if EXAMPLES_DIR.is_dir() else []

if not EXAMPLE_SCRIPTS:
    pytest.skip("examples directory not available", allow_module_level=True)


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.stem)
def test_example_runs(script: Path) -> None:
    # run_name="__main__" so each script's `if __name__ == "__main__"` block runs.
    runpy.run_path(str(script), run_name="__main__")
