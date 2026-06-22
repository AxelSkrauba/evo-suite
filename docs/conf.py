"""Sphinx configuration for the evo-suite documentation."""

from __future__ import annotations

import importlib.metadata

# ── Project information ──────────────────────────────────────────────────────
project = "evo-suite"
author = "Axel Skrauba"
copyright = "2026, Axel Skrauba and the evo-suite contributors"

try:
    release = importlib.metadata.version("evo-gafs")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    release = "0.0.0"
version = release

# ── General configuration ────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── Autodoc / autosummary ────────────────────────────────────────────────────
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# ── Napoleon (NumPy-style docstrings) ────────────────────────────────────────
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# ── Intersphinx ──────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# ── MyST (Markdown) ──────────────────────────────────────────────────────────
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# ── HTML output ──────────────────────────────────────────────────────────────
html_theme = "furo"
html_title = f"evo-suite {release}"
