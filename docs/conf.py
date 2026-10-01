# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
from importlib.metadata import version as package_version

# autodoc imports nnodely, which imports Keras: pick a backend before that
# happens. The documentation build installs the jax extra.
os.environ.setdefault("KERAS_BACKEND", "jax")

# -- Project information -----------------------------------------------------

project = "nnodely"
author = "tonegas"
copyright = "2024, tonegas"
release = package_version("nnodely")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "none"

exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
}

html_static_path = ["_static"]
html_logo = "_static/logo.png"

# -- Options for EPUB output -------------------------------------------------
epub_copyright = "2024, tonegas"
