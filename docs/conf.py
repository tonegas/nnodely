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
author = "Gastone Pietro Rosati Papini"
copyright = "2024-2026, Gastone Pietro Rosati Papini"
release = package_version("nnodely")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",
]

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "none"

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The tutorials are notebooks, shown with the outputs they were saved with:
# the documentation build does not run them.
nb_execution_mode = "off"
nb_merge_streams = True
# $...$ and $$...$$ are math in the notebooks, as Jupyter shows them.
myst_enable_extensions = ["dollarmath"]

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
epub_copyright = "2024-2026, Gastone Pietro Rosati Papini"
