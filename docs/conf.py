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
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_nb",
]

# A reference page is the class docstring: the constructor holds no text of
# its own.
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_typehints = "none"

# Every public name has a page of its own, generated from the lists of the API
# reference with the templates of _templates/autosummary.
autosummary_generate = True
templates_path = ["_templates"]

# The docstrings are written in the numpydoc style. Next to its sections, a
# reference page has the syntax of every calling form, the streams a layer is
# called on, and its examples in groups, from the simplest to the edge cases.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# Types and defaults are text, "int, default 1", not cross-references.
napoleon_use_param = False
napoleon_use_ivar = True
napoleon_custom_sections = [
    "Syntax",
    ("Call arguments", "params_style"),
    "Basic Examples",
    "Scope",
    "Options",
    "Applications",
    "Properties & Relations",
    "Possible Issues",
]

# Every cross-reference has to resolve. The types of attributes are text, which
# Sphinx still makes into references: those in words ("dict of str to
# float"), of other libraries and the builtins are not checked.
nitpicky = True
nitpick_ignore_regex = [
    ("py:class", r".*\s.*"),
    ("py:class", r"(keras|numpy|np|pandas|pd|sympy|matplotlib|pathlib)\..*"),
    ("py:class", r"str|int|float|bool|dict|list|tuple|callable|None|y_true|y_pred"),
]

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
html_css_files = ["nnodely.css"]
html_logo = "_static/logo.png"

# -- Options for EPUB output -------------------------------------------------
epub_copyright = "2024-2026, Gastone Pietro Rosati Papini"


# Methods every layer has for Keras or for the graph, which a user never calls.
_INTERNAL_MEMBERS = {
    "build_layer",
    "call",
    "compute_output_shape",
    "from_config",
    "get_config",
    "input_signature",
    "output_shape",
    # the hooks Keras calls on a training printer
    "on_epoch_begin",
    "on_epoch_end",
    "on_train_batch_begin",
    "on_train_batch_end",
    "on_train_begin",
    "on_train_end",
    "set_model",
    "set_params",
}


def _skip_internal_member(app, what, name, obj, skip, options):
    return True if name in _INTERNAL_MEMBERS else None


def setup(app):
    app.connect("autodoc-skip-member", _skip_internal_member)
