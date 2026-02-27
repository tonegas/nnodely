# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
import os
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), '..', 'nnodely', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = __package__
author = 'tonegas'
release = read_version()
version = read_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
#    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
]

templates_path = []
exclude_patterns = []
# exclude_patterns = ['Thumbs.db', '.DS_Store', 'docs', 'examples', 'tests', 'mplplots']
# autodoc_default_options = {
#     'exclude-members': 'nnodely.activation'
# }

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = '_static/logo.png'

# -- Options for EPUB output -------------------------------------------------
epub_copyright = '2024, tonegas'  # Add this line
