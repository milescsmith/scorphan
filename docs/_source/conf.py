# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# sys.path.insert(0, os.path.abspath('.'))
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE.parent))

# -- Project information -----------------------------------------------------

project = "scorphan"
copyright = f"{datetime.now():%Y}, Miles Smith"
author = "Miles Smith"
repository_url = "https://github.com/milescsmith/scorphan.git"


# -- General configuration ---------------------------------------------------

nitpicky = True  # Warn about broken links. This is here for a reason: Do not change.
needs_sphinx = "9.1.0"  # Nicer param docs

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    # "autodoc2",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.jquery",
    "sphinx_datatables",
    "sphinx_autodoc_typehints",
]
autodoc2_packages = [
    {
        "path": "../../src/scorphan",
        "auto_mode": True,
    }
]
intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "decoupler": ("https://decoupler.readthedocs.io/en/latest/", None),
    "mudata": ("https://mudata.readthedocs.io/stable/", None),
    "muon": ("https://muon.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pydeseq2": ("https://pydeseq2.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}

main_doc = "index"
source_suffix = {".rst": "restructuredtext"}
default_role = "literal"
pygments_style = "sphinx"
nitpick_ignore = [("py:class", "type")]
autodoc_type_aliases = {
    "Iterable": "Iterable",
    "ArrayLike": "numpy.typing.ArrayLike",
}


autosummary_generate = True
autodoc_member_order = "bysource"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
nitpicky = True  # Report broken links

typehints_use_rtype = False
typehints_defaults = "braces"
always_use_bars_union = True

autosectionlabel_prefix_document = True
todo_include_todos = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store",  "**.ipynb_checkpoints",]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 2}
html_show_sphinx = False
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
