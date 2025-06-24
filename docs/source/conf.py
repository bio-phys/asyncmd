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
import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

# read the version from the pyproject.toml
def _get_version_from_pyproject():
    """Get version string from pyproject.toml file."""
    pyproject_toml = os.path.join(os.path.dirname(__file__),
                                  "../../pyproject.toml")
    with open(pyproject_toml) as f:
        line = f.readline()
        while line:
            if line.startswith("version ="):
                version_line = line
                break
            line = f.readline()
    version = version_line.strip().split(" = ")[1]
    version = version.replace('"', '').replace("'", "")
    return version

project = 'asyncmd'
copyright = '2022-now, Hendrik Jung'
author = 'Hendrik Jung'

# The full version, including alpha/beta/rc tags
version = release = _get_version_from_pyproject()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
        ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for autosummary extension----------------------------------------
autosummary_imported_members = False  # default = False
autosummary_ignore_module_all = True  # default = True

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
                    # document members (default = False)
                    "members": False,
                           }
