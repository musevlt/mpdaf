# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MPDAF'
copyright = '2010-2026, CRAL'
author = 'MPDAF Team'

release = importlib.metadata.version("mpdaf")
# for example take major/minor
version = '.'.join(release.split('.')[:2])

# -- General configuration ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    'pytest_doctestplus.sphinx.doctestplus',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'sphinx_copybutton',
    "sphinxcontrib.globalsubs",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'ndcube': ('https://docs.sunpy.org/projects/ndcube/en/stable/', None),
}

# autodoc_default_flags = ['members', 'special-members', 'inherited-members']
autodoc_member_order = 'bysource'

autosummary_generate = True

automodapi_toctreedirnm = 'api'
automodsumm_inherited_members = True

# Debug:
# automodapi_writereprocessed = True
# automodsumm_writereprocessed = True

numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {'type', 'optional', 'default', 'or', 'of', 'method'}
numpydoc_xref_aliases = {
    # python
    'sequence': ':term:`python:sequence`',
    'iterable': ':term:`python:iterable`',
    'string': 'str',
    # numpy
    'array': 'numpy.ndarray',
    'dtype': 'numpy.dtype',
    'ndarray': 'numpy.ndarray',
    'array-like': ':term:`numpy:array_like`',
    'array_like': ':term:`numpy:array_like`',
}
# numpydoc_use_plots = True

ipython_savefig_dir = '_static/_generated'
ipython_warning_is_error = False
ipython_execlines = """\
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpdaf import setup_logging
if os.path.relpath(os.curdir, start=os.pardir) != 'data': os.chdir('../src/mpdaf/data')
setup_logging(stream=sys.stdout)
plt.rcParams.update({
    "axes.labelsize": "large",
    "figure.figsize": (6, 6),
    "figure.subplot.hspace": 0.5,
    "savefig.bbox": "tight",
    "savefig.facecolor": "none",
})
""".splitlines()

# Class documentation should contain *both* the class docstring and
# the __init__ docstring
autoclass_content = "both"

gitlab_url = 'https://git-cral.univ-lyon1.fr/MUSE/mpdaf'
extlinks = {'issue': (gitlab_url + '/issues/%s', '#%s'),
            'pull': (gitlab_url + '/merge_requests/%s', '!%s')}

# Render inheritance diagrams in SVG
graphviz_output_format = "svg"

graphviz_dot_args = [
    '-Nfontsize=10',
    '-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Efontsize=10',
    '-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Gfontsize=10',
    '-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '_templates']

# The reST default role (used for this markup: `text`) to use for all
# documents. Set to the "smart" one.
default_role = 'obj'

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'sphinx'

# -- Options for HTML output --------------------------------------------------

html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "collapse_navigation": True,
    "github_url": "https://github.com/muse/mpdaf",
    "gitlab_url": "https://git-cral.univ-lyon1.fr/MUSE/mpdaf",
    "show_nav_level": 3,
    "show_prev_next": False,
    "show_toc_level": 2,
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "musevlt",
    "github_repo": "mpdaf",
    "github_version": "master",
    "doc_path": "docs",
}

html_sidebars = {
  "changelog": [],
  "contribute": [],
  "credits": [],
  "installation": [],
  "start": [],
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/logo/logo_mpdaf_small.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

global_substitutions: dict[str, str] = {
    # Table
    "Column": ":class:`~astropy.table.Column`",
    "MaskedColumn": ":class:`~astropy.table.MaskedColumn`",
    "TableColumns": ":class:`~astropy.table.TableColumns`",
    "Row": ":class:`~astropy.table.Row`",
    "Table": ":class:`~astropy.table.Table`",
    "QTable": ":class:`~astropy.table.QTable`",
}
