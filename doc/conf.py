# -*- coding: utf-8 -*-
#
# mpdaf documentation build configuration file, created by
# sphinx-quickstart on Fri Jun 22 10:03:09 2012.

from __future__ import print_function

import datetime
import os
import sys
import warnings

try:
    import astropy_helpers
except ImportError:
    # Building from inside the doc/ directory?
    if os.path.basename(os.getcwd()) == 'doc':
        a_h_path = os.path.abspath(os.path.join('..', 'astropy_helpers'))
        if os.path.isdir(a_h_path):
            sys.path.insert(1, a_h_path)

# Load all of the global Astropy configuration
from astropy_helpers.sphinx.conf import *

# Get configuration information from setup.cfg
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser
conf = ConfigParser()

conf.read([os.path.join(os.path.dirname(__file__), '..', 'setup.cfg')])
setup_cfg = dict(conf.items('metadata'))

sys.path.insert(0, os.path.abspath('./ext'))
# sys.setrecursionlimit(1500)

# -- General configuration ----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions += [
    'sphinx.ext.ifconfig',
    'IPython.sphinxext.ipython_console_highlighting',
    'ipython_directive',
    # 'sphinx_automodapi.automodapi',
    # 'sphinx_automodapi.smart_resolver'
]
extensions.remove('astropy_helpers.sphinx.ext.changelog_links')

autodoc_member_order = 'bysource'

automodsumm_inherited_members = True

# Debug:
# automodapi_writereprocessed = True
# automodsumm_writereprocessed = True

numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
# numpydoc_use_plots = True

ipython_savefig_dir = '_static/_generated'
ipython_execlines = """\
from __future__ import division, print_function
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpdaf import setup_logging
if os.path.relpath(os.curdir, start=os.pardir) != 'data': os.chdir('../lib/mpdaf/data')
setup_logging(stream=sys.stdout)
""".splitlines()

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '_templates']

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog = """
"""

# The reST default role (used for this markup: `text`) to use for all
# documents. Set to the "smart" one.
default_role = 'obj'

# General information about the project.
project = setup_cfg['package_name']
author = setup_cfg['author']
copyright = '{0}, {1}'.format(
    datetime.datetime.now().year, setup_cfg['author'])

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

__import__(setup_cfg['package_name'])
package = sys.modules[setup_cfg['package_name']]

# The short X.Y version.
version = package.__version__.split('-', 1)[0]
# The full version, including alpha/beta/rc tags.
release = package.__version__

# -- Options for HTML output --------------------------------------------------

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/logo/logo_mpdaf_small.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%d %b %Y'

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
html_use_opensearch = 'http://mpdaf.readthedocs.io/en/latest/'

# Output file base name for HTML help builder.
htmlhelp_basename = 'mpdafdoc'


# -- Options for LaTeX output -------------------------------------------------

# The paper size ('letter' or 'a4').
# latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [('index', project + '.tex', project + u' Documentation',
                    author, 'manual')]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Additional stuff for the LaTeX preamble.
# latex_preamble = ''

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True
