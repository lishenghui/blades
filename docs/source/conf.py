import os.path as osp
import sys
import pyg_sphinx_theme


project = "blades"
copyright = "2023, Shenghui Li"
author = "Shenghui Li"
release = "0.2"

sys.path.append(osp.join(osp.dirname(pyg_sphinx_theme.__file__), "extension"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "pyg",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "pydata_sphinx_theme"
html_theme = "pyg_sphinx_theme"
html_static_path = ["_static"]

html_logo = "_static/blades_logo.png"
# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"


intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "torch": ("https://pytorch.org/docs/master", None),
}

# def setup(app):
#     def rst_jinja_render(app, _, source):
#         rst_context = {'torch_geometric': torch_geometric}
#         source[0] = app.builder.templates.render_string(source[0], rst_context)
#
#     app.connect('source-read', rst_jinja_render)
#     app.add_js_file('js/version_alert.js')
