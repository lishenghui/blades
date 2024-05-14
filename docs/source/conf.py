import os.path as osp
import sys
import blades_sphinx_theme

project = "blades"
copyright = "2023, Shenghui Li"
author = "Shenghui Li"
release = "0.2"

sys.path.append(osp.join(osp.dirname(blades_sphinx_theme.__file__), "extension"))

example_path = osp.abspath("../../examples")

print(example_path)
sys.path.insert(0, example_path)  # 添加 example 目录到路径
sys.path.insert(0, osp.abspath("../../blades/aggregators"))  # 添加 example 目录到路径

# sys.path.insert(0, os.path.abspath("/Users/sheli564/Desktop/blades/src/blades"))
# sys.path.insert(0, os.path.abspath("/Users/sheli564/Desktop/blades/"))


extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx_gallery.load_style",
    "blades_sphinx",
    "m2r2",
]


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "blades_sphinx_theme"
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

sphinx_gallery_conf = {
    "line_numbers": False,
    "ignore_pattern": "/todo_",
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "_examples",  # path to where to save gallery generated output
}

# def setup(app):
#     def rst_jinja_render(app, _, source):
#         rst_context = {'torch_geometric': torch_geometric}
#         source[0] = app.builder.templates.render_string(source[0], rst_context)
#
#     app.connect('source-read', rst_jinja_render)
#     app.add_js_file('js/version_alert.js')
