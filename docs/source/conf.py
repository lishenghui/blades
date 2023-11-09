# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "blades"
copyright = "2023, Shenghui Li"
author = "Shenghui Li"
release = "0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

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


# html_theme_options = {
#     "navigation_depth": 5,
#     "collapse_navigation": False,
#     "icon_links": [
#         {
#             "name": "GitHub",
#             "url": "https://github.com/bladesteam/blades",
#             "icon": "fab fa-github-square",
#         },
#     ],
# }
