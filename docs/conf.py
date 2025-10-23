import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "jaxion"
copyright = "2025, Philip Mocz"
author = "Philip Mocz"
language = "en"
version = "main"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath(".."))

html_title = "jaxion"
html_logo = "_static/jaxion-logo.svg"
html_favicon = "_static/jaxion-favicon.png"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxemoji.sphinxemoji",
]

# sphinx_book_theme options
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 4,
    "repository_url": "https://github.com/JaxionProject/jaxion",
    "path_to_docs": "docs/",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/JaxionProject/jaxion",
            "icon": "fa-brands fa-github",
        },
    ],
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
