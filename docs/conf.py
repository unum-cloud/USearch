# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Unum · USearch"
copyright = "2023, Unum"
author = "Unum"
release = open("../VERSION", "r").read().strip()
with open("_static/custom.js", "r+") as js:
    content = js.read()
    js.seek(0)
    js.truncate()
    js.write(content.replace("$(VERSION)", release))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "breathe",
    "m2r2",
    "sphinx.ext.autodoc",
    "sphinx_js",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.jquery",
    "sphinxcontrib.googleanalytics",
    # Sadly, javasphinx is not maintained anymore
    # "javasphinx",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.md"]


googleanalytics_id = "341385789"
googleanalytics_enabled = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "../assets/unum.png"
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_baseurl = "/docs/usearch/"

breathe_projects = {"USearch": "../build/xml"}
breathe_default_project = "USearch"

# To switch to TypeScript, uncomment the following lines:
#
#   js_language = "typescript"
#   js_source_path = "../javascript/usearch.ts"
#   jsdoc_config_path = "../javascript/tsconfig-cjs.json"
js_source_path = "../javascript/dist/cjs/usearch.js"
