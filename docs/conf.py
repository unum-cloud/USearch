"""Builds the Sphinx reference pages and configures their theme.

File: docs/conf.py
Author: Ash Vardanian
Date: October 27, 2026
"""

from pathlib import Path

from breathe.renderer.sphinxrenderer import SphinxRenderer

project = "USearch"
copyright = "2023-2026, Unum"
author = "Unum"
release = Path("../VERSION").read_text().strip()
version = release

extensions = ["breathe", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinxcontrib.googleanalytics"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The compiled `usearch` package needs none of these, only its pure-Python helpers do.
autodoc_mock_imports = ["numpy", "numkong", "tqdm"]
# The getters document each attribute already, so the "Attributes:" sections must not declare them again.
napoleon_use_ivar = True
cpp_id_attributes = ["__device__", "__forceinline__", "__global__", "__noinline__"]
cpp_paren_attributes = ["__launch_bounds__"]
suppress_warnings = ["duplicate_declaration.cpp"]
"""Sphinx reads constrained overloads as duplicates, while Doxygen still checks real ones."""

googleanalytics_id = "341385789"
googleanalytics_enabled = True

html_title = f"USearch {release}"
html_logo = "../assets/unum.png"
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Poppins:wght@200;400;500&family=Red+Hat+Mono:wght@400;500&display=swap",
    "custom.css",
]
html_baseurl = "https://unum-cloud.github.io/USearch/"
pygments_dark_style = "github-dark"
html_theme_options = {
    "source_repository": "https://github.com/unum-cloud/USearch/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "font-stack": '"Poppins", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        "font-stack--headings": '"Poppins", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        "font-stack--monospace": '"Red Hat Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
        "color-foreground-primary": "#000000",
        "color-foreground-secondary": "#545a71",
        "color-foreground-muted": "#787d8e",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f9fafb",
        "color-background-border": "#f0f1f4",
        "color-inline-code-background": "#f9fafb",
        "color-brand-primary": "#000000",
        "color-brand-content": "#2962ff",
        "color-api-name": "#000000",
        "color-highlight-on-target": "#f0f1f4",
    },
    "dark_css_variables": {
        "color-foreground-primary": "#eceef2",
        "color-foreground-secondary": "#a9aebd",
        "color-foreground-muted": "#8a90a2",
        "color-background-primary": "#111318",
        "color-background-secondary": "#181a20",
        "color-background-border": "#363a46",
        "color-inline-code-background": "#181a20",
        "color-brand-primary": "#eceef2",
        "color-brand-content": "#6690ff",
        "color-api-name": "#eceef2",
        "color-highlight-on-target": "#23262e",
    },
}

breathe_projects = {"USearch": "../build/xml"}
breathe_default_project = "USearch"


def visit_typedef(renderer, node, visit_other_typedef=SphinxRenderer.visit_typedef):
    """Declares a function pointer typedef, whose Doxygen 1.18 type `void(*)` Breathe 5.1 cannot name."""
    type_ = "".join(text.astext() for text in renderer.render(node.type))
    if not type_.endswith("(*)"):
        return visit_other_typedef(renderer, node)
    name = renderer.join_nested_name([*renderer.get_qualification(), node.name])
    return renderer.handle_declaration(node, node.kind.value, f"{type_[:-1]}{name}){node.argsstring}")


SphinxRenderer.visit_typedef = visit_typedef
