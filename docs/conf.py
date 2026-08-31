import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "QSPR Modeling of Ionic Liquid properties"
author = "Shamkhal Baybekov, Kamran Heydarov"
copyright = ""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]
autodoc_member_order = "bysource"
autodoc_mock_imports = ["mordred", "rdkit", "xgboost", "streamlit"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
