"""MkDocs hooks: generate the example pages from the notebooks before each build."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_examples_doc import generate_all  # noqa: E402


def on_pre_build(config):
    """Regenerate docs/examples/ from examples/**/*.ipynb."""
    generate_all()
