"""The single source of truth for the `keras_proba` version.

`pyproject.toml` reads the package version from here (`[tool.setuptools.dynamic]`),
and `keras_proba.__version__` re-exports it. Do not write the version anywhere else.
"""

__version__ = "0.1.0"
