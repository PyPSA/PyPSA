"""
Python for Power Systems Analysis (PyPSA)

Energy system modelling library.
"""

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2024 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import re
from importlib.metadata import version

from pypsa import (
    clustering,
    components,
    contingency,
    descriptors,
    examples,
    geo,
    io,
    optimization,
    pf,
    plot,
    statistics,
)
from pypsa.components import Network, SubNetwork

# e.g. "0.17.1" or "0.17.1.dev4+ga3890dc0" (if installed from git)
__version__ = version("pypsa")
# e.g. "0.17.0" # TODO, in the network structure it should use the dev version
match = re.match(r"(\d+\.\d+(\.\d+)?)", __version__)
assert match, f"Could not determine release_version of pypsa: {__version__}"
release_version = match.group(0)
assert not __version__.startswith("0.0"), "Could not determine version of pypsa."

__all__ = [
    "clustering",
    "components",
    "contingency",
    "descriptors",
    "examples",
    "geo",
    "io",
    "optimization",
    "pf",
    "plot",
    "statistics",
    "Network",
    "SubNetwork",
]
