# -*- coding: utf-8 -*-
"""
Python for Power Systems Analysis (PyPSA)

Energy system modelling library.
"""

import sys

from pypsa import (
    clustering,
    components,
    contingency,
    descriptors,
    examples,
    geo,
    io,
    linopf,
    linopt,
    optimization,
    pf,
    plot,
    statistics,
)
from pypsa.components import Network, SubNetwork

if sys.version_info < (3, 12):
    from pypsa import opf, opt

__version__ = "0.28.0"

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2024 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)
