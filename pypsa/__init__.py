
## Copyright 2015-2021 PyPSA Developers

## You can find the list of PyPSA Developers at
## https://pypsa.readthedocs.io/en/latest/developers.html

## PyPSA is released under the open source MIT License, see
## https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt

"""Python for Power Systems Analysis (PyPSA)

Grid calculation library.
"""

from . import components, descriptors
from . import (pf, opf, opt, plot, networkclustering, io, contingency, geo,
               stats, linopf, linopt, examples)

from .components import Network, SubNetwork

__version__ = "0.18.1"

__author__ = "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
__copyright__ = ("Copyright 2015-2021 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
                 "MIT License")
