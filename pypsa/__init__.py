

## Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), David
## Schlachtberger (FIAS)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""Python for Power Systems Analysis (PyPSA)

Grid calculation library.
"""


from __future__ import absolute_import

from . import components
from . import pf, opf, plot, networkclustering, io, contingency, geo

from .components import Network, SubNetwork

import logging
logging.basicConfig(level=logging.INFO)

__version__ = "0.12.0"
__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS)"
__copyright__ = "Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS), GNU GPL 3"
