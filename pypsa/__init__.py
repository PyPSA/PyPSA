

## Copyright 2015-2019 Tom Brown (KIT, FIAS), Jonas Hoersch (KIT,
## FIAS), David Schlachtberger (FIAS)

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

from . import components, descriptors
from . import (pf, opf, opt, plot, networkclustering, io, contingency, geo,
               stats, linopf, linopt)

from .components import Network, SubNetwork

__version__ = "0.17.0"
__author__ = "Tom Brown (KIT, FIAS), Jonas Hoersch (KIT, FIAS), Fabian Hofmann (FIAS), Fabian Neumann (KIT), David Schlachtberger (FIAS)"
__copyright__ = "Copyright 2015-2020 Tom Brown (KIT, FIAS), Jonas Hoersch (KIT, FIAS), Fabian Hofmann (FIAS), Fabian Neumann (KIT), David Schlachtberger (FIAS), GNU GPL 3"
