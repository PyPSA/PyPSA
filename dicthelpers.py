

## Copyright 2015 Jonas Hoersch (FIAS), Tom Brown (FIAS)

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


# make the code as Python 3 compatible as possible
from __future__ import print_function, division


__version__ = "0.1"
__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"


import numpy as np


def attrfilter(d, indexed=False, **attr):
    """Takes dictionary or list of objects and returns list filtered by
    attributes, or if indexed=True, an enumerated list.
    """

    test = lambda x: all(getattr(x, k, None) == v for k, v in attr.iteritems())
    if isinstance(d, dict):
        d = d.itervalues()
    if indexed:
        vs = []
        indices = []
        for i, x in enumerate(d):
            if test(x):
                vs.append(x)
                indices.append(i)
        return vs, indices
    else:
        return filter(test, d)

def attrdata(d, k, s=None):
    """Takes dictionary and returns array of attribute k, sliced by s."""

    if s is None: s = slice(None)
    return np.asarray([getattr(v, k)[s] for v in d.itervalues()])
