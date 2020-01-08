
## Copyright 2015-2019 Tom Brown (FIAS), Jonas Hoersch (FIAS), 2019-2020 Fabian Neumann (KIT)

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

"""General utility functions.
"""

# make the code as Python 3 compatible as possible
from __future__ import division
from __future__ import absolute_import
from six import iteritems, string_types
from six.moves import map, range, reduce


__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS), Fabian Neumann (KIT)"
__copyright__ = "Copyright 2015-2019 Tom Brown (FIAS), Jonas Hoersch (FIAS); Copyright 2019-2020 Fabian Neumann (KIT); GNU GPL 3"

import numpy as np


def _normed(s):
    tot = s.sum()
    if tot == 0:
        return 1.
    else:
        return s/tot


def _flatten_multiindex(m, join=' '):
    if m.nlevels <= 1: return m
    levels = map(m.get_level_values, range(m.nlevels))
    return reduce(lambda x, y: x+join+y, levels, next(levels))


def _make_consense(component, attr):
    def consense(x):
        v = x.iat[0]
        assert ((x == v).all() or x.isnull().all()), (
            "In {} cluster {} the values of attribute {} do not agree:\n{}"
            .format(component, x.name, attr, x)
        )
        return v
    return consense


def _haversine(coords):
    lon, lat = np.deg2rad(np.asarray(coords)).T
    a = np.sin((lat[1]-lat[0])/2.)**2 + np.cos(lat[0]) * np.cos(lat[1]) * np.sin((lon[0] - lon[1])/2.)**2
    return 6371.000 * 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )

def branch_select_b(lines, cname, sel):
    """
    Returns boolean pd.Series of lines based on the choice of the parameter `sel`.

    Parameters
    ----------
    sel : string
        Specifies selection of passive branches.
        If `None` (default) it includes both operative and inoperative lines.
        If `"operative"` it includes only operative lines.
        If `"inoperative"` it includes only inoperative lines.
        If `"potential"` it includes operative or candidate lines (i.e. not operative but extendable).
        If `"candidate"` it includes candidate lines; i.e. not operative but extendable lines.
        If `"used"` it includes operative and built candidate lines; can only be called after successful optimisation.

    Returns
    -------
    pd.Series
    """
    
    if cname != 'Line':
        return slice(None)

    if sel == 'operative':	
        return lines.operative == True	
    elif sel == 'inoperative':	
        return lines.operative == False	
    elif sel == 'potential':	
        return (lines.operative == True) | (lines.s_nom_extendable == True)	
    elif sel == 'candidate':	
        return (lines.operative == False) & (lines.s_nom_extendable == True)	
    elif sel == 'used':	
        return (lines.s_nom_opt > 0.0)
    else:
        return slice(None)


def branch_select_i(c, sel=None):
    """
    Returns intersection of `c.ind` and indices of a selection of lines
    for passive branch components based on the choice of the parameter `sel`.
    No effect for links.

    Parameters
    ----------
    sel : string
        Specifies selection of passive branches.
        If `None` (default) it includes both operative and inoperative lines.
        If `"operative"` it includes only operative lines.
        If `"inoperative"` it includes only inoperative lines.
        If `"potential"` it includes operative or candidate lines (i.e. not operative but extendable).
        If `"candidate"` it includes candidate lines; i.e. not operative but extendable lines.
        If `"used"` it includes operative and built candidate lines; can only be called after successful optimisation.

    Returns
    -------
    pandas.Index
    """

    if c.name == 'Line':
        selection_b = branch_select_b(c.df, c.name, sel)
        selection_i = c.df[selection_b].index
        if c.ind is None:
            s = selection_i
        else:
            s = c.ind.intersection(selection_i)
    else:
        if c.ind is None:
            s = slice(None)
        else:
            s = c.ind

    return s