## Copyright 2016-2017 Tom Brown (FIAS)

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

"""Functionality to help with georeferencing and calculate
distances/areas.

"""


# make the code as Python 3 compatible as possible
from __future__ import division, absolute_import


__author__ = "Tom Brown (FIAS)"
__copyright__ = "Copyright 2016-2017 Tom Brown (FIAS), GNU GPL 3"

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def haversine(a0,a1):
    """
    Compute the distance in km between two sets of points in long/lat.

    One dimension of a* should be 2; longitude then latitude. Uses haversine
    formula.

    Parameters
    ----------
    a0: array/list of at most 2 dimensions
        One dimension must be 2
    a1: array/list of at most 2 dimensions
        One dimension must be 2

    Returns
    -------
    distance_km:  np.array
        2-dimensional array of distances in km between points in a0, a1


    Examples
    --------

    haversine([10.1,52.6],[[10.8,52.1],[-34,56.]])

    returns array([[   73.15416698,  2836.6707696 ]])

    """

    a = [np.asarray(a0,dtype=float),np.asarray(a1,dtype=float)]

    for i in range(2):
        if len(a[i].shape) == 1:
            a[i] = np.reshape(a[i],(1,a[i].shape[0]))

        if a[i].shape[1] != 2:
            logger.error("Inputs to haversine have the wrong shape!")
            return

        a[i] = np.deg2rad(a[i])

    distance_km = np.zeros((a[0].shape[0],a[1].shape[0]))

    for i in range(a[1].shape[0]):
        b = np.sin((a[0][:,1] - a[1][i,1])/2.)**2 + np.cos(a[0][:,1]) * np.cos(a[1][i,1]) * np.sin((a[0][:,0] - a[1][i,0])/2.)**2
        distance_km[:,i] = 6371.000 * 2 * np.arctan2( np.sqrt(b), np.sqrt(1-b) )

    return distance_km




#This function follows http://toblerity.org/shapely/manual.html

def area_from_lon_lat_poly(geometry):
    """
    Compute the area in km^2 of a shapely geometry, whose points are in longitude and latitude.

    Parameters
    ----------
    geometry: shapely geometry
        Points must be in longitude and latitude.

    Returns
    -------
    area:  float
        Area in km^2.

    """

    import pyproj
    from shapely.ops import transform
    from functools import partial


    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'), # Source: Lon-Lat
        pyproj.Proj(proj='aea')) # Target: Albers Equal Area Conical https://en.wikipedia.org/wiki/Albers_projection

    new_geometry = transform(project, geometry)

    #default area is in m^2
    return new_geometry.area/1e6
