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


__author__ = "Tom Brown (FIAS)"
__copyright__ = "Copyright 2016-2017 Tom Brown (FIAS), GNU GPL 3"

import numpy as np

import logging
logger = logging.getLogger(__name__)

def haversine_pts(a, b):
    """
    Determines crow-flies distance between points in a and b

    ie. distance[i] = crow-fly-distance between a[i] and b[i]

    Parameters
    ----------
    a, b : N x 2 - array of dtype float
        Geographical coordinates in longitude, latitude ordering

    Returns
    -------
    c : N - array of dtype float
        Distance in km

    See also
    --------
    haversine : Matrix of distances between all pairs in a and b
    """

    lon0, lat0 = np.deg2rad(np.asarray(a, dtype=float)).T
    lon1, lat1 = np.deg2rad(np.asarray(b, dtype=float)).T

    c = (np.sin((lat1-lat0)/2.)**2 + np.cos(lat0) * np.cos(lat1) *
         np.sin((lon0 - lon1)/2.)**2)
    return 6371.000 * 2 * np.arctan2( np.sqrt(c), np.sqrt(1-c) )

def haversine(a, b):
    """
    Compute the distance in km between two sets of points in long/lat.

    One dimension of a* should be 2; longitude then latitude. Uses haversine
    formula.

    Parameters
    ----------
    a : array/list of at most 2 dimensions
        One dimension must be 2
    b : array/list of at most 2 dimensions
        One dimension must be 2

    Returns
    -------
    distance_km : array
        2-dimensional array of distances in km between points in a, b

    Examples
    --------
    >>> haversine([10.1,52.6], [[10.8,52.1], [-34,56.]])
    array([[   73.15416698,  2836.6707696 ]])

    See also
    --------
    haversine_pts : Determine pointwise crow-fly distance
    """

    def ensure_dimensions(a):
        a = np.asarray(a)

        if a.ndim == 1:
            a = a[np.newaxis, :]

        assert a.shape[1] == 2, "Inputs to haversine have the wrong shape!"

        return a

    a = ensure_dimensions(a)
    b = ensure_dimensions(b)

    return haversine_pts(a[np.newaxis,:], b[:,np.newaxis])


#This function follows http://toblerity.org/shapely/manual.html

def area_from_lon_lat_poly(geometry):
    """
    Compute the area in km^2 of a shapely geometry, whose points are in
    longitude and latitude.

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
