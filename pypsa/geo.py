"""
Functionality to help with georeferencing and calculate distances/areas.
"""

from __future__ import annotations

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2024 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)


import logging
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import numpy as np

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from numpy.typing import ArrayLike


logger = logging.getLogger(__name__)


def haversine_pts(a: np.ndarray | ArrayLike, b: np.ndarray | ArrayLike) -> np.ndarray:
    """
    Determine crow-flies distance between points in a and b.

    ie. distance[i] = crow-fly-distance between a[i] and b[i]

    Parameters
    ----------
    a, b : N x 2 - array of dtype float
        Geographical coordinates in longitude, latitude ordering

    Returns
    -------
    c : N - array of dtype float
        Distance in km

    See Also
    --------
    haversine : Matrix of distances between all pairs in a and b
    """
    lon0, lat0 = np.deg2rad(np.asarray(a, dtype=float)).T
    lon1, lat1 = np.deg2rad(np.asarray(b, dtype=float)).T

    c = (
        np.sin((lat1 - lat0) / 2.0) ** 2
        + np.cos(lat0) * np.cos(lat1) * np.sin((lon0 - lon1) / 2.0) ** 2
    )
    return 6371.000 * 2 * np.arctan2(np.sqrt(c), np.sqrt(1 - c))


def haversine(a: np.ndarray | ArrayLike, b: np.ndarray | ArrayLike) -> np.ndarray:
    """
    Compute the distance in km between two sets of points in long/lat.

    One dimension of a* should be 2; longitude then latitude. Uses haversine
    formula.

    Parameters
    ----------
    a : N x 2 - array of dtype float
        Coordinates of first point, dimensions (N, 2)
    b : array-like of at most 2 dimensions
        Coordinates of second point, dimensions (M, 2)

    Returns
    -------
    distance_km : array
        2-dimensional array of distances in km between points in a, b

    Examples
    --------
    >>> haversine([10.1, 52.6], [[10.8, 52.1], [-34, 56.], [12.1, 53.1]])
    array([[  73.15416698, 2836.6707696 ,  145.34871388]])

    See Also
    --------
    haversine_pts : Determine pointwise crow-fly distance
    """

    #
    def ensure_dimensions(arr: np.ndarray | ArrayLike) -> np.ndarray:
        """
        Ensure correct shape for haversine calculation.

        Parameters
        ----------
        arr :
            Array to check

        Returns
        -------
        array:
            N x 2 array

        """
        arr = np.asarray(arr)

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if arr.shape[1] != 2:
            msg = "Array must have shape (N, 2)"
            raise ValueError(msg)

        return arr

    a = ensure_dimensions(a)
    b = ensure_dimensions(b)

    return haversine_pts(a[np.newaxis, :], b[:, np.newaxis])


def compute_bbox(
    x: ArrayLike, y: ArrayLike, margin: float = 0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute bounding box for given x, y coordinates.

    Also adds a margin around the bounding box, if desired. Defaults to 0.

    Parameters
    ----------
    x, y : array-like
        Arrays of x and y coordinates
    margin : float, optional
        Margin around the bounding box, by default 0

    Returns
    -------
    tuple
        Tuple of two tuples, representing the lower left (x1, y1) and upper right
        (x2, y2) corners of the bounding box

    """
    # set margins
    pos = np.asarray((x, y))
    minxy, maxxy = pos.min(axis=1), pos.max(axis=1)
    xy1 = minxy - margin * (maxxy - minxy)
    xy2 = maxxy + margin * (maxxy - minxy)
    return tuple(xy1), tuple(xy2)


def get_projection_from_crs(crs: int | str) -> ccrs.Projection:
    """
    Get cartopy projection from EPSG code or proj4 string.

    If the projection is not found, a warning is issued and the default
    PlateCarree projection is returned.

    Parameters
    ----------
    crs : int | str
        EPSG code or proj4 string

    Returns
    -------
    projection : cartopy.crs.Projection
        Cartopy projection object

    """
    try:
        return ccrs.epsg(crs)
    except ValueError:
        pass

    if crs != 4326 and not crs.endswith("4326"):
        logger.warning(
            "Could not find projection for '%s'. Falling back to latlong.", crs
        )

    return ccrs.PlateCarree()


def get_projected_area_factor(ax: GeoAxes, original_crs: int | str = 4326) -> float:
    """
    Get scale of current vs original projection in terms of area.

    The default 'original crs' is assumed to be 4326, which translates
    to the cartopy default cartopy.crs.PlateCarree()
    """
    if not hasattr(ax, "projection"):
        return 1
    x1, x2, y1, y2 = ax.get_extent()
    pbounds = get_projection_from_crs(original_crs).transform_points(
        ax.projection, np.array([x1, x2]), np.array([y1, y2])
    )

    return np.sqrt(
        abs((x2 - x1) * (y2 - y1)) / abs((pbounds[0] - pbounds[1])[:2].prod())
    )
