"""Functionality to help with georeferencing and calculate distances/areas."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from pypsa.constants import DEFAULT_EPSG

if TYPE_CHECKING:
    import cartopy.crs as ccrs
    from cartopy.mpl.geoaxes import GeoAxes
    from numpy.typing import ArrayLike


logger = logging.getLogger(__name__)


def haversine_pts(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Determine crow-flies distance between points in a and b.

    ie. distance[i] = crow-fly-distance between a[i] and b[i]

    Parameters
    ----------
    a, b : N x 2 - array of dtype float
        Geographical coordinates in longitude, latitude ordering

    Returns
    -------
    c : N - array of dtype float
        Distance in km


    Examples
    --------
    >>> a = np.array([[10.1, 52.6], [10.8, 52.1]])
    >>> b = np.array([[10.8, 52.1], [-34, 56.]])
    >>> haversine_pts(a, b)
    array([  73.15416698, 2903.73511621])

    """
    lon0, lat0 = np.deg2rad(np.asarray(a, dtype=float)).T
    lon1, lat1 = np.deg2rad(np.asarray(b, dtype=float)).T

    c = (
        np.sin((lat1 - lat0) / 2.0) ** 2
        + np.cos(lat0) * np.cos(lat1) * np.sin((lon0 - lon1) / 2.0) ** 2
    )
    return 6371.000 * 2 * np.arctan2(np.sqrt(c), np.sqrt(1 - c))


def haversine(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Compute the distance in km between two sets of points in long/lat.

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
    >>> haversine([10.1, 52.6], [[10.8, 52.1], [-34, 56.]])
    array([[  73.15416698, 2836.6707696 ]])

    """

    def ensure_dimensions(a: np.ndarray | ArrayLike) -> np.ndarray:
        """Ensure correct shape for haversine calculation.

        Parameters
        ----------
        a : array-like
            Array to check

        Returns
        -------
        array:
            N x 2 array

        """
        a = np.asarray(a)

        if a.ndim == 1:
            a = a[np.newaxis, :]

        if a.shape[1] != 2:
            msg = "Array must have shape (N, 2)"
            raise ValueError(msg)

        return a

    a = ensure_dimensions(a)
    b = ensure_dimensions(b)

    return haversine_pts(a[np.newaxis, :], b[:, np.newaxis])


def compute_bbox(
    x: ArrayLike, y: ArrayLike, margin: float = 0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute bounding box for given x, y coordinates.

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

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> compute_bbox(x, y)  # doctest: +ELLIPSIS
    ((np.int64(1), np.int64(4)), (np.int64(3), np.int64(6)))

    >>> # With margin to expand the bounding box
    >>> compute_bbox(x, y, margin=0.1)  # doctest: +ELLIPSIS
    ((..., ...), (..., ...))

    """
    # set margins
    pos = np.asarray((x, y))
    minxy, maxxy = np.nanmin(pos, axis=1), np.nanmax(pos, axis=1)
    xy1 = minxy - margin * (maxxy - minxy)
    xy2 = maxxy + margin * (maxxy - minxy)
    return tuple(xy1), tuple(xy2)


def get_projection_from_crs(crs: int | str) -> ccrs.Projection:
    """Get cartopy projection from EPSG code or proj4 string.

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

    Examples
    --------
    >>> get_projection_from_crs(4326)
    <Projected CRS: +proj=eqc +ellps=WGS84 +a=6378137.0 +lon_0=0.0 +to ...>
    Name: unknown
    Axis Info [cartesian]:
    - E[east]: Easting (unknown)
    - N[north]: Northing (unknown)
    - h[up]: Ellipsoidal height (metre)
    Area of Use:
    - undefined
    Coordinate Operation:
    - name: unknown
    - method: Equidistant Cylindrical
    Datum: Unknown based on WGS 84 ellipsoid
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich
    <BLANKLINE>

    """
    import cartopy.crs as ccrs

    try:
        return ccrs.epsg(crs)
    except ValueError:
        pass

    if crs != 4326 and not str(crs).endswith("4326"):
        logger.warning(
            "Could not find projection for '%s'. Falling back to latlong.", crs
        )

    return ccrs.PlateCarree()


def get_projected_area_factor(
    ax: GeoAxes, original_crs: int | str = DEFAULT_EPSG
) -> float:
    """Get scale of current vs original projection in terms of area.

    The default 'original crs' is assumed to be 4326, which translates
    to the cartopy default cartopy.crs.PlateCarree()

    Examples
    --------
    >>> import cartopy.crs as ccrs
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Mercator()})
    >>> ax.set_extent([-10, 10, 40, 60], crs=ccrs.PlateCarree())
    >>> area_factor = get_projected_area_factor(ax)
    >>> area_factor
    np.float64(140056.26937534288)

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
