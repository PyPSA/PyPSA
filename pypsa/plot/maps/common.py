"""Define common functions for plotting maps in PyPSA."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pypsa import Network


def as_branch_series(  # noqa
    ser: pd.Series | dict | list, arg: str, c_name: str, n: "Network"
) -> pd.Series:
    ser = pd.Series(ser, index=n.static(c_name).index)
    if ser.isnull().any():
        msg = f"{c_name}_{arg}s does not specify all "
        f"entries. Missing values for {c_name}: {list(ser[ser.isnull()].index)}"
        raise ValueError(msg)
    return ser


@overload
def apply_layouter(
    n: "Network",
    layouter: Callable[..., Any] | None = None,
    inplace: Literal[True] = True,
) -> None: ...


@overload
def apply_layouter(
    n: "Network",
    layouter: Callable[..., Any] | None = None,
    inplace: Literal[False] = False,
) -> tuple[pd.Series, pd.Series]: ...


def apply_layouter(
    n: "Network",
    layouter: Callable[..., Any] | None = None,
    inplace: Literal[True, False] = False,
) -> Any:
    """Automatically generate bus coordinates for the network graph.

    Layouting function from `networkx <https://networkx.github.io/>`_ is used to
    determine the coordinates of the buses in the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to generate coordinates for.
    layouter : networkx.drawing.layout function, default None
        Layouting function from `networkx <https://networkx.github.io/>`_. See
        `list <https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout>`_
        of available options. By default, coordinates are determined for a
        `planar layout <https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.planar_layout.html#networkx.drawing.layout.planar_layout>`_
        if the network graph is planar, otherwise for a
        `Kamada-Kawai layout <https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.kamada_kawai_layout.html#networkx.drawing.layout.kamada_kawai_layout>`_.
    inplace : bool, default False
        Assign generated coordinates to the network bus coordinates
        at ``n.buses[['x', 'y']]`` if True, otherwise return them.

    Returns
    -------
    coordinates : pd.DataFrame or None
        DataFrame with x and y coordinates for each bus. Only returned if
        inplace is False.

    Examples
    --------
    >>> import pypsa
    >>> n = pypsa.examples.ac_dc_meshed()
    >>> x, y = apply_layouter(n, layouter=nx.circular_layout)
    >>> x
    London        1.000000
    Norwich       0.766044
    Norwich DC    0.173648
    Manchester   -0.500000
    Bremen       -0.939693
    Bremen DC    -0.939693
    Frankfurt    -0.500000
    Norway        0.173648
    Norway DC     0.766044
    Name: x, dtype: float64
    >>> y
    London        1.986821e-08
    Norwich       6.427876e-01
    Norwich DC    9.848077e-01
    Manchester    8.660254e-01
    Bremen        3.420202e-01
    Bremen DC    -3.420201e-01
    Frankfurt    -8.660254e-01
    Norway       -9.848077e-01
    Norway DC    -6.427877e-01
    Name: y, dtype: float64

    """
    G = n.graph()

    if layouter is None:
        if nx.check_planarity(G)[0]:
            layouter = nx.planar_layout
        else:
            layouter = nx.kamada_kawai_layout

    coordinates = pd.DataFrame(layouter(G)).T.rename({0: "x", 1: "y"}, axis=1)

    if inplace:
        n.c.buses.static[["x", "y"]] = coordinates
        return None
    return coordinates.x, coordinates.y


def to_rgba255(
    color: str,
    alpha: float = 1.0,
) -> list[int]:
    """Convert a Matplotlib color name/hex to an RGBA list with 0-255 integer values.

    Parameters
    ----------
    color : str
        Matplotlib color name or hex string.

    alpha : float, default 1.0
        Alpha transparency value between 0 (transparent) and 1 (opaque).

    Returns
    -------
    list of int
        List of RGBA values as integers in the range 0-255.

    """
    rgb = [round(c * 255) for c in mcolors.to_rgb(color)]
    a = round(alpha * 255)
    return rgb + [a]


# Geometric functions
def rotate_polygon(
    poly: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """Rotate polygon around origin by angle in radians.

    Parameters
    ----------
    poly : np.ndarray
        Nx2 array of polygon vertices.
    angle_rad : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Rotated polygon as Nx2 array.

    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return poly @ R.T


def flip_polygon(
    poly: np.ndarray,
    axis: str = "x",
) -> np.ndarray:
    """Flip polygon around specified axis ('x' or 'y').

    Parameters
    ----------
    poly : np.ndarray
        Nx2 array of polygon vertices.
    axis : str, default 'x'
        Axis to flip around, either 'x' or 'y'.

    Returns
    -------
    np.ndarray
        Flipped polygon as Nx2 array.

    """
    if axis == "x":
        return poly * np.array([1, -1])
    elif axis == "y":
        return poly * np.array([-1, 1])
    else:
        msg = "Axis must be 'x' or 'y'."
        raise ValueError(msg)


def scale_polygon_by_width(
    poly: np.ndarray,
    target_width: float,
) -> np.ndarray:
    """Scale a polygon so that its base width = base_width_m. Proportions are preserved.

    Parameters
    ----------
    poly : np.ndarray
        Nx2 array of polygon vertices.
    target_width : float
        Desired width of the polygon base.

    Returns
    -------
    np.ndarray
        Scaled polygon as Nx2 array.

    """
    width = poly[:, 1].max() - poly[:, 1].min()
    return poly * (target_width / width)


def translate_polygon(
    poly: np.ndarray,
    offset: tuple[float, float],
) -> np.ndarray:
    """Translate polygon by offset (dx, dy).

    Parameters
    ----------
    poly : np.ndarray
        Nx2 array of polygon vertices.
    offset : tuple of float
        (dx, dy) translation offsets.

    Returns
    -------
    np.ndarray
        Translated polygon as Nx2 array.

    """
    return poly + np.array(offset)


def calculate_midpoint(
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> tuple[float, float]:
    """Calculate the midpoint between two points p0 and p1.

    Parameters
    ----------
    p0 : tuple of float
        (x0, y0) coordinates of the first point.
    p1 : tuple of float
        (x1, y1) coordinates of the second point.

    Returns
    -------
    tuple of float
        (x, y) coordinates of the midpoint.

    """
    return ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)


def calculate_angle(
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> float:
    """Calculate the angle in radians between two points p0 and p1.

    Parameters
    ----------
    p0 : tuple of float
        (x0, y0) coordinates of the first point.
    p1 : tuple of float
        (x1, y1) coordinates of the second point.

    Returns
    -------
    float
        Angle in radians from p0 to p1.

    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return np.arctan2(dy, dx)


def meters_to_lonlat(poly: np.ndarray, p0_m: tuple[float, float]) -> np.ndarray:
    """Convert polygon vertices from local meters to lon/lat relative to a reference point p0.

    Parameters
    ----------
    poly : np.ndarray
        Nx2 array of polygon vertices in meters.
    p0_m : tuple of float
        (lon0, lat0) reference point in degrees.

    Returns
    -------
    np.ndarray
        Nx2 array of polygon vertices in (lon, lat) degrees.

    """
    R = 6378137.0  # equitorial radius in meters
    lon0, lat0 = p0_m
    x, y = poly[:, 0], poly[:, 1]
    dlon = (x / (R * np.cos(np.radians(lat0)))) * (180.0 / np.pi)
    dlat = (y / R) * (180.0 / np.pi)
    return np.column_stack((lon0 + dlon, lat0 + dlat))
