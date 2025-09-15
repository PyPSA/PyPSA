"""Define common functions for plotting maps in PyPSA."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pypsa.constants import EARTH_RADIUS

if TYPE_CHECKING:
    from pypsa import Network


def apply_cmap(  # noqa
    colors: pd.Series,
    cmap: str | mcolors.Colormap | None,
    cmap_norm: mcolors.Normalize | None = None,
) -> pd.Series:
    if np.issubdtype(colors.dtype, np.number):
        if not isinstance(cmap, mcolors.Colormap):
            cmap = plt.get_cmap(cmap)
        if not cmap_norm:
            cmap_norm = plt.Normalize(vmin=colors.min(), vmax=colors.max())
        colors = colors.apply(lambda cval: cmap(cmap_norm(cval)))
    return colors


def as_branch_series(  # noqa
    ser: pd.Series | dict | list, arg: str, c_name: str, n: "Network"
) -> pd.Series:
    ser = pd.Series(ser, index=n.components[c_name].static.index)
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


def add_jitter(
    x: pd.Series, y: pd.Series, jitter: float
) -> tuple[pd.Series, pd.Series]:
    """Add random jitter to Series data, preserving index and name.

    Parameters
    ----------
    x : pd.Series
        X data.
    y : pd.Series
        Y data.
    jitter : float
        The amount of jitter to add. Function adds a random number between -jitter and
        jitter to each element in the data arrays.

    Returns
    -------
    x_jittered : pd.Series
        X data with added jitter.
    y_jittered : pd.Series
        Y data with added jitter.

    """
    rng = np.random.default_rng()  # Create a random number generator
    x_jittered = x + rng.uniform(low=-jitter, high=jitter, size=len(x))
    y_jittered = y + rng.uniform(low=-jitter, high=jitter, size=len(y))

    return x_jittered, y_jittered


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


def to_rgba255_css(color: str, alpha: float = 1.0) -> str:
    """Convert Matplotlib color to CSS rgba() string.

    Parameters
    ----------
    color : str
        Matplotlib color name or hex string.
    alpha : float, default 1.0
        Alpha transparency value between 0 (transparent) and 1 (opaque).

    Returns
    -------
    str
        CSS rgba() string.

    """
    rgb = [round(c * 255) for c in mcolors.to_rgb(color)]
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.2f})"


def set_tooltip_style(
    background_alpha: float = 0.7,
    background_color: str = "black",
    font_color: str = "white",
    font_family: str = "Arial",
    font_size: int = 12,
    max_width: int = 300,
    padding: int = 10,
) -> dict[str, str]:
    """Set CSS style for pydeck tooltips.

    Parameters
    ----------
    background_alpha : float
        Alpha transparency for background color (0 to 1).
    background_color : str
        Matplotlib color name or hex string for background.
    font_color : str
        Font color name or hex string.
    font_family : str
        Font family name.
    font_size : int
        Font size in pixels.
    max_width : int
        Maximum width of tooltip in pixels.
    padding : int
        Padding inside tooltip in pixels.

    Returns
    -------
    dict
        Dictionary of CSS styles for pydeck tooltip.

    """
    return {
        "backgroundColor": to_rgba255_css(background_color, background_alpha),
        "color": font_color,
        "fontFamily": font_family,
        "fontSize": f"{font_size}px",
        "borderCollapse": "collapse",
        "border": "1px solid white",
        "padding": f"{padding}px",
        "maxWidth": f"{max_width}px",
        "overflowWrap": "break-word",
        "overflow": "hidden",
    }


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


def meters_to_lonlat(
    poly: np.ndarray, p0_m: tuple[float, float], r: float = EARTH_RADIUS
) -> np.ndarray:
    """Convert polygon vertices from local meters to lon/lat relative to a reference point p0.

    Parameters
    ----------
    poly : np.ndarray
        Nx2 array of polygon vertices in meters.
    p0_m : tuple of float
        (lon0, lat0) reference point in degrees.
    r : float, default EARTH_RADIUS
        Earth radius in meters.

    Returns
    -------
    np.ndarray
        Nx2 array of polygon vertices in (lon, lat) degrees.

    """
    lon0, lat0 = p0_m
    x, y = poly[:, 0], poly[:, 1]
    dlon = (x / (r * np.cos(np.radians(lat0)))) * (180.0 / np.pi)
    dlat = (y / r) * (180.0 / np.pi)
    return np.column_stack((lon0 + dlon, lat0 + dlat))


def shorten_string(s: Any, max_length: int | None = None) -> str:
    """Convert any object to a string and shorten it with an ellipsis ("...") if it exceeds the specified maximum length.

    Parameters
    ----------
    s : Any
        The object to convert to a string.
    max_length : int, optional
        Maximum allowed string length. If None, no shortening is applied.

    Returns
    -------
    str
        The string representation of the input, shortened if necessary.

    """
    s_str = str(s)
    if max_length is not None and len(s_str) > max_length:
        return s_str[:max_length] + "..."
    return s_str


def round_value(
    v: Any, rounding: int | dict[str, int] | None = None, key: str | None = None
) -> Any:
    """Round a numeric value based on the rounding specification.

    Parameters
    ----------
    v : Any
        The value to round.
    rounding : int or dict of str to int, optional
        - If int, rounds all numbers to this precision.
        - If dict, looks up precision using `key`.
        - If None, no rounding is applied.
    key : str, optional
        Identifier used for dict-based rounding.

    Returns
    -------
    Any
        Rounded numeric value if applicable, otherwise the original value.

    """
    if isinstance(v, (int | float)):
        if isinstance(rounding, int):
            if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
                return int(v)
            return round(v, rounding)
        elif isinstance(rounding, dict) and key in rounding:
            r = rounding[key]
            rounded = round(v, r)
            return int(rounded) if rounded.is_integer() else rounded
    return v


def series_to_html_str(
    df_row: pd.Series,
    columns: list[str] | None = None,
    bold_header: bool = True,
    headline: str | None = None,
    rounding: int | dict | None = None,
    value_align: str = "left",
    max_header_length: int | None = None,
    max_value_length: int | None = None,
) -> str:
    """Convert a pd.Series to html string representation of a vertical table (columns become rows) with optional headline, bold left column, right-aligned values, and rounding.

    Parameters
    ----------
    df_row : pd.Series
        A Series representing a single row of a DataFrame.
    columns : list of str, optional
        Columns to include. Defaults to empty (no rows).
    bold_header : bool
        Whether to make the left column (headers) bold.
    headline : str, optional
        Optional headline to display above the table.
    rounding : int or dict, optional
        Number of decimals to round numeric values. If dict, keys are column names.
    value_align : str
        Alignment for value column: "right", "left", or "center".
    max_header_length : int, optional
        Maximum length of headline. Longer headlines are truncated with "...".
    max_value_length : int, optional
        Maximum length of each value string. Longer values are truncated with "...".

    Returns
    -------
    str
        HTML string representation of the table.

    """
    if not columns and not headline:
        return ""

    # Headline
    table_html = ""
    if headline:
        table_html += f"<b>{shorten_string(headline, max_header_length)}</b>\n"

    if not columns:
        return table_html

    # Extract and process values
    values = df_row[columns].to_numpy(dtype=object)
    if rounding is not None:
        values = np.array(
            [
                round_value(v, rounding, col)
                for v, col in zip(values, columns, strict=False)
            ],
            dtype=object,
        )

    values = np.array(
        [shorten_string(v, max_value_length) for v in values],
        dtype=object,
    )

    # Header column
    left_style = "font-weight:bold" if bold_header else ""
    left_arr = [
        f"<td style='{left_style}'>{col}:</td>" if left_style else f"<td>{col}:</td>"
        for col in columns
    ]

    # Value column
    right_arr = [f"<td style='text-align:{value_align}'>{v}</td>" for v in values]

    # Combine rows
    row_html_arr = [
        f"<tr>{l}{r}</tr>" for l, r in zip(left_arr, right_arr, strict=False)
    ]
    table_html += "<table>\n" + "\n".join(row_html_arr) + "\n</table>"

    return table_html


def df_to_html_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    bold_header: bool = True,
    max_header_length: int | None = None,
    max_value_length: int | None = None,
    rounding: int | dict | None = None,
    value_align: str = "left",
) -> pd.Series:
    """Convert a DataFrame row to a vertical HTML table (columns become rows) with optional headline, bold left column, right-aligned values, and rounding.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with one or more rows.
    columns : list of str, optional
        Columns to include. Defaults to empty (no rows).
    bold_header : bool
        Whether to make the left column (headers) bold.
    max_header_length : int, optional
        Maximum length of headline. Longer headlines are truncated with "...".
    max_value_length : int, optional
        Maximum length of each value string. Longer values are truncated with "...".
    rounding : int or dict, optional
        Number of decimals to round numeric values. If dict, keys are column names.
    value_align : str
        Alignment for value column: "right", "left", or "center".

    Returns
    -------
    pd.Series
        Series of HTML strings for each row in the DataFrame.

    """
    return df.apply(
        lambda row: series_to_html_str(
            row,
            columns=columns,
            bold_header=bold_header,
            headline=row.name,
            rounding=rounding,
            value_align=value_align,
            max_header_length=max_header_length,
            max_value_length=max_value_length,
        ),
        axis=1,
    )
