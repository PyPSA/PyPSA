"""Define common functions for plotting maps in PyPSA."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import networkx as nx
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
        n.buses[["x", "y"]] = coordinates
        return None
    return coordinates.x, coordinates.y
