# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Animation support for PyPSA network map plots."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from pypsa.plot.maps.static import _create_plotter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pypsa.networks import Network

logger = logging.getLogger(__name__)


def _slice_kwargs_for_snapshot(
    kwargs: dict[str, Any],
    snapshot: Any,
) -> dict[str, Any]:
    """Extract per-snapshot data from DataFrame kwargs.

    For each kwarg that is a DataFrame whose index contains ``snapshot``,
    extract the row as a Series. All other kwargs pass through unchanged.
    """
    return {
        key: value.loc[snapshot]
        if isinstance(value, pd.DataFrame) and snapshot in value.index
        else value
        for key, value in kwargs.items()
    }


def animate(  # noqa: D417
    n: Network,
    layouter: Callable | None = None,
    boundaries: tuple[float, float, float, float] | None = None,
    margin: float | None = 0.05,
    projection: Any = None,
    geomap: bool | str = True,
    geomap_resolution: Literal["10m", "50m", "110m"] = "50m",
    geomap_color: dict | bool | None = None,
    title: str = "",
    jitter: float | None = None,
    snapshots: pd.Index | None = None,
    fps: int = 24,
    interval: int | None = None,
    blit: bool = False,
    repeat: bool = False,
    timestamp_kwargs: dict[str, Any] | None = None,
    path: str | Path | None = None,
    writer: str | Any | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    margin_adjustment: dict[str, float] | None = None,
    **kwargs: Any,
) -> FuncAnimation | None:
    """Animate the network plot over snapshots.

    Creates a `matplotlib.animation.FuncAnimation` that iterates over the
    given snapshots, updating time-varying parameters (passed as DataFrames)
    on each frame. Accepts all keyword arguments of
    :func:`~pypsa.plot.maps.static.plot`.

    Parameters
    ----------
    snapshots : pd.Index, optional
        Snapshots to animate over. Defaults to ``n.snapshots``.
    fps : int, default 24
        Frames per second.
    interval : int, optional
        Milliseconds between frames. Overrides ``fps`` if provided.
    blit : bool, default False
        Whether to use blitting for faster drawing.
    repeat : bool, default False
        Whether the animation should loop.
    timestamp_kwargs : dict, optional
        Keyword arguments for the timestamp text artist (e.g. ``fontsize``,
        ``x``, ``y``, ``ha``, ``va``). The ``format`` key, if present, is
        used as a strftime format string for the timestamp.
    path : str or Path, optional
        If provided, save the animation to this file path. When set, the
        figure is closed after saving and ``None`` is returned to avoid
        expensive in-memory rendering in notebooks.
    writer : str or matplotlib writer, optional
        Writer to use when saving (e.g. ``"ffmpeg"``, ``"pillow"``).
    figsize : tuple[float, float], optional
        Figure size ``(width, height)`` in inches.
    dpi : int, optional
        Resolution in dots per inch. Used both for the figure and when
        saving to file.
    margin_adjustment : dict, optional
        Keyword arguments passed to
        :meth:`matplotlib.figure.Figure.subplots_adjust` to control
        whitespace around the plot. Defaults to
        ``{"left": 0, "right": 1, "top": 1, "bottom": 0}`` which
        removes all padding. Pass an empty dict ``{}`` to keep
        matplotlib defaults.
    **kwargs
        All parameters accepted by
        :meth:`~pypsa.plot.maps.static.MapPlotter.draw_map`. Parameters that
        are ``pd.DataFrame`` objects with an index matching ``snapshots`` are
        treated as time-varying and sliced per frame.

    Returns
    -------
    matplotlib.animation.FuncAnimation or None
        The animation object when no ``path`` is given; ``None`` otherwise.

    Examples
    --------
    >>> n.optimize()
    >>> anim = n.plot.animate(line_flow=n.lines_t.p0, geomap=False)
    >>> anim.save("flows.mp4", writer="ffmpeg")

    >>> n.plot.animate(
    ...     line_flow=n.lines_t.p0,
    ...     snapshots=n.snapshots[:24],
    ...     path="day1.gif",
    ...     writer="pillow",
    ...     fps=5,
    ... )

    """
    if snapshots is None:
        snapshots = n.snapshots
    if interval is None:
        interval = 1000 // fps

    # For DataFrame bus_size, peek at first snapshot to determine buses
    bus_size = kwargs.get("bus_size")
    if isinstance(bus_size, pd.DataFrame):
        bus_size = bus_size.iloc[0]

    plotter, geomap, geomap_resolution = _create_plotter(
        n, layouter, boundaries, margin, jitter, geomap, geomap_resolution,
        bus_size,
    )

    # Initialize axis once (draws geomap background)
    plotter.init_axis(
        projection=projection,
        geomap=geomap,
        geomap_resolution=geomap_resolution,
        geomap_color=geomap_color,
        title=title,
        boundaries=boundaries,
        figsize=figsize,
        dpi=dpi,
    )
    ax = plotter.ax
    fig = ax.get_figure()  # type: ignore

    # Reduce whitespace around the plot
    adj = {"left": 0, "right": 1, "top": 1, "bottom": 0}
    if margin_adjustment is not None:
        adj = margin_adjustment
    if adj:
        fig.subplots_adjust(**adj)  # type: ignore[union-attr]

    # Timestamp text
    ts_kw: dict[str, Any] = {
        "ha": "left", "va": "top", "fontsize": 12,
    }
    ts_format = None
    if timestamp_kwargs:
        ts_format = timestamp_kwargs.pop("format", None)
        ts_kw.update(timestamp_kwargs)
    timestamp_text = ax.text(  # type: ignore
        ts_kw.pop("x", 0.02), ts_kw.pop("y", 0.98), "",
        transform=ts_kw.pop("transform", ax.transAxes),  # type: ignore
        **ts_kw,
    )

    collections: list[Any] = []

    def _update(frame_idx: int) -> list:
        for coll in collections:
            coll.remove()
        collections.clear()

        snapshot = snapshots[frame_idx]
        frame_kwargs = _slice_kwargs_for_snapshot(kwargs, snapshot)

        result = plotter.draw_map(
            ax=plotter.ax, projection=projection, geomap=False,
            title=title, _skip_init_axis=True, **frame_kwargs,
        )

        for key in ("nodes", "branches", "flows"):
            collections.extend(c for c in result[key].values() if c is not None)

        if ts_format and hasattr(snapshot, "strftime"):
            timestamp_text.set_text(snapshot.strftime(ts_format))
        else:
            timestamp_text.set_text(str(snapshot))

        return collections + [timestamp_text]

    anim = FuncAnimation(
        fig, _update, frames=len(snapshots), interval=interval,  # type: ignore
        blit=blit, repeat=repeat,
    )

    if path is not None:
        anim.save(str(path), writer=writer, fps=fps, dpi=dpi)
        plt.close(fig)
        return None

    return anim
