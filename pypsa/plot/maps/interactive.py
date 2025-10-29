# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Plot the network interactively using plotly or pydeck."""

import logging
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pltly
import pydeck as pdk
import pyproj

from pypsa.common import _convert_to_series, deprecated_kwargs
from pypsa.components.common import as_components
from pypsa.plot.maps.common import (
    _is_cartopy_available,
    add_jitter,
    apply_cmap,
    apply_layouter,
    as_branch_series,
    calculate_angle,
    calculate_midpoint,
    create_rgba_colors,
    df_to_html_table,
    feature_to_geojson,
    flip_polygon,
    get_global_stat,
    meters_to_lonlat,
    rotate_polygon,
    scale_polygon_by_width,
    scale_to_max_abs,
    series_to_pdk_path,
    set_tooltip_style,
    shapefile_to_geojson,
    to_rgba255,
)

if TYPE_CHECKING:
    from pypsa.networks import Network


logger = logging.getLogger(__name__)


_token_required_mb_styles = [
    "basic",
    "streets",
    "outdoors",
    "light",
    "dark",
    "satellite",
    "satellite-streets",
]

_open__mb_styles = [
    "open-street-map",
    "white-bg",
    "carto-positron",
    "carto-darkmatter",
    "stamen-terrain",
    "stamen-toner",
    "stamen-watercolor",
]


@deprecated_kwargs(
    deprecated_in="1.0",
    removed_in="2.0",
    bus_sizes="bus_size",
    bus_colors="bus_color",
    bus_split_circles="bus_split_circle",
    branch_colors="branch_color",
    branch_widths="branch_width",
    arrow_colors="arrow_color",
    geomap_colors="geomap_color",
    line_colors="line_color",
    line_widths="line_width",
    link_colors="link_color",
    link_widths="link_width",
    transformer_colors="transformer_color",
    transformer_widths="transformer_width",
)
def iplot(
    n: "Network",
    fig: dict | None = None,
    bus_color: str | dict | pd.Series = "cadetblue",
    bus_alpha: float = 1,
    bus_size: float | pd.Series = 10,
    bus_cmap: str | mcolors.Colormap | None = None,
    bus_colorbar: dict | None = None,
    bus_text: pd.Series | None = None,
    line_color: str | pd.Series = "rosybrown",
    link_color: str | pd.Series = "darkseagreen",
    transformer_color: str | pd.Series = "orange",
    line_width: float | pd.Series = 3,
    link_width: float | pd.Series = 3,
    transformer_width: float | pd.Series = 3,
    line_text: pd.Series | None = None,
    link_text: pd.Series | None = None,
    transformer_text: pd.Series | None = None,
    layouter: Callable | None = None,
    title: str = "",
    size: tuple[int, int] | None = None,
    branch_components: Sequence[str] | set[str] | None = None,
    iplot: bool = True,
    jitter: float | None = None,
    mapbox: bool = False,
    mapbox_style: str = "open-street-map",
    mapbox_token: str = "",
    mapbox_parameters: dict | None = None,
) -> dict:
    """Plot the network buses and lines interactively using plotly.

    Parameters
    ----------
    n : pypsa.Network
        The network to plot.
    fig : dict, default None
        If not None, figure is built upon this fig.
    bus_color : dict/pandas.Series
        Colors for the buses, defaults to "cadetblue". If bus_size is a
        pandas.Series with a Multiindex, bus_color defaults to the
        n.c.carriers.static['color'] column.
    bus_alpha : float
        Add alpha channel to buses, defaults to 1.
    bus_size : float/pandas.Series
        Sizes of bus points, defaults to 10.
    bus_cmap : mcolors.Colormap/str
        If bus_color are floats, this color map will assign the colors
    bus_colorbar : dict
        Plotly colorbar, e.g. {'title' : 'my colorbar'}
    bus_text : pandas.Series
        Text for each bus, defaults to bus names
    line_color : str/pandas.Series
        Colors for the lines, defaults to 'rosybrown'.
    link_color : str/pandas.Series
        Colors for the links, defaults to 'darkseagreen'.
    transformer_color : str/pandas.Series
        Colors for the transfomer, defaults to 'orange'.
    line_width : dict/pandas.Series
        Widths of lines, defaults to 1.5
    link_width : dict/pandas.Series
        Widths of links, defaults to 1.5
    transformer_width : dict/pandas.Series
        Widths of transformer, defaults to 1.5
    line_text : pandas.Series
        Text for lines, defaults to line names.
    link_text : pandas.Series
        Text for links, defaults to link names.
    transformer_text : pandas.Series
        Text for transformers, defaults to transformer names.
    layouter : networkx.drawing.layout function, default None
        Layouting function from [networkx](https://networkx.github.io/) which
        overrules coordinates given in `n.buses[['x', 'y']]`. See
        [list](https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout)
        of available options.
    title : string
        Graph title
    size : None|tuple
        Tuple specifying width and height of figure; e.g. (width, heigh).
    branch_components : list of str
        Branch components to be plotted, defaults to Line and Link.
    iplot : bool, default True
        Automatically do an interactive plot of the figure.
    jitter : None|float
        Amount of random noise to add to bus positions to distinguish
        overlapping buses
    mapbox : bool, default False
        Switch to use Mapbox.
    mapbox_style : str, default 'open-street-map'
        Define the mapbox layout style of the interactive plot. If this is set
        to a mapbox layout, the argument `mapbox_token` must be a valid Mapbox
        API access token.

        Valid open layouts are:
            open-street-map, white-bg, carto-positron, carto-darkmatter,
            stamen-terrain, stamen-toner, stamen-watercolor

        Valid mapbox layouts are:
            basic, streets, outdoors, light, dark, satellite, satellite-streets

    mapbox_token : string
        Mapbox API access token. Obtain from `https://www.mapbox.com`.
        Can also be included in mapbox_parameters as `accesstoken=mapbox_token`.
    mapbox_parameters : dict
        Configuration parameters of the Mapbox layout.
        E.g. {"bearing": 5, "pitch": 10, "zoom": 1, "style": 'dark'}.


    Returns
    -------
    fig: dictionary for plotly figure

    """
    if fig is None:
        fig = {"data": [], "layout": {}}

    if bus_text is None:
        bus_text = "Bus " + n.c.buses.static.index
    if mapbox_parameters is None:
        mapbox_parameters = {}
    x, y = apply_layouter(n, layouter=layouter, inplace=False)

    rng = np.random.default_rng()  # Create a random number generator
    if jitter is not None:
        x = x + rng.uniform(low=-jitter, high=jitter, size=len(x))
        y = y + rng.uniform(low=-jitter, high=jitter, size=len(y))

    bus_trace = {
        "x": x,
        "y": y,
        "text": bus_text,
        "type": "scatter",
        "mode": "markers",
        "hoverinfo": "text",
        "opacity": bus_alpha,
        "marker": {"color": bus_color, "size": bus_size},
    }

    if bus_cmap is not None:
        bus_trace["marker"]["colorscale"] = bus_cmap

    if bus_colorbar is not None:
        bus_trace["marker"]["colorbar"] = bus_colorbar

    if branch_components is None:
        branch_components = n.branch_components

    branch_color = {
        "Line": line_color,
        "Link": link_color,
        "Transformer": transformer_color,
    }
    branch_width = {
        "Line": line_width,
        "Link": link_width,
        "Transformer": transformer_width,
    }
    branch_text = {
        "Line": line_text,
        "Link": link_text,
        "Transformer": transformer_text,
    }

    shapes = []
    shape_traces = []

    for c in n.components:
        if c.name not in branch_components:
            continue
        b_widths = as_branch_series(branch_width[c.name], "width", c.name, n)
        b_colors = as_branch_series(branch_color[c.name], "color", c.name, n)
        b_text = branch_text[c.name]

        if b_text is None:
            b_text = c.name + " " + c.static.index

        x0 = c.static.bus0.map(x)
        x1 = c.static.bus1.map(x)
        y0 = c.static.bus0.map(y)
        y1 = c.static.bus1.map(y)

        shapes.extend(
            [
                {
                    "type": "line",
                    "opacity": 0.8,
                    "x0": x0[b],
                    "y0": y0[b],
                    "x1": x1[b],
                    "y1": y1[b],
                    "line": {"color": b_colors[b], "width": b_widths[b]},
                }
                for b in c.static.index
            ]
        )

        shape_traces.append(
            {
                "x": 0.5 * (x0 + x1),
                "y": 0.5 * (y0 + y1),
                "text": b_text,
                "type": "scatter",
                "mode": "markers",
                "hoverinfo": "text",
                "marker": {"opacity": 0.0},
            }
        )

    if mapbox:
        shape_traces_latlon = []
        for st in shape_traces:
            st["lon"] = st.pop("x")
            st["lat"] = st.pop("y")
            shape_traces_latlon.append(go.Scattermapbox(st))
        shape_traces = shape_traces_latlon

        shapes_mapbox = []
        for s in shapes:
            s["lon"] = [s.pop("x0"), s.pop("x1")]
            s["lat"] = [s.pop("y0"), s.pop("y1")]
            shapes_mapbox.append(go.Scattermapbox(s, mode="lines"))
        shapes = shapes_mapbox

        bus_trace["lon"] = bus_trace.pop("x")
        bus_trace["lat"] = bus_trace.pop("y")
        bus_trace = go.Scattermapbox(bus_trace)

        fig["data"].extend(shapes + shape_traces + [bus_trace])
    else:
        fig["data"].extend([bus_trace] + shape_traces)

    fig["layout"].update({"title": title, "hovermode": "closest", "showlegend": False})

    if size is not None:
        if len(size) != 2:
            msg = "Parameter size must specify a tuple (width, height)."
            raise ValueError(msg)
        fig["layout"].update({"width": size[0], "height": size[1]})

    if mapbox:
        if mapbox_token != "":
            mapbox_parameters["accesstoken"] = mapbox_token

        mapbox_parameters.setdefault("style", mapbox_style)

        if (
            mapbox_parameters["style"] in _token_required_mb_styles
            and "accesstoken" not in mapbox_parameters
        ):
            msg = (
                "Using Mapbox layout styles requires a valid access token from "
                "[https://www.mapbox.com/](https://www.mapbox.com/), style which do not require a token "
                "are:\n{', '.join(_open__mb_styles)}."
            )
            raise ValueError(msg)

        if "center" not in mapbox_parameters:
            lon = (n.c.buses.static.x.min() + n.c.buses.static.x.max()) / 2
            lat = (n.c.buses.static.y.min() + n.c.buses.static.y.max()) / 2
            mapbox_parameters["center"] = {"lat": lat, "lon": lon}

        if "zoom" not in mapbox_parameters:
            mapbox_parameters["zoom"] = 2

        fig["layout"]["mapbox"] = mapbox_parameters
    else:
        fig["layout"]["shapes"] = shapes

    if iplot:
        pltly.iplot(fig)

    return fig


class PydeckPlotter:
    """Class to create and manage an interactive pydeck map for a PyPSA network."""

    # Class-level constants
    VALID_MAP_STYLES = {
        "light": pdk.map_styles.LIGHT,
        "dark": pdk.map_styles.DARK,
        "road": pdk.map_styles.ROAD,
        "dark_no_labels": pdk.map_styles.DARK_NO_LABELS,
        "light_no_labels": pdk.map_styles.LIGHT_NO_LABELS,
        "none": "",
    }
    ARROW = np.array(
        [
            [1, 0],  # reference triangle as arrow head
            [0, 0.5],  # base right
            [0, -0.5],  # base left
        ]
    )
    ARROW = ARROW - ARROW.mean(axis=0)  # center at geometric center

    PROJ = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    PROJ_INV = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    def __init__(
        self,
        n: "Network",
        map_style: str,
        view_state: dict | pdk.ViewState | None = None,
        layouter: Callable | None = None,
        jitter: float | None = None,
    ) -> None:
        """Initialize the PydeckPlotter.

        Parameters
        ----------
        n : Network
            The PyPSA network to plot.
        map_style : str
            Map style to use for the plot. One of 'light', 'dark', 'road', 'dark_no_labels', 'light_no_labels', and 'none'.
        view_state : dict/pdk.ViewState/None, optional
            Initial view state for the map. If None, a default view state is created.
            If a dict is provided, it should contain keys like 'longitude', 'latitude', 'zoom', 'pitch', and 'bearing'.
        layouter : Callable | None, optional
            Layouting function from [networkx](https://networkx.github.io/) which
            overrules coordinates given in `n.buses[['x', 'y']]`. See
            [list](https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout)
            of available options.
        jitter : float, optional
            Amount of random noise to add to node positions

        """
        self._n: Network = n
        self._x: pd.Series
        self._y: pd.Series
        self._init_xy(layouter=layouter)
        if jitter:
            self._x, self._y = add_jitter(x=self._x, y=self._y, jitter=jitter)

        self._map_style: str = self._init_map_style(map_style)
        self._view_state: pdk.ViewState = self._init_view_state(
            view_state=view_state,
        )
        self._layers: dict[str, pdk.Layer] = {}
        self._tooltip_style: dict[str, str] = set_tooltip_style()
        self._component_data: dict[str, pd.DataFrame] = {}

    @property
    def map_style(self) -> str:
        """Get the current map style."""
        return self._map_style

    @property
    def tooltip_style(self) -> dict:
        """Get the current tooltip CSS styles."""
        return self._tooltip_style

    @property
    def layers(self) -> dict[str, pdk.Layer]:
        """Get the layers of the interactive map."""
        return self._layers

    def _init_xy(
        self,
        layouter: Callable | None = None,
    ) -> None:
        """Initialize x and y coordinates from the network buses."""
        buses = self._n.c.buses.static

        # Check if all x and y are missing/zero → then fallback to layouter
        is_empty = (buses[["x", "y"]].isnull() | (buses[["x", "y"]] == 0)).all().all()

        if layouter or is_empty:
            self._x, self._y = apply_layouter(self._n, layouter, inplace=False)
        else:
            self._x, self._y = buses["x"], buses["y"]

        # Validation mask for WGS84 coordinates
        valid = (
            self._x.notnull()
            & self._y.notnull()
            & (self._x >= -180)
            & (self._x <= 180)  # longitude
            & (self._y >= -90)
            & (self._y <= 90)  # latitude
        )

        # Keep only valid buses
        dropped = (~valid).sum()
        if dropped:
            logger.warning("Dropping %d buses with invalid WGS84 coordinates", dropped)
            self._x, self._y = self._x[valid], self._y[valid]

    def _init_map_style(self, map_style: str) -> str:
        """Set the initial map style for the interactive map."""
        if map_style not in self.VALID_MAP_STYLES:
            msg = (
                f"Invalid map style '{map_style}'.\n"
                f"Must be one of: {', '.join(self.VALID_MAP_STYLES)}."
            )
            raise ValueError(msg)
        return map_style

    def _init_view_state(
        self,
        view_state: dict | pdk.ViewState | None = None,
    ) -> pdk.ViewState:
        """Compute the initial view state based on network bus coordinates.

        Parameters
        ----------
        view_state : dict/pdk.ViewState/None, optional
            Initial view state for the map. If None, a default view state is created.
            If a dict is provided, it should contain keys like 'longitude', 'latitude', 'zoom', 'pitch', and 'bearing'.

        Returns
        -------
        pdk.ViewState
            The initialized view state for the map.

        """
        if isinstance(view_state, pdk.ViewState):
            return view_state

        vs = {
            "longitude": self._x.mean(),
            "latitude": self._y.mean(),
            "zoom": 4,
            "min_zoom": None,
            "max_zoom": None,
            "pitch": 0,
            "bearing": 0,
        }

        if isinstance(view_state, dict):
            vs.update(view_state)

        return pdk.ViewState(**vs)

    @property
    def view_state(self) -> pdk.ViewState:
        """Get the current view state of the map.

        Returns
        -------
        pdk.ViewState
            The current view state of the map.

        """
        return self._view_state

    @staticmethod
    def _make_arrows(
        flow: float,
        p0_geo: tuple[float, float],
        p1_geo: tuple[float, float],
        arrow_size_factor: float,
    ) -> list[tuple[float, float]]:
        """Create arrows scaled and projected for a given flow and p0, p1 geographical coordinates. Additional scaling by arrow_size_factor.

        Parameters
        ----------
        flow : float
            Flow value to determine arrow direction and size.
        p0_geo : tuple
            Geographical coordinates (lon, lat) of the start point.
        p1_geo : tuple
            Geographical coordinates (lon, lat) of the end point.
        arrow_size_factor : float
            Factor to scale the arrow size.

        Returns
        -------
        list of tuple
            List of (lon, lat) tuples representing the arrow polygon coordinates.

        """
        # project end points from lon/lat into meters
        p0_m = PydeckPlotter.PROJ.transform(*p0_geo)
        p1_m = PydeckPlotter.PROJ.transform(*p1_geo)

        # return empty list if any bus coordinates are nan
        if np.any(np.isnan(p0_m)) or np.any(np.isnan(p1_m)):
            return []

        # center point (tuple) between p0 and p1 and angle
        p_center = calculate_midpoint(p0_m, p1_m)
        angle = calculate_angle(p0_m, p1_m)

        # geometric operations on arrow
        base_width_m = abs(flow) * arrow_size_factor
        arrow = scale_polygon_by_width(PydeckPlotter.ARROW, base_width_m)
        if flow < 0:
            arrow = flip_polygon(arrow, "y")
        arrow = rotate_polygon(arrow, angle)

        # transform back to lon/lat relative to center point
        coords_center = PydeckPlotter.PROJ_INV.transform(*p_center)
        arrow = meters_to_lonlat(arrow, coords_center)

        return arrow.tolist()

    # Data wrangling
    def prepare_component_data(
        self,
        component: str,
        default_columns: list[str] | None = None,
        extra_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Prepare data for a specific component type.

        Parameters
        ----------
        component : str
            Name of the component type, e.g. "Bus", "Line", "Link", "Transformer".
        default_columns : list of str, optional
            List of default columns to include for the component.
        extra_columns : list of str, optional
            Additional columns to include for the component.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the prepared data for the component.

        """
        df = as_components(self._n, component).static
        if default_columns is None:
            default_columns = []

        layer_columns = default_columns

        if extra_columns:
            extra_columns = [
                col for col in extra_columns if col not in default_columns
            ]  # Drop default columns from extra_columns
            missing_columns = [col for col in extra_columns if col not in df.columns]
            valid_columns = [col for col in extra_columns if col in df.columns]

            if missing_columns:
                msg = (
                    f"Columns {missing_columns} not found in {component}. "
                    f"Using only valid columns: {valid_columns}."
                )
                logger.warning(msg)

            layer_columns.extend(valid_columns)

        df = df[layer_columns].copy()
        df.index.name = "name"

        return df

    # Layer functions
    def add_bus_layer(
        self,
        bus_size: float | dict | pd.Series = 25,  # km²
        bus_size_factor: float = 1.0,
        bus_size_max: float = 10000,  # km²
        bus_color: str | dict | pd.Series = "cadetblue",
        bus_cmap: str | mcolors.Colormap = "Reds",
        bus_cmap_norm: mcolors.Normalize | None = None,
        bus_alpha: float | dict | pd.Series = 0.9,
        bus_columns: list | None = None,
        auto_scale: bool = False,
        tooltip: bool = True,
    ) -> None:
        """Add a bus layer of Pydeck type ScatterplotLayer to the interactive map.

        Parameters
        ----------
        bus_size : float/dict/pandas.Series
            Sizes of bus points in km² (corresponds to circle area), defaults to 25 km².
        bus_size_factor : float, default 1.0
            Bus sizes are scaled by this factor.
        bus_size_max : float, default 10000
            Maximum size of bus points in km² when auto-scaling.
        bus_color : str/dict/pandas.Series
            Colors for the buses, defaults to "cadetblue". If bus_size is a
            pandas.Series with a Multiindex, bus_color defaults to the
            n.c.carriers.static['color'] column.
        bus_cmap : mcolors.Colormap/str, default 'Reds'
            If bus_color are floats, this color map will assign the colors.
        bus_cmap_norm : mcolors.Normalize/None
            Normalization for bus_cmap, defaults to None.
        bus_alpha : float/dict/pandas.Series
            Add alpha channel to buses, defaults to 0.9.
        bus_columns : list, default None
            List of bus columns to include.
        auto_scale : bool, default False
            Whether to auto-scale bus sizes to fit within bus_size_max.
        tooltip : bool, default True
            Whether to show a tooltip on hover.

        Returns
        -------
        None

        """
        msg = "bus_size_factor must be non-negative"
        if bus_size_factor < 0:
            raise ValueError(msg)

        bus_data = self.prepare_component_data("Bus", extra_columns=bus_columns)

        valid_idx = self._x.index.intersection(bus_data.index)
        bus_data = bus_data.loc[valid_idx].copy()
        self._component_data["Bus"] = bus_data
        bus_data["x"] = self._x.loc[bus_data.index]
        bus_data["y"] = self._y.loc[bus_data.index]

        # Handle bus sizes
        bus_size_series = _convert_to_series(bus_size, bus_data.index).clip(lower=0)
        if bus_size_series.eq(0).all():
            return
        bus_data["size"] = bus_size_series

        if auto_scale:
            bus_size_series = scale_to_max_abs(bus_size_series, bus_size_max)
        else:
            bus_size_series = bus_size_series * bus_size_factor

        bus_data["size_pdk"] = (
            bus_size_series * 1e6 / np.pi
        ) ** 0.5  # convert to meters for Pydeck

        # 6. Handle colors and alpha
        color_series = _convert_to_series(bus_color, bus_data.index)
        alpha_series = _convert_to_series(bus_alpha, bus_data.index)
        color_series = apply_cmap(color_series, bus_cmap, bus_cmap_norm)
        bus_data["rgba"] = [
            to_rgba255(c, a) for c, a in zip(color_series, alpha_series, strict=False)
        ]

        # 7. Create tooltips if requested
        if tooltip:
            self.create_tooltips("Bus")

        # 8. Create Pydeck layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=bus_data,
            get_position=["x", "y"],
            get_color="rgba",
            get_radius="size_pdk",
            pickable=True,
            auto_highlight=True,
            parameters={"depthTest": False},  # prevent z-fighting
        )

        # 9. Store the layer
        self._layers["Bus"] = layer

    @staticmethod
    def _pie_slice_vertices(
        p_center: tuple[float, float],
        radius: float,
        start_angle: float,
        end_angle: float,
        points_per_radian: int = 5,
    ) -> list[list[float]]:
        """Generate vertices of a pie slice as a closed polygon using numpy.

        Parameters
        ----------
        p_center : tuple of float
            Center point of the pie chart as (lon, lat).
        radius : float
            Radius of the pie chart in meters.
        start_angle : float
            Starting angle of the pie slice in radians.
        end_angle : float
            Ending angle of the pie slice in radians.
        points_per_radian : int, default 5
            Number of points per radian for pie chart resolution.

        Returns
        -------
        list of list of float
            List of [lon, lat] coordinates representing the pie slice polygon.

        """
        angle_span = end_angle - start_angle
        steps = max(1, int(np.ceil(points_per_radian * abs(angle_span))))
        angles = np.linspace(start_angle, end_angle, steps + 1)

        # arc vertices in meters
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        arc = np.column_stack((x, y))  # create 2D arc polygon
        arc = meters_to_lonlat(
            arc,
            p_center,
        )  # convert arc vertices into lon/lat relative to center point

        # full slice polygon (closed polygon center -> arc -> back to center)
        coords = np.vstack(
            (
                p_center,
                arc,
                p_center,
            )
        )
        return (
            coords.tolist()
        )  # object needs to be list for pydeck's json serialisation to work

    @staticmethod
    def _make_pie(
        bus: str,
        p_center: tuple[float, float],
        radius_m: float,
        values: np.ndarray,
        colors: list[list[int]],
        labels: list[str],
        points_per_radian: int = 5,
        flip_y: bool = False,
        bus_split_circle: bool = False,
    ) -> list[dict]:
        """Create pie chart polygons with metadata using numpy.

        Parameters
        ----------
        bus : str
            Name of the bus for which the pie chart is created.
        p_center : tuple of float
            Center point of the pie chart as (lon, lat).
        radius_m : float
            Radius of the pie chart in meters.
        values : np.ndarray
            Values for each pie slice.
        colors : list of list of int
            List of RGBA colors for each pie slice.
        labels : list of str
            Labels for each pie slice.
        points_per_radian : int, default 5
            Number of points per radian for pie chart resolution.
        flip_y : bool, default False
            Flip the pie chart vertically. Useful for negative values in bus_split_circle mode.
        bus_split_circle : bool, default False
            Draw half circles if True. The upper half circle includes all positive values,
            the lower half circle all negative values.

        Returns
        -------
        list of dict
            List of dictionaries containing pie slice polygons and metadata.

        """
        EPS = 1e-6
        if len(values) == 0 or (
            np.sum(values[values > 0]) < EPS and np.sum(values[values < 0]) > -EPS
        ):
            return []

        circ = np.pi if bus_split_circle else 2 * np.pi
        flip_y_factor = -1 if flip_y else 1
        rotate_by_quarter = 0 if bus_split_circle else np.pi / 2

        angles = flip_y_factor * np.array(values) / np.sum(values) * circ
        start_angles = np.concatenate(([0], np.cumsum(angles)[:-1])) + rotate_by_quarter

        polygons = [
            {
                "polygon": PydeckPlotter._pie_slice_vertices(
                    p_center,
                    radius_m,
                    start,
                    start + delta,
                    points_per_radian,
                ),
                "color": color,
                "bus": bus,
                "label": label,
                "size": val,
            }
            for val, color, label, start, delta in zip(
                values, colors, labels, start_angles, angles, strict=False
            )
        ]
        return polygons

    def add_pie_chart_layer(
        self,
        bus_size: pd.Series,
        bus_size_factor: float | None = None,
        bus_size_max: float = 10000,  # km²
        bus_split_circle: bool = False,
        bus_alpha: float | dict | pd.Series = 0.9,
        bus_columns: list | None = None,
        points_per_radian: int = 5,
        auto_scale: bool = False,
        tooltip: bool = True,
    ) -> None:
        """Add a bus layer of Pydeck type ScatterplotLayer to the interactive map.

        Parameters
        ----------
        bus_size : float/dict/pandas.Series
            Sizes of bus points in radius² (km²), defaults to 25.
        bus_size_factor : float, default 1.0
            Bus sizes are scaled by this factor.
        bus_size_max : float, default 10000
            Maximum size of bus points in km² when auto-scaling.
        bus_split_circle : bool, default False
            Draw half circles if bus_size is a pandas.Series with a Multiindex.
            If set to true, the upper half circle per bus then includes all positive values
            of the series, the lower half circle all negative values. Defaults to False.
        bus_alpha : float/dict/pandas.Series
            Add alpha channel to buses, defaults to 0.9.
        bus_columns : list, default None
            List of bus columns to include.
        points_per_radian : int, default 5
            Number of points per radian for pie chart resolution.
        auto_scale : bool, default False
            Whether to auto-scale bus sizes to fit within bus_size_max.
        tooltip : bool, default True
            Whether to show a tooltip on hover.

        Returns
        -------
        None

        """
        EPS = 1e-6  # Small epsilon to avoid numerical issues
        bus_data = self.prepare_component_data(
            "Bus",
            extra_columns=bus_columns,
        )

        # Only keep buses with valid coordinates, same index order as self._x and self._y
        bus_data = bus_data.loc[self._x.index[self._x.index.isin(bus_data.index)]]

        # Drop tiny values to avoid numerical issues
        bus_size = bus_size.drop(bus_size[abs(bus_size) < EPS].index)

        # Reindex first level of MultiIndex to only valid buses
        bus_size = bus_size.reindex(
            bus_data.index.intersection(bus_size.index.get_level_values(0)),
            level=0,
        )
        bus_size = bus_size.unstack(level=1, fill_value=0)
        carrier_order = bus_size.columns.to_numpy()

        valid_buses = bus_data.index.intersection(bus_size.index)
        bus_size = bus_size.loc[valid_buses]

        # --- Split positive and negative contributions ---
        bus_area_pos = bus_size.clip(lower=0).sum(axis=1)  # positive only
        bus_area_neg = (-bus_size.clip(upper=0)).sum(axis=1)  # negative magnitudes

        # --- Global scaling factor based on largest magnitude ---
        max_val = max(bus_area_pos.max(), bus_area_neg.max())

        if auto_scale:
            scale_factor = bus_size_max / max_val if max_val > 0 else 1
        else:
            scale_factor = bus_size_factor

        # Apply scaling
        bus_area_pos = bus_area_pos * scale_factor
        bus_area_neg = bus_area_neg * scale_factor

        # Convert to m²
        bus_area_pos = bus_area_pos * 1e6
        bus_area_neg = bus_area_neg * 1e6

        # Radii in meters
        bus_radius_pos = (bus_area_pos / np.pi) ** 0.5
        bus_radius_neg = (bus_area_neg / np.pi) ** 0.5

        # Convert to NumPy arrays for speed-up
        bus_coords = np.column_stack(
            [self._x.loc[valid_buses], self._y.loc[valid_buses]]
        )  # assumes that bus_data is aligned with self._x and self._y, done above
        bus_indices = valid_buses.to_numpy()
        bus_values = bus_size.to_numpy()

        alphas = _convert_to_series(bus_alpha, bus_size.index)
        carrier_colors = self._n.c.carriers.static["color"]
        carrier_rgba = {
            bus: {c: to_rgba255(col, alphas[bus]) for c, col in carrier_colors.items()}
            for bus in bus_size.index
        }

        polygons = []
        for i, bus in enumerate(bus_indices):
            values = bus_values[i]
            x, y = bus_coords[i]

            pos_mask = values > 0
            neg_mask = values < 0

            if bus_split_circle and np.any(values < 0):
                if np.any(pos_mask) and bus_radius_pos[bus] > 0:
                    colors = [carrier_rgba[bus][c] for c in carrier_order[pos_mask]]
                    labels = list(carrier_order[pos_mask])
                    vals = values[pos_mask].round(3)

                    poly_pos = PydeckPlotter._make_pie(
                        bus=bus,
                        p_center=(x, y),
                        radius_m=bus_radius_pos[bus],
                        values=vals,
                        colors=colors,
                        labels=labels,
                        points_per_radian=points_per_radian,
                        flip_y=False,
                        bus_split_circle=True,
                    )
                    polygons.extend(poly_pos)

                if np.any(neg_mask) and bus_radius_neg[bus] > 0:
                    colors = [carrier_rgba[bus][c] for c in carrier_order[neg_mask]]
                    labels = list(carrier_order[neg_mask])
                    vals = (-values[neg_mask]).round(3)
                    poly_neg = PydeckPlotter._make_pie(
                        bus=bus,
                        p_center=(x, y),
                        radius_m=bus_radius_neg[bus],
                        values=vals,
                        colors=colors,
                        labels=labels,
                        points_per_radian=points_per_radian,
                        flip_y=True,
                        bus_split_circle=True,
                    )
                    polygons.extend(poly_neg)
            elif np.any(pos_mask) and bus_radius_pos[bus] > 0:
                colors = [carrier_rgba[bus][c] for c in carrier_order[pos_mask]]
                labels = list(carrier_order[pos_mask])
                vals = values[pos_mask].round(3)
                poly_pos = PydeckPlotter._make_pie(
                    bus=bus,
                    p_center=(x, y),
                    radius_m=bus_radius_pos[bus],
                    values=vals,
                    colors=colors,
                    labels=labels,
                    points_per_radian=points_per_radian,
                    flip_y=False,
                    bus_split_circle=False,
                )
                polygons.extend(poly_pos)

        p_data = pd.DataFrame(polygons)
        p_data.set_index("bus", inplace=True)

        # Tooltip
        if tooltip:
            columns = ["label", "size"] + bus_data.columns.tolist()
            p_data = bus_data.reindex(p_data.index).join(p_data)

            p_data["tooltip_html"] = df_to_html_table(
                p_data,
                columns=columns,
                rounding=2,
                value_align="left",
                max_header_length=30,
            )

        layer = pdk.Layer(
            "PolygonLayer",
            data=p_data,
            get_polygon="polygon",
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            parameters={
                "depthTest": False  # To prevent z-fighting issues/flickering in 3D space
            },
        )

        self._layers["PieChart"] = layer

    def init_branch_component_data(
        self,
        c_name: str,
        branch_columns: list | None = None,
    ) -> None:
        """Initialize data for a specific branch component type.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer
        branch_columns : list, default None
            List of branch columns to include. Specify additional columns to include in the tooltip.

        Returns
        -------
        None

        """
        if as_components(self._n, c_name).empty:
            msg = f"No data found for component '{c_name}'. Skipping layer creation."
            logger.warning(msg)
            return

        # Prepare data for lines
        c_data = self.prepare_component_data(
            c_name,
            default_columns=["bus0", "bus1"],
            extra_columns=branch_columns,
        )

        # Only keep rows where both bus0 and bus1 are present
        valid = c_data["bus0"].isin(self._x.index) & c_data["bus1"].isin(self._x.index)
        if not valid.all():
            dropped = (~valid).sum()
            logger.warning(
                "Dropping %d row(s) in '%s' with missing buses", dropped, c_name
            )
            c_data = c_data[valid]

            if c_data.empty:
                return

        self._component_data[c_name] = c_data

    def create_branch_paths(
        self,
        c_name: str,
        geometry: bool = False,
    ) -> None:
        """Create path geometries for a specific branch component type.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer".
        geometry : bool, default False
            Whether to use the geometry column of the branch components.

        Returns
        -------
        None

        """
        c_data = self._component_data[c_name]
        static_data = as_components(self._n, c_name).static

        # Build path column as list of [lon, lat] pairs for each line
        # Assuming x, y are aligned
        if geometry and "geometry" in static_data.columns:
            geoms = static_data["geometry"].reindex(c_data.index)
            branch_paths = series_to_pdk_path(geoms)
        else:
            branch_paths = [
                [[x0, y0], [x1, y1]]
                for x0, y0, x1, y1 in zip(
                    self._x.loc[c_data["bus0"]],
                    self._y.loc[c_data["bus0"]],
                    self._x.loc[c_data["bus1"]],
                    self._y.loc[c_data["bus1"]],
                    strict=False,
                )
            ]
        c_data["path"] = branch_paths

    def create_branch_color(
        self,
        c_name: str,
        branch_color: str | dict | pd.Series = "rosybrown",
        branch_cmap: str | mcolors.Colormap = "viridis",
        branch_cmap_norm: mcolors.Normalize | None = None,
        branch_alpha: float | dict | pd.Series = 0.9,
    ) -> None:
        """Create colors for a specific branch component type.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer
        branch_color : str/dict/pandas.Series
            Colors for the branch component, defaults to 'rosybrown'.
        branch_cmap : str/matplotlib.colors.Colormap, default 'viridis'
            Colormap to use if branch_color is a numeric pandas.Series.
        branch_cmap_norm : matplotlib.colors.Normalize, optional
            Normalization to use if branch_color is a numeric pandas.Series.
        branch_alpha : float/dict/pandas.Series
            Add alpha channel to branch components, defaults to 0.9.

        Returns
        -------
        None

        """
        c_data = self._component_data[c_name]
        create_rgba_colors(
            df=c_data,
            color=branch_color,
            cmap=branch_cmap,
            cmap_norm=branch_cmap_norm,
            alpha=branch_alpha,
            target_col="rgba",
        )

    # Add branch layer
    def create_branch_layer(
        self,
        c_name: str,
    ) -> None:
        """Add a line layer of Pydeck type PathLayer to the interactive map.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer".

        Returns
        -------
        None

        """
        c_data = self._component_data[c_name]

        # Create PathLayer, use "path" column for get_path
        layer = pdk.Layer(
            "PathLayer",
            data=c_data.reset_index(),
            get_path="path",
            get_width="width_pdk",
            get_color="rgba",
            pickable=True,
            auto_highlight=True,
            parameters={
                "depthTest": False
            },  # To prevent z-fighting issues/flickering in 3D space
        )

        self._layers[c_name] = layer

    def init_arrow_data(
        self,
        c_name: str,
        branch_flow: float | dict | pd.Series = 0,
        arrow_size_factor: float = 1.5,
    ) -> None:
        """Initialize arrow data for a specific branch component type.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer
        branch_flow : float/dict/pandas.Series
            Flow values for the branch component, defaults to 0.
            If not 0, arrows will be drawn on the lines.
        arrow_size_factor : float, default 1.5
            Factor to scale the arrow size. If 0, no arrows will be drawn.

        Returns
        -------
        None

        """
        c_data = self._component_data[c_name]

        # Arrow layer
        branch_flow = _convert_to_series(branch_flow, c_data.index)
        branch_flow = branch_flow * 1e3  # Convert flow from km to m
        flows_are_zero = branch_flow.eq(0).all()

        if (
            not as_components(self._n, c_name).empty
            and not flows_are_zero
            and arrow_size_factor != 0
        ):
            c_data["flow"] = c_data.index.map(branch_flow)

    def create_arrows(
        self,
        c_name: str,
        arrow_size_factor: float = 1.5,
    ) -> None:
        """Create and scale arrows for a specific branch component type.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer".
        arrow_size_factor : float, default 1.5
            Factor to scale the arrow size. If 0, no arrows will be drawn.

        """
        c_data = self._component_data[c_name]

        def center_segment(path: list[list[float]]) -> tuple[list[float], list[float]]:
            """Return the two “center” points of a path for arrow placement."""
            n = len(path)
            mid_idx = n // 2
            return path[mid_idx - 1], path[mid_idx]

        # Precompute start/end points for arrows
        arrow_points = c_data["path"].apply(
            lambda path: center_segment(path) if len(path) > 2 else (path[0], path[-1])
        )

        # Apply _make_arrows using the precomputed points
        c_data["arrow"] = c_data.apply(
            lambda row: PydeckPlotter._make_arrows(
                row["flow_pdk"],
                arrow_points.loc[row.name][0],  # p0_geo
                arrow_points.loc[row.name][1],  # p1_geo
                arrow_size_factor,
            ),
            axis=1,
        )

    def create_arrow_color(
        self,
        c_name: str,
        arrow_color: str | dict | pd.Series | None = None,
        arrow_cmap: str | mcolors.Colormap = "viridis",
        arrow_cmap_norm: mcolors.Normalize | None = None,
        arrow_alpha: float | dict | pd.Series = 0.9,
    ) -> None:
        """Create arrow colors for a specific branch component type.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer
        arrow_color : str/dict/pandas.Series/None
            Colors for the arrows, defaults to None.
            If None, the branch color is used for the arrows.
        arrow_cmap : str/matplotlib.colors.Colormap, default 'viridis'
            Colormap to use if arrow_color is a numeric pandas.Series.
        arrow_cmap_norm : matplotlib.colors.Normalize, optional
            Normalization to use if arrow_color is a numeric pandas.Series.
        arrow_alpha : float/dict/pandas.Series
            Add alpha channel to arrows, defaults to 0.9.

        Returns
        -------
        None

        """
        c_data = self._component_data[c_name]
        create_rgba_colors(
            df=c_data,
            color=arrow_color,
            cmap=arrow_cmap,
            cmap_norm=arrow_cmap_norm,
            alpha=arrow_alpha,
            target_col="rgba_arrow",
            fallback_col="rgba",  # reuse branch colors if arrow_color is None
        )

    def create_arrow_layer(
        self,
        c_name: str,
    ) -> None:
        """Create arrow PolygonLayer for a specific branch component type.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer

        Returns
        -------
        None

        """
        c_data = self._component_data[c_name]

        layer = pdk.Layer(
            "PolygonLayer",
            data=c_data.reset_index(),
            get_polygon="arrow",
            get_fill_color="rgba_arrow",
            pickable=True,
            auto_highlight=True,
            parameters={
                "depthTest": False
            },  # To prevent z-fighting issues/flickering in 3D space
        )
        self._layers[f"{c_name}_arrows"] = layer

    def scale_branch_param(
        self,
        c_name: str,
        branch_param_name: str,
        branch_param: float | dict | pd.Series = 2,
        branch_param_factor: float = 1,
        branch_param_max: float = 10,  # km
        global_param_max: float | None = 1,  # km
        keep_algebraic_sign: bool = True,
        auto_scale: bool = False,
    ) -> None:
        """Scale branch params (width, flow) for a specific branch component type.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer
        branch_param_name : str
            Name of the branch parameter to be scaled, e.g. "width", "flow".
        branch_param : float/dict/pandas.Series/None
            Parameter of branch component in km. If None, width falls back to 1.5 km.
        branch_param_factor : float/None
            If None, branch params are auto-scaled to branch_param_max.
            If a float is provided, branch params are scaled by this factor.
        branch_param_max : float, default 10
            Maximum param of branch component in km when auto-scaling.
        global_param_max : float/None, default 1
            If multiple branch components are plotted, this ensures that the maximum params are scaled proportionally.
        keep_algebraic_sign : bool, default True
            If True, the algebraic sign of branch_param is readded to the scaled parameter.
        auto_scale : bool, default False
            Whether to auto-scale branch params to fit within branch_param_max.

        Returns
        -------
        None

        """
        c_data = self._component_data[c_name]

        branch_param_series: pd.Series = _convert_to_series(branch_param, c_data.index)

        if branch_param_series.eq(0).all():
            c_data[f"{branch_param_name}_pdk"] = 0
            return

        c_data[branch_param_name] = branch_param_series

        # Use absolute values
        sign = (
            branch_param_series.apply(np.sign)
            if keep_algebraic_sign
            else pd.Series(1, index=branch_param_series.index)
        )
        branch_param_deck = branch_param_series.abs()
        local_param_max = branch_param_deck.max()

        if auto_scale:
            # if global_param_max is None or global_param_max == 0:
            #     global_param_max = local_param_max

            scaling_factor = local_param_max / global_param_max * 1e3
            branch_param_deck = scaling_factor * scale_to_max_abs(
                branch_param_deck, branch_param_max
            )
        else:
            branch_param_deck = branch_param_deck * branch_param_factor * 1e3

        c_data[f"{branch_param_name}_pdk"] = branch_param_deck * sign

    def create_tooltips(
        self,
        c_name: str,
        columns: list | None = None,
    ) -> None:
        """Create tooltip HTML for a specific branch component type.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer".
        columns : list, default None
            List of branch columns to include. If None, all columns are used.

        Returns
        -------
        None

        """
        c_data = self._component_data[c_name]
        if columns is None:
            columns = list(c_data.columns)

        exclude_cols = [
            "path",
            "width_pdk",
            "arrow",
            "rgba",
            "rgba_arrow",
            "geometry",
            "size_pdk",
        ]
        columns = [
            col for col in columns if col not in exclude_cols and col in c_data.columns
        ]

        c_data["tooltip_html"] = df_to_html_table(
            c_data,
            columns=columns,
            rounding=2,
            value_align="left",
            max_header_length=30,
        )

    def add_branch_and_arrow_layer(
        self,
        branch_components: list | set | None = None,
        branch_width_factor: float = 1,
        branch_width_max: float = 10,  # km
        line_flow: float | dict | pd.Series = 0,
        line_color: str | dict | pd.Series = "rosybrown",
        line_cmap: str | mcolors.Colormap = "viridis",
        line_cmap_norm: mcolors.Normalize | None = None,
        line_alpha: float | dict | pd.Series = 0.9,
        line_width: float | dict | pd.Series = 2,
        line_columns: list | None = None,
        link_flow: float | dict | pd.Series = 0,
        link_color: str | dict | pd.Series = "darkorange",
        link_cmap: str | mcolors.Colormap = "viridis",
        link_cmap_norm: mcolors.Normalize | None = None,
        link_alpha: float | dict | pd.Series = 0.9,
        link_width: float | dict | pd.Series = 2,
        link_columns: list | None = None,
        transformer_flow: float | dict | pd.Series = 0,
        transformer_color: str | dict | pd.Series = "purple",
        transformer_cmap: str | mcolors.Colormap = "viridis",
        transformer_cmap_norm: mcolors.Normalize | None = None,
        transformer_alpha: float | dict | pd.Series = 0.9,
        transformer_width: float | dict | pd.Series = 2,
        transformer_columns: list | None = None,
        arrow_color: str | dict | pd.Series | None = None,
        arrow_cmap: str | mcolors.Colormap = "viridis",
        arrow_cmap_norm: mcolors.Normalize | None = None,
        arrow_alpha: float | dict | pd.Series = 0.9,
        arrow_size_factor: float = 1.5,
        geometry: bool = False,
        auto_scale: bool = False,
        tooltip: bool = True,
    ) -> None:
        """Add branch and arrow layers of Pydeck type PathLayer and PolygonLayer to the interactive map.

        Parameters
        ----------
        branch_components : list, set, optional, default ['Line', 'Link', 'Transformer']
            Branch components to be plotted.
        branch_width_factor : float, default 1.0
            Branch widths are scaled by this factor.
        branch_width_max : float, default 10
            Maximum width of branch component in km when auto-scaling.
        line_flow : float/dict/pandas.Series, default 0
            Series of line flows indexed by line names, defaults to 0. If 0, no arrows will be created.
            If a float is provided, it will be used as a constant flow for all lines.
        line_color : str/dict/pandas.Series
            Colors for the lines, defaults to 'rosybrown'.
        line_cmap : matplotlib.colors.Colormap/str
            If line_color are floats, this color map will assign the colors.
        line_cmap_norm : matplotlib.colors.Normalize
            The norm applied to the line_cmap.
        line_alpha : float/dict/pandas.Series
            Add alpha channel to lines, defaults to 0.9.
        line_width : float/dict/pandas.Series, default 2
            Widths of line component in km.
        link_flow : float/dict/pandas.Series, default 0
            Series of link flows indexed by link names, defaults to 0. If 0, no arrows will be created.
            If a float is provided, it will be used as a constant flow for all links.
        link_color : str/dict/pandas.Series
            Colors for the links, defaults to 'darkseagreen'.
        link_cmap : matplotlib.colors.Colormap/str, default 'viridis'
            If link_color are floats, this color map will assign the colors.
        link_cmap_norm : matplotlib.colors.Normalize|matplotlib.colors.*Norm
            The norm applied to the link_cmap.
        link_alpha : float/dict/pandas.Series
            Add alpha channel to links, defaults to 0.9.
        link_width : float/dict/pandas.Series, default 2
            Widths of link component in km.
        transformer_flow : float/dict/pandas.Series, default 0
            Series of transformer flows indexed by transformer names, defaults to 0. If 0, no arrows will be created.
            If a float is provided, it will be used as a constant flow for all transformers.
        transformer_color : str/dict/pandas.Series
            Colors for the transformers, defaults to 'orange'.
        transformer_cmap : matplotlib.colors.Colormap/str
            If transformer_color are floats, this color map will assign the colors.
        transformer_cmap_norm : matplotlib.colors.Normalize|matplotlib.colors.*Norm
            The norm applied to the transformer_cmap.
        transformer_alpha : float/dict/pandas.Series
            Add alpha channel to transformers, defaults to 0.9.
        transformer_width : float/dict/pandas.Series, default 2
            Widths of transformer in km.
        arrow_color : str/dict/pandas.Series/None
            Colors for the arrows, defaults to None.
            If None, the branch color is used for the arrows.
        arrow_cmap : str/matplotlib.colors.Colormap, default 'viridis'
            Colormap to use if arrow_color is a numeric pandas.Series.
        arrow_cmap_norm : matplotlib.colors.Normalize, optional
            Normalization to use if arrow_color is a numeric pandas.Series.
        arrow_alpha : float/dict/pandas.Series
            Add alpha channel to arrows, defaults to 0.9.
        arrow_size_factor : float, default 1.5
            Factor to scale the arrow size. If 0, no arrows will be drawn.
        geometry : bool, default False
            Whether to use the geometry column of the branch components.
        auto_scale : bool, default False
            Whether to auto-scale branch widths to fit within branch_width_max.
        line_columns : list, default None
            List of line columns to include. Specify additional columns to include in the tooltip.
        link_columns : list, default None
            List of link columns to include. Specify additional columns to include in the tooltip.
        transformer_columns : list, default None
            List of transformer columns to include. Specify additional columns to include in the tooltip.
        tooltip : bool, default True
            Whether to show a tooltip on hover.

        Returns
        -------
        None

        """
        n = self._n

        global_width_max = get_global_stat(
            elements=[line_width, link_width, transformer_width],
            stat="max",
            absolute=True,
        )  # If elements empty, global_width_max is None

        global_flow_max = get_global_stat(
            elements=[line_flow, link_flow, transformer_flow],
            stat="max",
            absolute=True,
        )  # If elements empty, global_flow_max is None

        for c in branch_components or n.branch_components:
            if c == "Line":
                branch_flow = line_flow
                branch_color = line_color
                branch_cmap = line_cmap
                branch_cmap_norm = line_cmap_norm
                branch_alpha = line_alpha
                branch_width = line_width
                branch_columns = line_columns
            elif c == "Link":
                branch_flow = link_flow
                branch_color = link_color
                branch_cmap = link_cmap
                branch_cmap_norm = link_cmap_norm
                branch_alpha = link_alpha
                branch_width = link_width
                branch_columns = link_columns
            elif c == "Transformer":
                branch_flow = transformer_flow
                branch_color = transformer_color
                branch_cmap = transformer_cmap
                branch_cmap_norm = transformer_cmap_norm
                branch_alpha = transformer_alpha
                branch_width = transformer_width
                branch_columns = transformer_columns

            if as_components(n, c).empty:
                continue

            # Branch lines
            self.init_branch_component_data(
                c_name=c,
                branch_columns=branch_columns,
            )
            self.create_branch_paths(
                c,
                geometry=geometry,
            )
            self.create_branch_color(
                c_name=c,
                branch_color=branch_color,
                branch_cmap=branch_cmap,
                branch_cmap_norm=branch_cmap_norm,
                branch_alpha=branch_alpha,
            )
            self.scale_branch_param(
                c_name=c,
                branch_param_name="width",
                branch_param=branch_width,
                branch_param_factor=branch_width_factor,
                branch_param_max=branch_width_max,  # km
                global_param_max=global_width_max,
                keep_algebraic_sign=False,
                auto_scale=auto_scale,
            )
            if tooltip:
                self.create_tooltips(
                    c_name=c,
                )
            self.create_branch_layer(
                c_name=c,
            )

            # Branch arrows
            self.init_arrow_data(
                c_name=c,
                branch_flow=branch_flow,
                arrow_size_factor=arrow_size_factor,
            )
            self.scale_branch_param(
                c_name=c,
                branch_param_name="flow",
                branch_param=branch_flow,
                branch_param_factor=branch_width_factor,
                branch_param_max=branch_width_max,  # km
                global_param_max=global_flow_max,
                keep_algebraic_sign=True,
                auto_scale=auto_scale,
            )
            self.create_arrows(
                c_name=c,
                arrow_size_factor=arrow_size_factor,
            )
            self.create_arrow_color(
                c_name=c,
                arrow_color=arrow_color,
                arrow_cmap=arrow_cmap,
                arrow_cmap_norm=arrow_cmap_norm,
                arrow_alpha=arrow_alpha,
            )
            if tooltip:
                self.create_tooltips(
                    c_name=c,
                    columns=["flow"],
                )
            self.create_arrow_layer(
                c_name=c,
            )

    def add_geomap_layer(
        self,
        geomap_alpha: float = 0.9,
        geomap_color: dict | None = None,
        geomap_resolution: Literal["110m", "50m", "10m"] = "50m",
    ) -> None:
        """Add a geomap layer of Pydeck type GeoJsonLayer to the interactive map.

        Parameters
        ----------
        geomap_alpha : float, default 0.9
            Alpha transparency for the geomap features.
        geomap_color : dict | None, default None
            Dictionary specifying colors for different geomap features. If None, default colors will be used: `{'land': 'whitesmoke', 'ocean': 'lightblue'}
        geomap_resolution : {'110m', '50m', '10m'}, default '50m'
            Resolution of the geomap features. One of '110m', '50m', or '10m'.

        Returns
        -------
        None

        """
        if not _is_cartopy_available():
            logger.warning(
                "Cartopy is not available. Falling back to non-geographic plotting."
            )
            return

        import cartopy.feature  # noqa: PLC0415

        if geomap_resolution not in ["110m", "50m", "10m"]:
            msg = "Resolution has to be one of '110m', '50m', or '10m'."
            raise ValueError(msg)

        if geomap_color is None:
            geomap_color = {
                "ocean": "lightblue",
                "land": "whitesmoke",
            }

        line_color = [100, 100, 100, 255]

        # Always render ocean first
        if "ocean" in geomap_color:
            features = feature_to_geojson(
                cartopy.feature.OCEAN.with_scale(geomap_resolution)
            )

            fill_color = to_rgba255(geomap_color["ocean"], geomap_alpha)
            layer = pdk.Layer(
                "PolygonLayer",
                data=features,
                get_polygon="geometry.coordinates",
                filled=True,
                stroked=True,
                get_fill_color=fill_color,
                get_line_color=line_color,
                line_width_min_pixels=1,
                auto_highlight=False,
                pickable=False,
            )
            self._layers["Geomap_ocean"] = layer

        # Then render land
        if "land" in geomap_color:
            features = shapefile_to_geojson(
                resolution=geomap_resolution,
                category="cultural",
                name="admin_0_countries",
                pole_buffer=1e-6,
            )
            fill_color = to_rgba255(geomap_color["land"], geomap_alpha)
            layer = pdk.Layer(
                "PolygonLayer",
                data=features,
                get_polygon="geometry.coordinates",
                filled=True,
                stroked=True,
                get_fill_color=fill_color,
                get_line_color=line_color,
                line_width_min_pixels=1,
                auto_highlight=False,
                pickable=False,
            )
            self._layers["Geomap_land"] = layer

    def deck(
        self,
        tooltip: bool = True,
    ) -> pdk.Deck:
        """Display the interactive map.

        Parameters
        ----------
        tooltip : bool, default True
            Whether to show a tooltip on hover.

        Returns
        -------
        pdk.Deck
            The Pydeck Deck object representing the interactive map.

        """
        layers = list(self._layers.values())

        tooltip_content: bool | dict[str, str | dict[str, str]]
        if not tooltip:
            tooltip_content = False
        else:
            tooltip_content = {"html": "{tooltip_html}", "style": self._tooltip_style}

        deck = pdk.Deck(
            layers=layers,
            map_style=self._map_style,
            tooltip=tooltip_content,
            initial_view_state=self.view_state,
            # set 3d view
        )
        return deck

    def build_layers(  # noqa: D417
        self,
        branch_components: list | set | None = None,
        branch_width_factor: float = 1,
        bus_size: float | dict | pd.Series = 25,
        bus_size_factor: float = 1.0,
        bus_split_circle: bool = False,
        bus_color: str | dict | pd.Series = "cadetblue",
        bus_cmap: str | mcolors.Colormap = "Reds",
        bus_cmap_norm: mcolors.Normalize | None = None,
        bus_alpha: float | dict | pd.Series = 0.9,
        line_flow: float | dict | pd.Series = 0,
        line_color: str | dict | pd.Series = "rosybrown",
        line_cmap: str | mcolors.Colormap = "viridis",
        line_cmap_norm: mcolors.Normalize | None = None,
        line_alpha: float | dict | pd.Series = 0.9,
        line_width: float | dict | pd.Series = 2,
        link_flow: float | dict | pd.Series = 0,
        link_color: str | dict | pd.Series = "darkseagreen",
        link_cmap: str | mcolors.Colormap = "viridis",
        link_cmap_norm: mcolors.Normalize | None = None,
        link_alpha: float | dict | pd.Series = 0.9,
        link_width: float | dict | pd.Series = 2,
        transformer_flow: float | dict | pd.Series = 0,
        transformer_color: str | dict | pd.Series = "orange",
        transformer_cmap: str | mcolors.Colormap = "viridis",
        transformer_cmap_norm: mcolors.Normalize | None = None,
        transformer_alpha: float | dict | pd.Series = 0.9,
        transformer_width: float | dict | pd.Series = 2,
        arrow_size_factor: float = 1.5,
        arrow_color: str | dict | pd.Series | None = None,
        arrow_cmap: str | mcolors.Colormap = "viridis",
        arrow_cmap_norm: mcolors.Normalize | None = None,
        arrow_alpha: float | dict | pd.Series = 0.9,
        tooltip: bool = True,
        auto_scale: bool = False,
        branch_width_max: float = 10,  # km
        bus_size_max: float = 10000,  # km²
        bus_columns: list | None = None,
        line_columns: list | None = None,
        link_columns: list | None = None,
        transformer_columns: list | None = None,
        geomap: bool = False,
        geomap_alpha: float = 0.9,
        geomap_color: dict | None = None,
        geomap_resolution: Literal["110m", "50m", "10m"] = "50m",
        geometry: bool = False,
    ) -> "PydeckPlotter":
        """Create an interactive map of the PyPSA network using Pydeck.

        Parameters
        ----------
        branch_width_factor : float, default 1.0
            Branch widths are scaled by this factor.
        bus_size : float/dict/pandas.Series
            Sizes of bus points in km² (corresponds to circle area), defaults to 25 km².
        bus_size_factor : float, default 1.0
            Bus sizes are scaled by this factor.
        bus_split_circle : bool, default False
            Draw half circles if bus_size is a pandas.Series with a Multiindex.
            If set to true, the upper half circle per bus then includes all positive values
            of the series, the lower half circle all negative values. Defaults to False.
        bus_color : str/dict/pandas.Series/None
            Colors for the buses, defaults to "cadetblue". If bus_size is a
            pandas.Series with a Multiindex, bus_color defaults to the
            n.c.carriers.static['color'] column.
        bus_cmap : mcolors.Colormap/str, default 'Reds'
            If bus_color are floats, this color map will assign the colors.
        bus_cmap_norm : mcolors.Normalize/None
            Normalization for bus_cmap, defaults to None.
        bus_alpha : float/dict/pandas.Series
            Add alpha channel to buses, defaults to 0.9.
        line_flow : float/dict/pandas.Series, default 0
            Series of line flows indexed by line names, defaults to 0. If 0, no arrows will be created.
            If a float is provided, it will be used as a constant flow for all lines.
        line_color : str/dict/pandas.Series
            Colors for the lines, defaults to 'rosybrown'.
        line_alpha : float/dict/pandas.Series
            Add alpha channel to lines, defaults to 0.9.
        line_width : float/dict/pandas.Series, default 2
            Widths of line component in km.
        link_flow : float/dict/pandas.Series, default 0
            Series of link flows indexed by link names, defaults to 0. If 0, no arrows will be created.
            If a float is provided, it will be used as a constant flow for all links.
        link_color : str/dict/pandas.Series
            Colors for the links, defaults to 'darkseagreen'.
        link_alpha : float/dict/pandas.Series
            Add alpha channel to links, defaults to 0.9.
        link_width : float/dict/pandas.Series, default 2
            Widths of link component in km.
        tooltip : bool, default True
            Whether to add a tooltip to the bus layer.

        Other Parameters
        ----------------
        branch_components : list, set, optional, default ['Line', 'Link', 'Transformer']
            Branch components to be plotted.
        branch_width_max : float, default 10
            Maximum width of branch component in km when `auto_scale` is True.
        bus_size_max : float, default 10000
            Maximum area size of bus component in km² when `auto_scale` is True.
        line_cmap : mcolors.Colormap/str, default 'viridis'
            If line_color are floats, this color map will assign the colors.
        line_cmap_norm : mcolors.Normalize
            The norm applied to the line_cmap.
        link_cmap : mcolors.Colormap/str, default 'viridis'
            If link_color are floats, this color map will assign the colors.
        link_cmap_norm : mcolors.Normalize|matplotlib.colors.*Norm
            The norm applied to the link_cmap.
        transformer_flow : float/dict/pandas.Series, default 0
            Series of transformer flows indexed by transformer names, defaults to 0. If 0, no arrows will be created.
            If a float is provided, it will be used as a constant flow for all transformers.
        transformer_color : str/dict/pandas.Series
            Colors for the transformers, defaults to 'orange'.
        transformer_cmap : mcolors.Colormap/str, default 'viridis'
            If transformer_color are floats, this color map will assign the colors.
        transformer_cmap_norm : matplotlib.colors.Normalize|matplotlib.colors.*Norm
            The norm applied to the transformer_cmap.
        transformer_alpha : float/dict/pandas.Series
            Add alpha channel to transformers, defaults to 0.9.
        transformer_width : float/dict/pandas.Series, default 2
            Widths of transformer in km.
        arrow_size_factor : float, default 1.5
            Multiplier on branch flows to scale the arrow size.
        arrow_color : str/dict/pandas.Series | None, default None
            Colors for the arrows. If not specified, defaults to the same colors as the respective branch component.
        arrow_cmap : str/matplotlib.colors.Colormap, default 'viridis'
            Colormap to use if arrow_color is a numeric pandas.Series.
        arrow_cmap_norm : matplotlib.colors.Normalize, optional
            Normalization to use if arrow_color is a numeric pandas.Series.
        arrow_alpha : float/dict/pandas.Series, default 0.9
            Add alpha channel to arrows, defaults to 0.9.
        bus_columns : list, default None
            List of bus columns to include.
            Specify additional columns to include in the tooltip.
        line_columns : list, default None
            List of line columns to include. If None, only the bus0 and bus1 columns are used.
            Specify additional columns to include in the tooltip.
        link_columns : list, default None
            List of link columns to include. If None, only the bus0 and bus1 columns are used.
            Specify additional columns to include in the tooltip.
        transformer_columns : list, default None
            List of transformer columns to include. If None, only the bus0 and bus1 columns are used.
            Specify additional columns to include in the tooltip.
        geomap : bool, default False
            Whether to add a geomap layer to the plot.
        geomap_alpha : float, default 0.9
            Alpha transparency for the geomap features.
        geomap_color : dict | None, default None
            Dictionary specifying colors for different geomap features. If None, default colors will be used: `{'land': 'whitesmoke', 'ocean': 'lightblue'}
        geomap_resolution : {'110m', '50m', '10m'}, default '50m'
            Resolution of the geomap features. One of '110m', '50m', or '10m'.
        geometry : bool, default False
            Whether to use the geometry column of the branch components.

        Returns
        -------
        PydeckPlotter
            The PydeckPlotter instance with the created layers.

        """
        n = self._n

        if geomap:
            self.add_geomap_layer(
                geomap_alpha=geomap_alpha,
                geomap_color=geomap_color,
                geomap_resolution=geomap_resolution,
            )

        # Branch layers
        if branch_components is None:
            branch_components = n.branch_components

        self.add_branch_and_arrow_layer(
            branch_components=branch_components,
            branch_width_factor=branch_width_factor,
            branch_width_max=branch_width_max,
            line_flow=line_flow,
            line_color=line_color,
            line_cmap=line_cmap,
            line_cmap_norm=line_cmap_norm,
            line_alpha=line_alpha,
            line_width=line_width,
            line_columns=line_columns,
            link_flow=link_flow,
            link_color=link_color,
            link_cmap=link_cmap,
            link_cmap_norm=link_cmap_norm,
            link_alpha=link_alpha,
            link_width=link_width,
            link_columns=link_columns,
            transformer_flow=transformer_flow,
            transformer_color=transformer_color,
            transformer_cmap=transformer_cmap,
            transformer_cmap_norm=transformer_cmap_norm,
            transformer_alpha=transformer_alpha,
            transformer_width=transformer_width,
            transformer_columns=transformer_columns,
            arrow_color=arrow_color,
            arrow_cmap=arrow_cmap,
            arrow_cmap_norm=arrow_cmap_norm,
            arrow_alpha=arrow_alpha,
            arrow_size_factor=arrow_size_factor,
            geometry=geometry,
            auto_scale=auto_scale,
            tooltip=tooltip,
        )

        # Bus layer
        if hasattr(bus_size, "index") and isinstance(bus_size.index, pd.MultiIndex):
            self.add_pie_chart_layer(
                bus_size=bus_size,
                bus_size_factor=bus_size_factor,
                bus_size_max=bus_size_max,
                bus_split_circle=bus_split_circle,
                bus_alpha=bus_alpha,
                bus_columns=bus_columns,
                points_per_radian=5,
                auto_scale=auto_scale,
                tooltip=tooltip,
            )
        else:
            self.add_bus_layer(
                bus_size=bus_size,
                bus_size_factor=bus_size_factor,
                bus_size_max=bus_size_max,
                bus_color=bus_color,
                bus_cmap=bus_cmap,
                bus_cmap_norm=bus_cmap_norm,
                bus_alpha=bus_alpha,
                bus_columns=bus_columns,
                auto_scale=auto_scale,
                tooltip=tooltip,
            )

        return self


# TODO: fix typing differences between PydeckPlotter.build_layers and explore function
@wraps(
    PydeckPlotter.build_layers,
    assigned=("__doc__", "__annotations__", "__type_params__"),
)
def explore(  # noqa: D103
    n: "Network",
    map_style: str = "road",
    view_state: dict | pdk.ViewState | None = None,
    layouter: Callable | None = None,
    jitter: float | None = None,
    **kwargs: Any,
) -> pdk.Deck:
    """Create an interactive map of the PyPSA network using Pydeck.

    <!-- md:badge-version v1.0.0 -->

    Returns
    -------
    pdk.Deck
        The Pydeck object representing the interactive map.

    """
    plotter = PydeckPlotter(
        n,
        map_style=map_style,
        view_state=view_state,
        layouter=layouter,
        jitter=jitter,
    )

    # Optional tooltip_kwargs
    tooltip_kwargs = kwargs.pop("tooltip_kwargs", {})
    plotter._tooltip_style = set_tooltip_style(**tooltip_kwargs)

    plotter.build_layers(**kwargs)

    tooltip = kwargs.get("tooltip", True)

    return plotter.deck(tooltip=tooltip)
