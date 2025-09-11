"""Plot the network interactively using plotly and folium."""

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pltly
import pydeck as pdk
import pyproj

from pypsa.common import _convert_to_series
from pypsa.plot.maps.common import (
    apply_layouter,
    as_branch_series,
    calculate_angle,
    calculate_midpoint,
    flip_polygon,
    meters_to_lonlat,
    rotate_polygon,
    scale_polygon_by_width,
    to_rgba255,
)

if TYPE_CHECKING:
    from pypsa import Network


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


def iplot(
    n: "Network",
    fig: dict | None = None,
    bus_colors: str | dict | pd.Series = "cadetblue",
    bus_alpha: float = 1,
    bus_sizes: float | pd.Series = 10,
    bus_cmap: str | mcolors.Colormap | None = None,
    bus_colorbar: dict | None = None,
    bus_text: pd.Series | None = None,
    line_colors: str | pd.Series = "rosybrown",
    link_colors: str | pd.Series = "darkseagreen",
    transformer_colors: str | pd.Series = "orange",
    line_widths: float | pd.Series = 3,
    link_widths: float | pd.Series = 3,
    transformer_widths: float | pd.Series = 3,
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
    bus_colors : dict/pandas.Series
        Colors for the buses, defaults to "cadetblue". If bus_sizes is a
        pandas.Series with a Multiindex, bus_colors defaults to the
        n.c.carriers.static['color'] column.
    bus_alpha : float
        Add alpha channel to buses, defaults to 1.
    bus_sizes : float/pandas.Series
        Sizes of bus points, defaults to 10.
    bus_cmap : mcolors.Colormap/str
        If bus_colors are floats, this color map will assign the colors
    bus_colorbar : dict
        Plotly colorbar, e.g. {'title' : 'my colorbar'}
    bus_text : pandas.Series
        Text for each bus, defaults to bus names
    line_colors : str/pandas.Series
        Colors for the lines, defaults to 'rosybrown'.
    link_colors : str/pandas.Series
        Colors for the links, defaults to 'darkseagreen'.
    transformer_colors : str/pandas.Series
        Colors for the transfomer, defaults to 'orange'.
    line_widths : dict/pandas.Series
        Widths of lines, defaults to 1.5
    link_widths : dict/pandas.Series
        Widths of links, defaults to 1.5
    transformer_widths : dict/pandas.Series
        Widths of transformer, defaults to 1.5
    line_text : pandas.Series
        Text for lines, defaults to line names.
    link_text : pandas.Series
        Text for links, defaults to link names.
    transformer_text : pandas.Series
        Text for transformers, defaults to transformer names.
    layouter : networkx.drawing.layout function, default None
        Layouting function from `networkx <https://networkx.github.io/>`_ which
        overrules coordinates given in ``n.buses[['x', 'y']]``. See
        `list <https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout>`_
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
        to a mapbox layout, the argument ``mapbox_token`` must be a valid Mapbox
        API access token.

        Valid open layouts are:
            open-street-map, white-bg, carto-positron, carto-darkmatter,
            stamen-terrain, stamen-toner, stamen-watercolor

        Valid mapbox layouts are:
            basic, streets, outdoors, light, dark, satellite, satellite-streets

    mapbox_token : string
        Mapbox API access token. Obtain from https://www.mapbox.com.
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
        "marker": {"color": bus_colors, "size": bus_sizes},
    }

    if bus_cmap is not None:
        bus_trace["marker"]["colorscale"] = bus_cmap

    if bus_colorbar is not None:
        bus_trace["marker"]["colorbar"] = bus_colorbar

    if branch_components is None:
        branch_components = n.branch_components

    branch_colors = {
        "Line": line_colors,
        "Link": link_colors,
        "Transformer": transformer_colors,
    }
    branch_widths = {
        "Line": line_widths,
        "Link": link_widths,
        "Transformer": transformer_widths,
    }
    branch_text = {
        "Line": line_text,
        "Link": link_text,
        "Transformer": transformer_text,
    }

    shapes = []
    shape_traces = []

    for c in n.iterate_components(branch_components):
        b_widths = as_branch_series(branch_widths[c.name], "width", c.name, n)
        b_colors = as_branch_series(branch_colors[c.name], "color", c.name, n)
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
                "https://www.mapbox.com/, style which do not require a token "
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
    BUS_COLORS = "cadetblue"
    VALID_MAP_STYLES = {
        "light": pdk.map_styles.LIGHT,
        "dark": pdk.map_styles.DARK,
        "road": pdk.map_styles.ROAD,
        "dark_no_labels": pdk.map_styles.DARK_NO_LABELS,
        "light_no_labels": pdk.map_styles.LIGHT_NO_LABELS,
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
        **kwargs: Any,
    ) -> None:
        """Initialize the PydeckPlotter.

        Parameters
        ----------
        n : Network
            The PyPSA network to plot.
        map_style : str
            Map style to use for the plot. One of 'light', 'dark', 'road', 'dark_no_labels', and 'light_no_labels'.
        kwargs : dict, optional
            Additional keyword arguments to configure the map, such as `view_state` for initial view settings.

        """
        self._n: Network = n
        self._map_style: str = self._init_map_style(map_style)
        self._view_state: pdk.ViewState = self._init_view_state(
            view_state=kwargs.get("view_state"),
        )
        self._layers: dict[str, pdk.Layer] = {}
        self._tooltip_columns: list[str] = ["value", "coords"]
        self._tooltip: dict | bool = False

    def _init_map_style(self, map_style: str) -> str:
        """Set the initial map style for the interactive map."""
        if map_style not in self.VALID_MAP_STYLES:
            msg = (
                f"Invalid map style '{map_style}'.\n"
                f"Must be one of: {', '.join(self.VALID_MAP_STYLES)}."
            )
            raise ValueError(msg)
        return map_style

    @property
    def map_style(self) -> str:
        """Get the current map style."""
        return self._map_style

    @property
    def layers(self) -> dict[str, pdk.Layer]:
        """Get the layers of the interactive map."""
        return self._layers

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
            "longitude": self._n.buses.x.mean(),
            "latitude": self._n.buses.y.mean(),
            "zoom": 4,  # Default zoom level
            "min_zoom": None,
            "max_zoom": None,
            "pitch": 0,  # Default pitch
            "bearing": 0,  # Default bearing
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
        p0_geo: float,
        p1_geo: float,
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
        default_columns: list[str],
        extra_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Prepare data for a specific component type.

        Parameters
        ----------
        component : str
            Name of the component type, e.g. "Bus", "Line", "Link", "Transformer".
        default_columns : list of str
            List of default columns to include for the component.
        extra_columns : list of str, optional
            Additional columns to include for the component.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the prepared data for the component.

        """
        df = self._n.static(component)
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
            self._tooltip_columns = list(set(self._tooltip_columns + valid_columns))

        df = df[layer_columns].copy()
        df.index.name = "name"

        if extra_columns:
            # Round all numeric columns to 3 decimal places. Only columns in valid_columns
            numeric_cols = [
                col for col in valid_columns if pd.api.types.is_numeric_dtype(df[col])
            ]
            for col in numeric_cols:
                df[col] = df[col].round(3)

        return df

    # Layer functions
    def add_bus_layer(
        self,
        bus_sizes: float | dict | pd.Series = 25000000,
        bus_colors: str | dict | pd.Series = "cadetblue",
        bus_alpha: float | dict | pd.Series = 0.9,
        bus_columns: list | None = None,
    ) -> None:
        """Add a bus layer of Pydeck type ScatterplotLayer to the interactive map.

        Parameters
        ----------
        bus_sizes : float/dict/pandas.Series
            Sizes of bus points in radius² (meters²), defaults to 25000000.
        bus_colors : str/dict/pandas.Series
            Colors for the buses, defaults to 'cadetblue'.
        bus_alpha : float/dict/pandas.Series
            Add alpha channel to buses, defaults to 0.9.
        bus_columns : list, default None
            List of bus columns to include. If None, only the bus index and x, y coordinates are used.
            Specify additional columns to include in the tooltip.

        Returns
        -------
        None

        """
        # Check if columns exist and only keep the ones that also exist in the network
        bus_data = self.prepare_component_data(
            "Bus",
            default_columns=["x", "y"],
            extra_columns=bus_columns,
        )

        # For default tooltip
        bus_data["value"] = _convert_to_series(bus_sizes, bus_data.index).round(3)
        bus_data["coords"] = bus_data[["x", "y"]].apply(
            lambda row: f"({row['x']:.3f}, {row['y']:.3f})", axis=1
        )

        # Map bus sizes
        bus_data["radius"] = bus_data["value"] ** 0.5

        # Convert colors to RGBA list
        colors = _convert_to_series(bus_colors, bus_data.index)
        alphas = _convert_to_series(bus_alpha, bus_data.index)

        bus_data["rgba"] = [
            to_rgba255(c, a) for c, a in zip(colors, alphas, strict=False)
        ]

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=bus_data.reset_index(),
            get_position=["x", "y"],
            get_color="rgba",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
            parameters={
                "depthTest": False
            },  # To prevent z-fighting issues/flickering in 3D space
        )

        # Append the bus layer to the layers property
        self._layers["Bus"] = layer

    @staticmethod
    def _pie_slice_vertices(
        p_center: tuple[float, float],
        radius: float,
        start_angle: float,
        end_angle: float,
        points_per_radian: int = 10,
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
        points_per_radian : int, default 10
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
        p_center: tuple[float, float],
        radius_m: float,
        values: np.ndarray,
        colors: list[list[int]],
        labels: list[str],
        points_per_radian: int = 10,
        flip_y: bool = False,
        bus_split_circles: bool = False,
    ) -> list[dict]:
        """Create pie chart polygons with metadata using numpy.

        Parameters
        ----------
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
        points_per_radian : int, default 10
            Number of points per radian for pie chart resolution.
        flip_y : bool, default False
            Flip the pie chart vertically. Useful for negative values in bus_split_circles mode.
        bus_split_circles : bool, default False
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

        circ = np.pi if bus_split_circles else 2 * np.pi
        flip_y_factor = -1 if flip_y else 1
        rotate_by_quarter = 0 if bus_split_circles else np.pi / 2

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
                "name": label,
                "value": val,
            }
            for val, color, label, start, delta in zip(
                values, colors, labels, start_angles, angles, strict=False
            )
        ]
        return polygons

    def add_pie_chart_layer(
        self,
        bus_sizes: pd.Series,
        bus_split_circles: bool = False,
        bus_alpha: float | dict | pd.Series = 0.9,
        points_per_radian: int = 10,
    ) -> None:
        """Add a bus layer of Pydeck type ScatterplotLayer to the interactive map.

        Parameters
        ----------
        bus_sizes : float/dict/pandas.Series
            Sizes of bus points in radius² (meters²), defaults to 25000000.
        bus_split_circles : bool, default False
            Draw half circles if bus_sizes is a pandas.Series with a Multiindex.
            If set to true, the upper half circle per bus then includes all positive values
            of the series, the lower half circle all negative values. Defaults to False.
        bus_alpha : float/dict/pandas.Series
            Add alpha channel to buses, defaults to 0.9.
        points_per_radian : int, default 10
            Number of points per radian for pie chart resolution.

        Returns
        -------
        None

        """
        EPS = 1e-6  # Small epsilon to avoid numerical issues
        bus_data = self.prepare_component_data(
            "Bus",
            default_columns=["x", "y"],
        )
        bus_sizes = bus_sizes.drop(bus_sizes[abs(bus_sizes) < EPS].index)
        bus_sizes = bus_sizes.unstack(level=1, fill_value=0)

        alphas = _convert_to_series(bus_alpha, bus_sizes.index)
        carrier_colors = self._n.c.carriers.static["color"]
        carrier_rgba = {
            bus: {c: to_rgba255(col, alphas[bus]) for c, col in carrier_colors.items()}
            for bus in bus_sizes.index
        }

        polygons = []
        # Convert to NumPy arrays for speed-up
        bus_indices = bus_sizes.index.to_numpy()
        bus_cols = bus_sizes.columns.to_numpy()
        bus_values = bus_sizes.to_numpy()
        bus_coords = bus_data.loc[bus_indices, ["x", "y"]].to_numpy()

        for i, bus in enumerate(bus_indices):
            values = bus_values[i]
            x, y = bus_coords[i]

            if values.sum() <= EPS:
                continue

            if bus_split_circles and np.any(values < 0):
                pos_mask = values > 0
                if np.any(pos_mask):
                    poly_pos = PydeckPlotter._make_pie(
                        p_center=(x, y),
                        radius_m=(values[pos_mask].sum()) ** 0.5,
                        values=values[pos_mask].round(3),
                        colors=[carrier_rgba[bus][c] for c in bus_cols[pos_mask]],
                        labels=[f"{bus} - {c}" for c in bus_cols[pos_mask]],
                        points_per_radian=points_per_radian,
                        flip_y=False,
                        bus_split_circles=True,
                    )
                    polygons.extend(poly_pos)

                neg_mask = values < 0
                if np.any(neg_mask):
                    poly_neg = PydeckPlotter._make_pie(
                        p_center=(x, y),
                        radius_m=(-values[neg_mask].sum()) ** 0.5,
                        values=-values[neg_mask].round(3),
                        colors=[carrier_rgba[bus][c] for c in bus_cols[neg_mask]],
                        labels=[f"{bus} - {c}" for c in bus_cols[neg_mask]],
                        points_per_radian=points_per_radian,
                        flip_y=True,
                        bus_split_circles=True,
                    )
                    polygons.extend(poly_neg)
            else:
                mask = values > 0
                if not np.any(mask):
                    continue
                poly_pos = PydeckPlotter._make_pie(
                    p_center=(x, y),
                    radius_m=(values[mask].sum()) ** 0.5,
                    values=values[mask].round(3),
                    colors=[carrier_rgba[bus][c] for c in bus_cols[mask]],
                    labels=[f"{bus} - {c}" for c in bus_cols[mask]],
                    points_per_radian=points_per_radian,
                    flip_y=False,
                    bus_split_circles=False,
                )
                polygons.extend(poly_pos)

        layer = pdk.Layer(
            "PolygonLayer",
            data=polygons,
            get_polygon="polygon",
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            parameters={
                "depthTest": False  # To prevent z-fighting issues/flickering in 3D space
            },
        )

        self._layers["PieChart"] = layer

    # Add branch layer
    def add_branch_layer(
        self,
        c_name: str,
        branch_flow: float | dict | pd.Series = 0,
        branch_colors: str | dict | pd.Series = "rosybrown",
        branch_alpha: float | dict | pd.Series = 0.9,
        branch_widths: float | dict | pd.Series = 1500,
        branch_columns: list | None = None,
        arrow_size_factor: float = 1.5,
        arrow_colors: str | dict | pd.Series = "black",
        arrow_alpha: float | dict | pd.Series = 0.9,
    ) -> None:
        """Add a line layer of Pydeck type PathLayer to the interactive map.

        Parameters
        ----------
        c_name : str
            Name of the branch component type, e.g. "Line", "Link", "Transformer".
        branch_flow : float/dict/pandas.Series
            Flow values for the branch component, defaults to 0.
            If not 0, arrows will be drawn on the lines.
        branch_colors : str/dict/pandas.Series
            Colors for the branch component, defaults to 'rosybrown'.
        branch_alpha : float/dict/pandas.Series
            Add alpha channel to branch components, defaults to 0.9.
        branch_widths : float/dict/pandas.Series
            Widths of branch component in meters, defaults to 1500.
        branch_columns : list, default None
            List of branch columns to include. If None, only the bus0 and bus1 columns are used.
            Specify additional columns to include in the tooltip.
        arrow_size_factor : float, default 1.5
            Factor to scale the arrow size. If 0, no arrows will be drawn.
        arrow_colors : str/dict/pandas.Series
            Colors for the arrows, defaults to 'black'.
        arrow_alpha : float/dict/pandas.Series
            Add alpha channel to arrows, defaults to 0.9.

        Returns
        -------
        None

        """
        if self._n.static(c_name).empty:
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
        missing_buses = c_data.loc[
            ~c_data["bus0"].isin(self._n.buses.index)
            | ~c_data["bus1"].isin(self._n.buses.index)
        ]
        if not missing_buses.empty:
            # TODO: Store missing buses and branches with missing buses in property if needed later
            msg = (
                f"Found {len(missing_buses)} row(s) in '{c_name}' with missing buses. "
                "These rows will be dropped."
            )
            logger.warning(msg)
            c_data = c_data.drop(missing_buses.index)

            # If no data remains after dropping missing buses, return early
            if c_data.empty:
                return

        # Build path column as list of [lon, lat] pairs for each line
        c_data["path"] = c_data.apply(
            lambda row: [
                [
                    self._n.buses.loc[row["bus0"], "x"],
                    self._n.buses.loc[row["bus0"], "y"],
                ],
                [
                    self._n.buses.loc[row["bus1"], "x"],
                    self._n.buses.loc[row["bus1"], "y"],
                ],
            ],
            axis=1,
        )

        # Convert colors to RGBA list
        colors = _convert_to_series(branch_colors, c_data.index)
        alphas = _convert_to_series(branch_alpha, c_data.index)

        c_data["rgba"] = [
            to_rgba255(c, a) for c, a in zip(colors, alphas, strict=False)
        ]

        # Map line widths
        c_data["width"] = _convert_to_series(branch_widths, c_data.index)
        c_data["width"] = c_data["width"].abs()

        # For default tooltip
        arrow_sym = "&#x2B95;"
        c_data["value"] = c_data["width"].round(3)
        c_data["coords"] = c_data.apply(
            lambda row: (
                f"({row['path'][0][0]:.3f}, {row['path'][0][1]:.3f}) {arrow_sym} "
                f"({row['path'][1][0]:.3f}, {row['path'][1][1]:.3f})"
            ),
            axis=1,
        )

        # Create PathLayer, use "path" column for get_path
        layer = pdk.Layer(
            "PathLayer",
            data=c_data.reset_index(),
            get_path="path",
            get_width="width",
            get_color="rgba",
            pickable=True,
            auto_highlight=True,
            parameters={
                "depthTest": False
            },  # To prevent z-fighting issues/flickering in 3D space
        )

        self._layers[c_name] = layer

        # Arrow layer
        branch_flow = _convert_to_series(branch_flow, c_data.index)
        flows_are_zero = (branch_flow == 0).all()

        if (
            not self._n.static(c_name).empty
            and not flows_are_zero
            and arrow_size_factor != 0
        ):
            if arrow_colors is None:
                arrow_colors = branch_colors

            # Map branch_flows to c_data
            c_data["flow"] = c_data.index.map(branch_flow)
            c_data["arrow"] = c_data.apply(
                lambda row: PydeckPlotter._make_arrows(
                    flow=row["flow"],
                    p0_geo=row["path"][0],
                    p1_geo=row["path"][-1],
                    arrow_size_factor=arrow_size_factor,
                ),
                axis=1,
            )

            colors = _convert_to_series(arrow_colors, c_data.index)
            alphas = _convert_to_series(arrow_alpha, c_data.index)

            c_data["rgba"] = [
                to_rgba255(c, a) for c, a in zip(colors, alphas, strict=False)
            ]

            layer = pdk.Layer(
                "PolygonLayer",
                data=c_data.reset_index(),
                get_polygon="arrow",
                get_fill_color="rgba",
                pickable=True,  # Disable tooltips for arrow heads
                auto_highlight=True,
                parameters={
                    "depthTest": False
                },  # To prevent z-fighting issues/flickering in 3D space
            )
            self._layers[f"{c_name}_arrows"] = layer

    # TODO: Find a way to hide empty tooltip columns. Note, tooltips per layer are not supported by Pydeck.
    def add_tooltip(self) -> None:
        """Add a tooltip to the interactive map."""
        tooltip_html = "<b>{name}</b><br/>"
        for col in self._tooltip_columns:
            tooltip_html += f"<b>{col}:</b> {{{col}}}<br/>"

        self._tooltip = {
            "html": tooltip_html,
            "style": {
                "backgroundColor": "black",
                "color": "white",
                "fontFamily": "Arial",
                "fontSize": "12px",
            },
        }

    def deck(self) -> pdk.Deck:
        """Display the interactive map.

        Returns
        -------
        pdk.Deck
            The Pydeck Deck object representing the interactive map.

        """
        layers = list(self._layers.values())

        deck = pdk.Deck(
            layers=layers,
            map_style=self._map_style,
            tooltip=self._tooltip,
            initial_view_state=self.view_state,
            # set 3d view
        )
        return deck


def explore(
    n: "Network",
    bus_sizes: float | dict | pd.Series = 25000000,
    bus_split_circles: bool = False,
    bus_colors: str | dict | pd.Series = "cadetblue",
    bus_alpha: float | dict | pd.Series = 0.9,
    line_flow: float | dict | pd.Series = 0,
    line_colors: str | dict | pd.Series = "rosybrown",
    line_alpha: float | dict | pd.Series = 0.9,
    line_widths: float | dict | pd.Series = 1500,
    link_flow: float | dict | pd.Series = 0,
    link_colors: str | dict | pd.Series = "darkseagreen",
    link_alpha: float | dict | pd.Series = 0.9,
    link_widths: float | dict | pd.Series = 1500,
    transformer_flow: float | dict | pd.Series = 0,
    transformer_colors: str | dict | pd.Series = "orange",
    transformer_alpha: float | dict | pd.Series = 0.9,
    transformer_widths: float | dict | pd.Series = 1500,
    arrow_size_factor: float = 1.5,
    arrow_colors: str | dict | pd.Series | None = None,
    arrow_alpha: float | dict | pd.Series = 0.9,
    map_style: str = "road",
    tooltip: bool = True,
    bus_columns: list | None = None,
    line_columns: list | None = None,
    link_columns: list | None = None,
    transformer_columns: list | None = None,
    **kwargs: Any,
) -> pdk.Deck:
    """Create an interactive map of the PyPSA network using Pydeck.

    Parameters
    ----------
    n : Network
        The PyPSA network to plot.
    bus_sizes : float/dict/pandas.Series
        Sizes of bus points in radius² (meters²), defaults to 25000000.
    bus_split_circles : bool, default False
        Draw half circles if bus_sizes is a pandas.Series with a Multiindex.
        If set to true, the upper half circle per bus then includes all positive values
        of the series, the lower half circle all negative values. Defaults to False.
    bus_colors : str/dict/pandas.Series
        Colors for the buses, defaults to "cadetblue".
    bus_alpha : float/dict/pandas.Series
        Add alpha channel to buses, defaults to 0.9.
    line_flow : float/dict/pandas.Series, default 0
        Series of line flows indexed by line names, defaults to 0. If 0, no arrows will be created.
        If a float is provided, it will be used as a constant flow for all lines.
    line_colors : str/dict/pandas.Series
        Colors for the lines, defaults to 'rosybrown'.
    line_alpha : float/dict/pandas.Series
        Add alpha channel to lines, defaults to 0.9.
    line_widths : float/dict/pandas.Series
        Widths of lines in meters, defaults to 1500.
    link_flow : float/dict/pandas.Series, default 0
        Series of link flows indexed by link names, defaults to 0. If 0, no arrows will be created.
        If a float is provided, it will be used as a constant flow for all links.
    link_colors : str/dict/pandas.Series
        Colors for the links, defaults to 'darkseagreen'.
    link_alpha : float/dict/pandas.Series
        Add alpha channel to links, defaults to 0.9.
    link_widths : float/dict/pandas.Series
        Widths of links in meters, defaults to 1500.
    transformer_flow : float/dict/pandas.Series, default 0
        Series of transformer flows indexed by transformer names, defaults to 0. If 0, no arrows will be created.
        If a float is provided, it will be used as a constant flow for all transformers.
    transformer_colors : str/dict/pandas.Series
        Colors for the transformers, defaults to 'orange'.
    transformer_alpha : float/dict/pandas.Series
        Add alpha channel to transformers, defaults to 0.9.
    transformer_widths : float/dict/pandas.Series
        Widths of transformers in meters, defaults to 1500.
    arrow_size_factor : float, default 1.5
        Factor to scale the arrow size in relation to line_flow. A value of 1 denotes a multiplier of 1 times line_width. If 0, no arrows will be created.
    arrow_colors : str/dict/pandas.Series | None, default None
        Colors for the arrows. If not specified, defaults to the same colors as the respective branch component.
    arrow_alpha : float/dict/pandas.Series, default 0.9
        Add alpha channel to arrows, defaults to 0.9.
    map_style : str, default 'road'
        Map style to use for the plot. One of 'light', 'dark', 'road', 'dark_no_labels', and 'light_no_labels'.
    tooltip : bool, default True
        Whether to add a tooltip to the bus layer.
    bus_columns : list, default None
        List of bus columns to include. If None, only the bus index and x, y coordinates are used.
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
    **kwargs : dict, optional
        Additional keyword arguments to pass to the PydeckPlotter. For example
        `view_state` to set the initial view state of the map, see pdk.ViewState for details.

    Returns
    -------
    pdk.Deck
        The interactive map as a Pydeck Deck object.

    """
    plotter = PydeckPlotter(n, map_style=map_style, **kwargs)

    # Branch layers
    for c in n.iterate_components(n.branch_components):
        if c.name == "Line":
            branch_colors = line_colors
            branch_alpha = line_alpha
            branch_widths = line_widths
            branch_columns = line_columns
            branch_flow = line_flow
        elif c.name == "Link":
            branch_colors = link_colors
            branch_alpha = link_alpha
            branch_widths = link_widths
            branch_columns = link_columns
            branch_flow = link_flow
        elif c.name == "Transformer":
            branch_colors = transformer_colors
            branch_alpha = transformer_alpha
            branch_widths = transformer_widths
            branch_columns = transformer_columns
            branch_flow = transformer_flow

        if not plotter._n.static(c.name).empty:
            plotter.add_branch_layer(
                c_name=c.name,
                branch_colors=branch_colors,
                branch_alpha=branch_alpha,
                branch_widths=branch_widths,
                branch_columns=branch_columns,
                branch_flow=branch_flow,
                arrow_size_factor=arrow_size_factor,
                arrow_colors=arrow_colors,
                arrow_alpha=arrow_alpha,
            )

    # Bus layer
    if hasattr(bus_sizes, "index") and isinstance(bus_sizes.index, pd.MultiIndex):
        plotter.add_pie_chart_layer(
            bus_sizes=bus_sizes,
            bus_split_circles=bus_split_circles,
            bus_alpha=bus_alpha,
            points_per_radian=10,
        )
    else:
        plotter.add_bus_layer(
            bus_sizes=bus_sizes,
            bus_colors=bus_colors,
            bus_alpha=bus_alpha,
            bus_columns=bus_columns,
        )

    # Tooltip
    if tooltip:
        plotter.add_tooltip()

    return plotter.deck()
