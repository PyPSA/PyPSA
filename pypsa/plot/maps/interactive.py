"""Plot the network interactively using plotly and folium."""

import importlib
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pydeck as pdk
from shapely import linestrings

from pypsa.common import _convert_to_series
from pypsa.plot.maps.common import apply_layouter, as_branch_series, to_rgba255
from pypsa.plot.maps.static import MapPlotter

if TYPE_CHECKING:
    from pypsa import Network

pltly_present = True
try:
    import plotly.graph_objects as go
    import plotly.offline as pltly
except ImportError:
    pltly_present = False


pdk.settings.pydeck_offline = True  # Embed JavaScript and CSS into the HTML file

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
        n.carriers['color'] column.
    bus_alpha : float
        Adds alpha channel to buses, defaults to 1.
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
        bus_text = "Bus " + n.buses.index
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
            lon = (n.buses.x.min() + n.buses.x.max()) / 2
            lat = (n.buses.y.min() + n.buses.y.max()) / 2
            mapbox_parameters["center"] = {"lat": lat, "lon": lon}

        if "zoom" not in mapbox_parameters:
            mapbox_parameters["zoom"] = 2

        fig["layout"]["mapbox"] = mapbox_parameters
    else:
        fig["layout"]["shapes"] = shapes

    if iplot:
        if not pltly_present:
            logger.warning("Plotly is not present, so interactive plotting won't work.")
        else:
            pltly.iplot(fig)

    return fig


class PydeckPlotter:
    """Class to create and manage an interactive pydeck map for a PyPSA network."""

    # Class-level constants
    BUS_COLORS = "cadetblue"
    VALID_MAP_STYLES = [
        pdk.map_styles.LIGHT,
        pdk.map_styles.DARK,
        pdk.map_styles.ROAD,
        pdk.map_styles.SATELLITE,
        pdk.map_styles.DARK_NO_LABELS,
        pdk.map_styles.LIGHT_NO_LABELS,
    ]

    def __init__(
        self,
        n: "Network",
        map_style: str,
    ) -> None:
        """Initialize the PydeckPlotter.

        Parameters
        ----------
        n : Network
            The PyPSA network to plot.
        map_style : str
            Map style to use for the plot. One of 'light', 'dark', 'road', 'satellite', 'dark_no_labels', and 'light_no_labels'.
        """
        self._n = n
        self._map_style: str = self._init_map_style(map_style)
        self._view_state: pdk.ViewState = self._init_view_state()
        self._layers: dict[str, pdk.Layer] = {}
        self._component_columns: list[str] = []
        self._tooltip: dict | bool = False
        self._mapplotter = MapPlotter(n)  # Embed static map plotting functionality

    def _init_map_style(self, map_style: str) -> None:
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

    def _init_view_state(self) -> pdk.ViewState:
        """Compute the initial view state based on network bus coordinates."""
        center_lon = self._n.buses.x.mean()
        center_lat = self._n.buses.y.mean()
        zoom = 5  # Default zoom level
        return pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom,
            pitch=0,  # Default pitch
            bearing=0,  # Default bearing
        )

    @property
    def view_state(self) -> pdk.ViewState:
        """Get the current view state of the map."""
        return self._view_state

    def add_bus_layer(
        self,
        bus_columns: list | None = None,
        bus_sizes: float | dict | pd.Series = 5000,
        bus_colors: str | dict | pd.Series = "cadetblue",
        bus_alpha: float | dict | pd.Series = 0.5,
    ) -> None:
        """
        Adds a bus layer of Pydeck type ScatterplotLayer to the interactive map.
        
        Parameters
        ----------
        bus_columns : list, default None
            List of bus columns to include. If None, only the bus index and x, y coordinates are used.
            Specify additional columns to include in the tooltip.
        bus_sizes : float/dict/pandas.Series
            Sizes of bus points, defaults to 1e-2. If a multiindexed Series is passed,
            the function will draw pies for each bus (first index level) with
            segments of different color (second index level). Such a Series is ob-
            tained by e.g. n.generators.groupby(['bus', 'carrier']).p_nom.sum().
        bus_colors : str/dict/pandas.Series
            Colors for the buses, defaults to "cadetblue". If bus_sizes is a
            pandas.Series with a Multiindex, bus_colors defaults to the
            n.carriers['color'] column.
        bus_alpha : float/dict/pandas.Series
            Adds alpha channel to buses, defaults to 0.5.

        """
        # Check if columns exist and only keep the ones that also exist in the network
        bus_layer_columns = ["x", "y"]  # Default columns for bus coordinates
        if bus_columns:
            missing_bus_columns = [
                col for col in bus_columns if col not in self._n.buses.columns
            ]
            valid_bus_columns = [
                col for col in bus_columns if col in self._n.buses.columns
            ]
            if missing_bus_columns:
                msg = (
                    f"Columns {missing_bus_columns} not found in buses. "
                    f"Using only valid columns: {valid_bus_columns}."
                )
                logger.warning(msg)
            bus_layer_columns = bus_layer_columns + valid_bus_columns
            self._component_columns.extend(valid_bus_columns)

        bus_data = self._n.buses[bus_layer_columns].copy()
        bus_data.index.name = "name"

        # Map bus sizes
        bus_data["radius"] = _convert_to_series(bus_sizes, bus_data.index)
        get_radius = "radius"

        # Map bus colors
        bus_data["color"] = _convert_to_series(bus_colors, bus_data.index)
        bus_data["alpha"] = _convert_to_series(bus_alpha, bus_data.index)

        bus_data["rgba"] = bus_data.apply(
            lambda row: to_rgba255(row["color"], row["alpha"]), axis=1
        )
        get_color = "rgba"

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=bus_data.reset_index(),
            get_position=["x", "y"],
            get_color=get_color,
            get_radius=get_radius,
            pickable=True,
            auto_highlight=True,
        )

        # Append the bus layer to the layers property
        self._layers["buses"] = layer

    def add_tooltip(self) -> None:
        """Add a tooltip to the interactive map."""
        tooltip_html = "<b>{name}</b><br/>"
        for col in self._component_columns:
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
        """Display the interactive map."""
        deck = pdk.Deck(
            layers=list(self._layers.values()),
            map_style=self._map_style,
            tooltip=self._tooltip,
            initial_view_state=self.view_state,
        )
        return deck


def explore(
    n: "Network",
    bus_columns: list | None = None,
    bus_sizes: float | dict | pd.Series = 5000,
    bus_colors: str | dict | pd.Series = "cadetblue",
    bus_alpha: float | dict | pd.Series = 0.5,
    map_style: str = "dark",
    tooltip: bool = True,
) -> pdk.Deck:
    """Create an interactive map of the PyPSA network using Pydeck.

    Parameters
    ----------
    n : Network
        The PyPSA network to plot.
    bus_columns : list, default None
        List of bus columns to include. If None, only the bus index and x, y coordinates are used.
        Specify additional columns to include in the tooltip.
    bus_sizes : float/dict/pandas.Series
        Sizes of bus points, defaults to 1e-2. If a multiindexed Series is passed,
        the function will draw pies for each bus (first index level) with
        segments of different color (second index level). Such a Series is ob-
        tained by e.g. n.generators.groupby(['bus', 'carrier']).p_nom.sum().
    bus_colors : str/dict/pandas.Series
        Colors for the buses, defaults to "cadetblue". If bus_sizes is a
        pandas.Series with a Multiindex, bus_colors defaults to the
        n.carriers['color'] column.
    bus_alpha : float/dict/pandas.Series
        Adds alpha channel to buses, defaults to 0.5.
    map_style : str
        Map style to use for the plot. One of 'light', 'dark', 'road', 'satellite', 'dark_no_labels', and 'light_no_labels'.
    tooltip : bool, default True
        Whether to add a tooltip to the bus layer.

    Returns
    -------
    pdk.Deck
        The interactive map as a Pydeck Deck object.

    """
    plotter = PydeckPlotter(n, map_style=map_style)

    plotter.add_bus_layer(
        bus_columns=bus_columns,
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        bus_alpha=bus_alpha,
    )

    if tooltip:
        plotter.add_tooltip()

    return plotter.deck()
