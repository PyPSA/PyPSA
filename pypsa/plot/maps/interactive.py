"""Plot the network interactively using plotly and pydeck (deck.gl)."""

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from pypsa.plot.maps.common import apply_layouter, as_branch_series

if TYPE_CHECKING:
    from pypsa import Network

pltly_present = True
try:
    import plotly.graph_objects as go
    import plotly.offline as pltly
except ImportError:
    pltly_present = False

pdk_present = True
try:
    import pydeck as pdk
except ImportError:
    pdk_present = False


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

    # Plot branches:
    if isinstance(line_widths, pd.Series) and isinstance(
        line_widths.index, pd.MultiIndex
    ):
        msg = (
            "Index of argument 'line_widths' is a Multiindex, "
            "this is not support since pypsa v0.17 and will be removed in v1.0. "
            "Set differing widths with arguments 'line_widths', "
            "'link_widths' and 'transformer_widths'."
        )
        raise DeprecationWarning(msg)
    if isinstance(line_colors, pd.Series) and isinstance(
        line_colors.index, pd.MultiIndex
    ):
        msg = (
            "Index of argument 'line_colors' is a Multiindex, "
            "this is not support since pypsa v0.17. and will be removed in v1.0. "
            "Set differing colors with arguments 'line_colors', "
            "'link_colors' and 'transformer_colors'."
        )
        raise DeprecationWarning(msg)

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


def component_data_from_network(
    n: "Network",
    component: str,
    columns: list[str] | None = None,
    filter_attrs: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Create a DataFrame from a PyPSA network component (e.g., 'buses', 'lines', etc.) with specified columns, including the component's index as the first column.

    Parameters
    ----------
    n : pypsa.Network
        The network containing the component.
    component : str
        The name of the component, e.g., 'buses', 'lines', 'generators'.
    columns : list of str, optional
        List of column names to include from the component DataFrame.
    filter_attrs : dict of str to list
        Dictionary where keys are attribute names and values are lists of allowed values.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the selected columns, with the index as the first column.

    """
    comp_data = n.static(component)
    attrs = n.component_attrs[component]

    if filter_attrs:
        for key, values in filter_attrs.items():
            if key not in attrs.index:
                msg = f"Filter column '{key}' not found in component '{component}'"
                raise ValueError(msg)
            if values is None or not isinstance(values, list):
                continue
            comp_data = comp_data[comp_data[key].isin(values)]

    if component == "Bus":
        # Ensure 'x' and 'y' coordinates are included for buses
        if "y" not in columns:
            columns.insert(0, "y")
        if "x" not in columns:
            columns.insert(0, "x")

    return comp_data[columns]


def load_pdk_buses_layer(
    n: "Network",
    bus_carriers: list[str] | None = None,
    bus_columns: list[str] | None = None,
) -> tuple["pdk.Layer", dict]:
    """Load a pydeck layer for the the buses of a PyPSA network.

    Parameters
    ----------
    n : pypsa.Network
        The network containing the buses.
    bus_carriers : list of str, optional
        If provided, filter buses by these carriers. If None, all bus carriers are included.
    bus_columns : list of str, optional
        List of columns from n.buses to display in the tooltip. If None, no columns are included.

    """
    # Filter and validate columns
    existing_columns = []
    if bus_columns:
        existing_columns = [col for col in bus_columns if col in n.buses.columns]
        missing = set(bus_columns) - set(existing_columns)
        if missing:
            msg = f"Columns not found in buses: {sorted(missing)}"
            logger.warning(msg)

    bus_data = component_data_from_network(
        n,
        "Bus",
        columns=existing_columns,
        filter_attrs={"carrier": bus_carriers},
    ).reset_index()

    # Dynamically create HTML tooltip string
    tooltip_html = "<b>{name}</b><br>" + "<br>".join(
        f"<b>{col}:</b> {{{col}}}"
        for col in existing_columns
        if col not in ["name", "x", "y"]
    )

    buses_tooltip = {
        "html": tooltip_html,
        "style": {
            "backgroundColor": "black",
            "color": "white",
            "fontFamily": "sans-serif",
            "fontSize": "10px",
        },
    }

    buses_layer = pdk.Layer(
        "ScatterplotLayer",
        data=bus_data,
        get_position="[x, y]",
        get_color="[255, 0, 0, 100]",
        get_radius=5000,
        pickable=True,
    )

    return buses_layer, buses_tooltip


def explore(
    n: "Network",
    components: set[str] | None = None,
    bus_carriers: list[str] | None = None,
    bus_columns: list[str] | None = None,
    map_style: str = "light",
) -> Any | None:
    """Create an interactive map from a PyPSA network using pydeck (built on deck.gl).

    Parameters
    ----------
    n : pypsa.Network
        The network to plot.
    components : set[str] | None, default None
        Set of components to include in the map. If no components are specified, an empty map is returned.
    bus_carriers : List[str] | None, default None
        If provided, filter buses by these carriers. If None, all bus carriers are included.
    bus_columns : List[str], default ["country"]
        List of columns from n.buses to display in the tooltip
    map_style : str, default "light"
        Map style to use for the plot. One of 'light', 'dark', 'road', 'satellite', 'dark_no_labels', and 'light_no_labels'.

    """
    if not pdk_present:
        logger.warning("pydeck is not present, so n.explore() won't work.")
        return None

    if components is None:
        return pdk.Deck(
            layers=[],
            map_style=map_style,
        )

    # Initialize layers list
    pdk_layers = []

    # Add bus layer if "Bus" is in components
    if "Bus" in components:
        buses_layer, buses_tooltip = load_pdk_buses_layer(
            n,
            bus_carriers=bus_carriers,
            bus_columns=bus_columns,
        )
        pdk_layers.append(buses_layer)

    return pdk.Deck(layers=pdk_layers, tooltip=buses_tooltip, map_style=map_style)
