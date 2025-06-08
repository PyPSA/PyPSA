"""Plot the network interactively using plotly and folium."""

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely import linestrings

from pypsa.constants import DEFAULT_EPSG
from pypsa.plot.maps.common import apply_layouter, as_branch_series

if TYPE_CHECKING:
    from pypsa import Network

pltly_present = True
try:
    import plotly.graph_objects as go
    import plotly.offline as pltly
except ImportError:
    pltly_present = False


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


def explore(
    n: "Network",
    crs: int | str | None = None,
    tooltip: bool = True,
    popup: bool = True,
    tiles: str = "OpenStreetMap",
    components: set[str] | None = None,
) -> Any | None:  # TODO: returns a FoliunMap or None
    """Create an interactive map displaying PyPSA network components using geopandas exlore() and folium.

    This function generates a Folium map showing buses, lines, links, and transformers from the provided network object.

    Parameters
    ----------
    n : PyPSA.Network object
        containing components `buses`, `links`, `lines`, `transformers`, `generators`, `loads`, and `storage_units`.
    crs : str, optional. If not specified, it will check whether `n.crs` exists and use it, else it will default to "EPSG:4326".
        Coordinate Reference System for the GeoDataFrames.
    tooltip : bool, optional, default=True
        Whether to include tooltips (on hover) for the features.
    popup : bool, optional, default=True
        Whether to include popups (on click) for the features.
    tiles : str, optional, default="OpenStreetMap"
        The tileset to use for the map. Options include "OpenStreetMap", "CartoDB Positron", and "CartoDB dark_matter".
    components : list-like, optional, default=None
        The components to plot. Default includes "Bus", "Line", "Link", "Transformer".

    Returns
    -------
    folium.Map
        A Folium map object with the PyPSA.Network components plotted.

    """
    try:
        import mapclassify  # noqa: F401
        from folium import Element, LayerControl, Map, TileLayer
    except ImportError:
        logger.warning(
            "folium and mapclassify need to be installed to use `n.explore()`."
        )
        return None

    if n.crs and crs is None:
        crs = n.crs
    else:
        crs = DEFAULT_EPSG

    if components is None:
        components = {"Bus", "Line", "Transformer", "Link"}

    # Map related settings
    bus_colors = mcolors.CSS4_COLORS["cadetblue"]
    line_colors = mcolors.CSS4_COLORS["rosybrown"]
    link_colors = mcolors.CSS4_COLORS["darkseagreen"]
    transformer_colors = mcolors.CSS4_COLORS["orange"]
    generator_colors = mcolors.CSS4_COLORS["purple"]
    load_colors = mcolors.CSS4_COLORS["red"]
    storage_unit_colors = mcolors.CSS4_COLORS["black"]

    # Initialize the map
    map = Map(tiles=None)

    # Add map title
    map_title = "PyPSA Network" + (f": {n.name}" if n.name else "")
    map.get_root().html.add_child(
        Element(
            f"<h4 style='position:absolute;z-index:100000;left:1vw;bottom:5px'>{map_title}</h4>"
        )
    )

    # Add tile layer legend entries
    TileLayer(tiles, name=tiles).add_to(map)

    components_possible = [
        "Bus",
        "Line",
        "Link",
        "Transformer",
        "Generator",
        "Load",
        "StorageUnit",
    ]
    components_present = []

    if not n.transformers.empty and "Transformer" in components:
        x1 = n.transformers.bus0.map(n.buses.x)
        y1 = n.transformers.bus0.map(n.buses.y)
        x2 = n.transformers.bus1.map(n.buses.x)
        y2 = n.transformers.bus1.map(n.buses.y)
        valid_rows = ~(x1.isna() | y1.isna() | x2.isna() | y2.isna())

        if num_invalid := sum(~valid_rows):
            logger.info(
                "Omitting %d transformers due to missing coordinates", num_invalid
            )

        gdf_transformers = gpd.GeoDataFrame(
            n.transformers[valid_rows],
            geometry=linestrings(
                np.stack(
                    [
                        (x1[valid_rows], y1[valid_rows]),
                        (x2[valid_rows], y2[valid_rows]),
                    ],
                    axis=1,
                ).T
            ),
            crs=crs,
        )

        gdf_transformers[gdf_transformers.is_valid].explore(
            m=map,
            color=transformer_colors,
            tooltip=tooltip,
            popup=popup,
            name="Transformers",
        )
        components_present.append("Transformer")

    if not n.lines.empty and "Line" in components:
        x1 = n.lines.bus0.map(n.buses.x)
        y1 = n.lines.bus0.map(n.buses.y)
        x2 = n.lines.bus1.map(n.buses.x)
        y2 = n.lines.bus1.map(n.buses.y)
        valid_rows = ~(x1.isna() | y1.isna() | x2.isna() | y2.isna())

        if num_invalid := sum(~valid_rows):
            logger.info("Omitting %d lines due to missing coordinates.", num_invalid)

        gdf_lines = gpd.GeoDataFrame(
            n.lines[valid_rows],
            geometry=linestrings(
                np.stack(
                    [
                        (x1[valid_rows], y1[valid_rows]),
                        (x2[valid_rows], y2[valid_rows]),
                    ],
                    axis=1,
                ).T
            ),
            crs=crs,
        )

        gdf_lines[gdf_lines.is_valid].explore(
            m=map, color=line_colors, tooltip=tooltip, popup=popup, name="Lines"
        )
        components_present.append("Line")

    if not n.links.empty and "Link" in components:
        x1 = n.links.bus0.map(n.buses.x)
        y1 = n.links.bus0.map(n.buses.y)
        x2 = n.links.bus1.map(n.buses.x)
        y2 = n.links.bus1.map(n.buses.y)
        valid_rows = ~(x1.isna() | y1.isna() | x2.isna() | y2.isna())

        if num_invalid := sum(~valid_rows):
            logger.info("Omitting %d links due to missing coordinates.", num_invalid)

        gdf_links = gpd.GeoDataFrame(
            n.links[valid_rows],
            geometry=linestrings(
                np.stack(
                    [
                        (x1[valid_rows], y1[valid_rows]),
                        (x2[valid_rows], y2[valid_rows]),
                    ],
                    axis=1,
                ).T
            ),
            crs=DEFAULT_EPSG,
        )

        gdf_links[gdf_links.is_valid].explore(
            m=map, color=link_colors, tooltip=tooltip, popup=popup, name="Links"
        )
        components_present.append("Link")

    if not n.buses.empty and "Bus" in components:
        gdf_buses = gpd.GeoDataFrame(
            n.buses, geometry=gpd.points_from_xy(n.buses.x, n.buses.y), crs=crs
        )

        gdf_buses[gdf_buses.is_valid].explore(
            m=map,
            color=bus_colors,
            tooltip=tooltip,
            popup=popup,
            name="Buses",
            marker_kwds={"radius": 4},
        )
        components_present.append("Bus")

    if not n.generators.empty and "Generator" in components:
        gdf_generators = gpd.GeoDataFrame(
            n.generators,
            geometry=gpd.points_from_xy(
                n.generators.bus.map(n.buses.x), n.generators.bus.map(n.buses.y)
            ),
            crs=crs,
        )

        gdf_generators[gdf_generators.is_valid].explore(
            m=map,
            color=generator_colors,
            tooltip=tooltip,
            popup=popup,
            name="Generators",
            marker_kwds={"radius": 2.5},
        )
        components_present.append("Generator")

    if not n.loads.empty and "Load" in components:
        loads = n.loads.copy()
        loads["p_set_sum"] = n.loads_t.p_set.sum(axis=0).round(1)
        gdf_loads = gpd.GeoDataFrame(
            loads,
            geometry=gpd.points_from_xy(
                loads.bus.map(n.buses.x), loads.bus.map(n.buses.y)
            ),
            crs=crs,
        )

        gdf_loads[gdf_loads.is_valid].explore(
            m=map,
            color=load_colors,
            tooltip=tooltip,
            popup=popup,
            name="Loads",
            marker_kwds={"radius": 1.5},
        )
        components_present.append("Load")

    if not n.storage_units.empty and "StorageUnit" in components:
        gdf_storage_units = gpd.GeoDataFrame(
            n.storage_units,
            geometry=gpd.points_from_xy(
                n.storage_units.bus.map(n.buses.x), n.storage_units.bus.map(n.buses.y)
            ),
            crs=crs,
        )

        gdf_storage_units[gdf_storage_units.is_valid].explore(
            m=map,
            color=storage_unit_colors,
            tooltip=tooltip,
            popup=popup,
            name="Storage Units",
            marker_kwds={"radius": 1},
        )
        components_present.append("StorageUnit")

    if len(components_present) > 0:
        logger.info(
            "Components rendered on the map: %s",
            ", ".join(sorted(components_present)),
        )
    if len(set(components) - set(components_present)) > 0:
        logger.info(
            "Components omitted as they are missing or not selected: %s",
            ", ".join(sorted(set(components_possible) - set(components_present))),
        )

    # Set the default view to the bounds of the elements in the map
    map.fit_bounds(map.get_bounds())

    # Add a Layer Control to toggle layers on and off
    LayerControl().add_to(map)

    return map
