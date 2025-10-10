# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import sys

import pydeck as pdk
import pytest

import pypsa.plot.maps.interactive as interactive
from pypsa.plot.maps.common import _is_cartopy_available

cartopy_present = _is_cartopy_available()


# Test explore() function
def test_plot_explore_alias_for_explore(ac_dc_network):
    """Test that n.plot.explore is an alias for n.explore()."""
    n = ac_dc_network

    result1 = n.plot.explore()
    result2 = n.explore()

    assert type(result1) is type(result2)


def test_explore_parameters(ac_dc_network):
    n = ac_dc_network

    # Custom map style
    deck = n.plot.explore(map_style="dark")
    assert deck.map_style == "dark" or pdk.Deck(map_style="dark").map_style

    # Custom view state
    view_state = pdk.ViewState(latitude=0, longitude=0, zoom=5)
    deck = n.plot.explore(view_state=view_state)
    assert deck.initial_view_state.latitude == 0
    assert deck.initial_view_state.zoom == 5

    # With jitter
    deck = n.plot.explore(jitter=0.1)
    assert isinstance(deck, pdk.Deck)


# Test subplotter properties
def test_pdkplotter_piecharts(ac_dc_network):
    n = ac_dc_network

    pies = n.statistics.installed_capacity(
        groupby=["bus", "carrier"],
    )
    pies.index = pies.index.droplevel(0)

    plotter = interactive.PydeckPlotter(n, map_style="light")
    plotter.build_layers(bus_size=pies)
    plotter.deck()


def test_pdk_plotter_piecharts_bus_split_circle(ac_dc_network):
    n = ac_dc_network
    pies = n.statistics.installed_capacity(
        groupby=["bus", "carrier"],
    )
    pies.index = pies.index.droplevel(0)

    load = -n.loads_t.p_set.sum(axis=0)
    load.index.name = "bus"

    pies = pies.unstack().fillna(0)
    pies["load"] = load
    pies.fillna(0, inplace=True)
    pies = pies.stack()

    plotter = interactive.PydeckPlotter(n, map_style="light")
    plotter.build_layers(bus_size=pies, bus_split_circle=True)
    plotter.deck()


def test_pdk_plotter_auto_scale(ac_dc_network):
    n = ac_dc_network
    plotter = interactive.PydeckPlotter(n, map_style="light")

    # Test autoscale with default parameters
    plotter.build_layers(auto_scale=True)
    plotter.deck()


def test_pdk_plotter_flows(ac_dc_network):
    n = ac_dc_network
    plotter = interactive.PydeckPlotter(n, map_style="light")

    line_flow = n.c.lines.static.s_nom / 1e3
    link_flow = n.c.links.static.p_nom / 1e3

    # Test flows with default parameters
    plotter.build_layers(
        line_flow=line_flow,
        link_flow=link_flow,
    )
    plotter.deck()


def test_pdk_color_cmaps(ac_dc_network):
    n = ac_dc_network
    plotter = interactive.PydeckPlotter(n, map_style="light")

    # Test with different colormaps
    bus_color = n.c.buses.static.v_nom
    line_color = n.c.lines.static.s_nom
    link_color = n.c.links.static.p_nom

    plotter.build_layers(
        bus_color=bus_color,
        line_color=line_color,
        link_color=link_color,
    )
    plotter.deck()


def test_pdk_tooltips(ac_dc_network):
    n = ac_dc_network
    plotter = interactive.PydeckPlotter(n, map_style="light")

    # Test with custom tooltip columns
    plotter.build_layers(
        bus_columns=["name", "v_nom"],
        line_columns=["name", "s_nom"],
        link_columns=["name", "p_nom"],
    )
    plotter.deck(tooltip=True)
    plotter.deck(tooltip=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Cartopy issues on macos.")
def test_geomap_warning(ac_dc_network, caplog):
    n = ac_dc_network
    with caplog.at_level("WARNING"):
        n.plot.explore(geomap=True)


def test_pdkplotter_properties(ac_dc_network):
    n = ac_dc_network
    plotter = interactive.PydeckPlotter(n, map_style="dark")

    assert plotter.map_style == "dark"
    assert isinstance(plotter._tooltip_style, dict)
    assert isinstance(plotter._layers, dict)


def test_init_map_style_valid():
    class Dummy:
        VALID_MAP_STYLES = {"light": "something", "dark": "something"}
        _init_map_style = interactive.PydeckPlotter._init_map_style

    d = Dummy()
    assert d._init_map_style("light") == "light"
    assert d._init_map_style("dark") == "dark"


def test_init_map_style_invalid():
    class Dummy:
        VALID_MAP_STYLES = {"light": "something"}
        _init_map_style = interactive.PydeckPlotter._init_map_style

    d = Dummy()
    with pytest.raises(ValueError, match="Invalid"):
        d._init_map_style("foo")


def test_extra_columns(ac_dc_network, caplog):
    n = ac_dc_network
    plotter = interactive.PydeckPlotter(n, map_style="light")

    with caplog.at_level("WARNING"):
        plotter.build_layers(line_columns=["bus0", "doesnotexist"])

    # Check that a warning was logged
    assert "not found" in caplog.text

    # Validate columns
    df = plotter._component_data["Line"]
    assert "bus0" in df.columns
    assert "doesnotexist" not in df.columns


@pytest.mark.skipif(sys.platform != "linux", reason="Cartopy issues on macos.")
@pytest.mark.skipif(not cartopy_present, reason="Cartopy not installed")
def test_geomap_params(ac_dc_network):
    n = ac_dc_network
    plotter = interactive.PydeckPlotter(n, map_style="light")

    # Test geomap with default parameters
    plotter.build_layers(
        geomap=True,
        geomap_alpha=1,
        geomap_resolution="110m",
    )
    plotter.deck()
