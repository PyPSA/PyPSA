import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pypsa.plot.maps.static import (
    add_legend_circles,
    add_legend_lines,
    add_legend_patches,
    add_legend_semicircles,
)
from pypsa.statistics import get_transmission_branches, groupers

try:
    import cartopy.crs as ccrs

    cartopy_present = True
except ImportError:
    cartopy_present = False


try:
    import folium  # noqa: F401
    import mapclassify  # noqa: F401

    explore_deps_present = True
except ImportError:
    explore_deps_present = False


@pytest.mark.parametrize("margin", [None, 0.1])
@pytest.mark.parametrize("jitter", [None, 1])
@pytest.mark.skipif(os.name == "nt", reason="tcl_findLibrary on Windows")
def test_plot_standard_params_wo_geomap(ac_dc_network, margin, jitter):
    n = ac_dc_network
    n.plot.map(geomap=False, margin=margin, jitter=jitter)
    plt.close()


@pytest.mark.skipif(not cartopy_present, reason="Cartopy not installed")
@pytest.mark.parametrize("margin", [None, 0.1])
@pytest.mark.parametrize("jitter", [None, 1])
def test_plot_standard_params_w_geomap(ac_dc_network, margin, jitter):
    n = ac_dc_network
    n.plot.map(geomap=True, margin=margin, jitter=jitter)
    plt.close()


def test_plot_on_axis_wo_geomap(ac_dc_network):
    n = ac_dc_network
    fig, ax = plt.subplots()
    n.plot.map(ax=ax, geomap=False)
    plt.close()


@pytest.mark.skipif(not cartopy_present, reason="Cartopy not installed")
def test_plot_on_axis_w_geomap(ac_dc_network):
    n = ac_dc_network
    fig, ax = plt.subplots()

    with pytest.raises(ValueError):
        n.plot.map(ax=ax, geomap=True)
    plt.close()


def test_plot_bus_circles(ac_dc_network):
    n = ac_dc_network

    bus_sizes = n.c.generators.static.groupby(["bus", "carrier"]).p_nom.mean()
    bus_sizes[:] = 1
    bus_colors = n.c.carriers.static.color
    n.plot.map(bus_sizes=bus_sizes, bus_colors=bus_colors, geomap=False)
    plt.close()

    # Retrieving the colors from carriers also should work
    n.c.carriers.static["color"] = bus_colors
    n.plot.map(bus_sizes=bus_sizes)
    plt.close()

    # Retrieving the colors from carriers also should work
    n.c.carriers.static["color"] = bus_colors
    n.plot.map(bus_sizes=bus_sizes)
    plt.close()


def test_plot_split_circles(ac_dc_network):
    n = ac_dc_network

    gen_sizes = n.c.generators.static.groupby(["bus", "carrier"]).p_nom.sum()
    gen_sizes[:] = 500
    n.loads.carrier = "load"
    load_sizes = (
        -n.c.loads.dynamic.p_set.mean()
        .groupby([n.c.loads.static.bus, n.c.loads.static.carrier])
        .max()
    )
    bus_sizes = pd.concat((gen_sizes, load_sizes)) / 1e3
    bus_colors = n.c.carriers.static.color
    n.plot.map(
        bus_sizes=bus_sizes, bus_colors=bus_colors, bus_split_circles=True, geomap=False
    )
    plt.close()


def test_plot_with_bus_cmap(ac_dc_network):
    n = ac_dc_network

    buses = n.c.buses.static.index
    rng = np.random.default_rng()  # Create a random number generator
    colors = pd.Series(rng.random(size=len(buses)), buses)
    n.plot.map(bus_colors=colors, bus_cmap="coolwarm", geomap=False)
    plt.close()


def test_plot_with_line_cmap(ac_dc_network):
    n = ac_dc_network

    lines = n.c.lines.static.index
    rng = np.random.default_rng()  # Create a random number generator
    colors = pd.Series(rng.random(size=len(lines)), lines)
    n.plot.map(line_colors=colors, line_cmap="coolwarm", geomap=False)
    plt.close()


def test_plot_alpha(ac_dc_network):
    n = ac_dc_network

    bus_sizes = n.c.generators.static.groupby(["bus", "carrier"]).p_nom.mean()
    bus_sizes[:] = 1
    bus_colors = n.c.carriers.static.color
    n.plot.map(
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        geomap=False,
        bus_alpha=0.5,
        line_alpha=0.5,
        link_alpha=0.5,
    )
    plt.close()

    # Retrieving the colors from carriers also should work
    n.c.carriers.static["color"] = bus_colors
    n.plot.map(bus_sizes=bus_sizes)
    plt.close()


def test_plot_line_subset(ac_dc_network):
    n = ac_dc_network

    lines = n.c.lines.static.index[:2]
    rng = np.random.default_rng()  # Create a random number generator
    colors = pd.Series(rng.random(size=len(lines)), lines)
    n.plot.map(line_colors=colors, line_cmap="coolwarm", geomap=False)
    plt.close()


def test_plot_bus_subset(ac_dc_network):
    n = ac_dc_network

    buses = n.c.buses.static.index[:2]
    rng = np.random.default_rng()  # Create a random number generator
    colors = pd.Series(rng.random(size=len(buses)), buses)
    n.plot.map(bus_colors=colors, bus_cmap="coolwarm", geomap=False)
    plt.close()

    bus_sizes = n.c.generators.static.groupby(["bus", "carrier"]).p_nom.mean()[:3]
    bus_sizes[:] = 1
    bus_colors = n.c.carriers.static.color
    n.plot.map(
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        geomap=False,
        bus_alpha=0.5,
        line_alpha=0.5,
        link_alpha=0.5,
    )
    plt.close()


def test_plot_from_statistics(ac_dc_network):
    n = ac_dc_network
    bus_carrier = "AC"

    grouper = groupers["bus", "carrier"]
    bus_sizes = n.statistics.installed_capacity(
        bus_carrier=bus_carrier, groupby=grouper, nice_names=False
    )
    bus_sizes = bus_sizes.Generator

    transmission_branches = get_transmission_branches(n, bus_carrier=bus_carrier)
    branch_widths = n.statistics.installed_capacity(groupby=False).loc[
        transmission_branches
    ]

    bus_scale = 5e-6
    branch_scale = 1e-4
    bus_colors = n.c.carriers.static.color

    n.plot.map(
        bus_sizes=bus_sizes * bus_scale,
        bus_alpha=0.8,
        bus_colors=bus_colors,
        link_widths=branch_widths.get("Link", 0) * branch_scale,
        line_widths=branch_widths.get("Line", 0) * branch_scale,
    )
    plt.close()


def test_plot_layouter(ac_dc_network):
    n = ac_dc_network

    n.plot.map(layouter=nx.layout.planar_layout, geomap=False)
    plt.close()


def test_plot_map_flow(ac_dc_network):
    n = ac_dc_network

    branches = n.branches()
    lines = branches.loc["Line"]
    line_flow = pd.Series(range(len(lines)), index=lines.index)
    links = branches.loc["Link"]
    link_flow = pd.Series(range(len(links)), index=links.index)
    n.plot.map(line_flow=line_flow, link_flow=link_flow, geomap=False)
    plt.close()

    n.c.lines.dynamic.p0.loc[:, line_flow.index] = 0
    n.c.lines.dynamic.p0 += line_flow
    n.plot.map(line_flow="mean", geomap=False)
    plt.close()

    n.plot.map(line_flow=n.snapshots[0], geomap=False)
    plt.close()


def test_plot_map_line_colorbar(ac_dc_network):
    n = ac_dc_network

    norm = plt.Normalize(vmin=0, vmax=10)

    n.plot.map(
        line_colors=n.c.lines.static.index.astype(int),
        line_cmap="viridis",
        line_cmap_norm=norm,
    )

    plt.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=norm), ax=plt.gca())


def test_plot_map_bus_colorbar(ac_dc_network):
    n = ac_dc_network

    norm = plt.Normalize(vmin=0, vmax=10)

    n.plot.map(bus_colors=n.c.buses.static.x, bus_cmap="viridis", bus_cmap_norm=norm)

    plt.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=norm), ax=plt.gca())


def test_plot_legend_lines(ac_dc_network):
    n = ac_dc_network

    fig, ax = plt.subplots()
    n.plot.map(ax=ax, geomap=False)

    add_legend_lines(
        ax,
        [2, 5],
        ["label a", "label b"],
        patch_kw={"alpha": 0.5},
        legend_kw={"frameon": False},
    )

    plt.close()


def test_plot_legend_patches(ac_dc_network):
    n = ac_dc_network

    fig, ax = plt.subplots()
    n.plot.map(ax=ax, geomap=False)

    add_legend_patches(
        ax,
        ["r", "g", "b"],
        ["red", "green", "blue"],
        legend_kw={"frameon": False},
    )

    plt.close()


def test_plot_legend_circles_no_geomap(ac_dc_network):
    n = ac_dc_network

    fig, ax = plt.subplots()
    n.plot.map(ax=ax, geomap=False)

    add_legend_circles(ax, 1, "reference size")

    plt.close()


@pytest.mark.skipif(not cartopy_present, reason="Cartopy not installed")
def test_plot_legend_circles_geomap(ac_dc_network):
    n = ac_dc_network

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    n.plot.map(ax=ax, geomap=True)

    add_legend_circles(ax, [1, 0.5], ["reference A", "reference B"])

    plt.close()


@pytest.mark.skipif(not cartopy_present, reason="Cartopy not installed")
def test_plot_legend_semicircles_geomap(ac_dc_network):
    n = ac_dc_network

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    n.plot.map(ax=ax, geomap=True)

    add_legend_semicircles(ax, [1, 0.5], ["reference A", "reference B"])

    plt.close()


@pytest.mark.skipif(
    not explore_deps_present,
    reason="Dependencies for n.plot.explore() not installed: folium, mapclassify",
)
def test_network_explore(ac_dc_network):
    n = ac_dc_network

    n.plot.explore()


def test_plot_alias_for_plot_map(ac_dc_network):
    """Test that n.plot() is an alias for n.plot.map()."""
    n = ac_dc_network

    # Both should return the same type of object and produce equivalent plots
    # Test without geomap to avoid external dependencies
    result1 = n.plot(geomap=False)
    plt.close()

    result2 = n.plot.map(geomap=False)
    plt.close()

    # Both should return the same type of object
    assert type(result1) is type(result2)


@pytest.mark.skipif(
    not explore_deps_present,
    reason="Dependencies for n.plot.explore() not installed: folium, mapclassify",
)
def test_plot_explore_alias_for_explore(ac_dc_network):
    """Test that n.plot.explore is an alias for n.explore()."""
    n = ac_dc_network

    # Both should return the same type of object
    result1 = n.plot.explore()
    result2 = n.explore()

    # Both should return folium Map objects
    assert type(result1) is type(result2)
