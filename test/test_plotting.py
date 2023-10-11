#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:13:59 2022.

@author: fabian
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches
from pypsa.statistics import get_transmission_branches

try:
    import cartopy.crs as ccrs

    cartopy_present = True
except ImportError:
    cartopy_present = False


@pytest.mark.parametrize("margin", (None, 0.1))
@pytest.mark.parametrize("jitter", (None, 1))
@pytest.mark.skipif(os.name == "nt", reason="tcl_findLibrary on Windows")
def test_plot_standard_params_wo_geomap(ac_dc_network, margin, jitter):
    n = ac_dc_network
    n.plot(geomap=False, margin=margin, jitter=jitter)
    plt.close()


@pytest.mark.skipif(not cartopy_present, reason="Cartopy not installed")
@pytest.mark.parametrize("margin", (None, 0.1))
@pytest.mark.parametrize("jitter", (None, 1))
def test_plot_standard_params_w_geomap(ac_dc_network, margin, jitter):
    n = ac_dc_network
    n.plot(geomap=True, margin=margin, jitter=jitter)
    plt.close()


def test_plot_on_axis_wo_geomap(ac_dc_network):
    n = ac_dc_network
    fig, ax = plt.subplots()
    n.plot(ax=ax, geomap=False)
    plt.close()


@pytest.mark.skipif(not cartopy_present, reason="Cartopy not installed")
def test_plot_on_axis_w_geomap(ac_dc_network):
    n = ac_dc_network
    fig, ax = plt.subplots()
    with pytest.raises(AssertionError):
        n.plot(ax=ax, geomap=True)
        plt.close()


def test_plot_bus_circles(ac_dc_network):
    n = ac_dc_network

    bus_sizes = n.generators.groupby(["bus", "carrier"]).p_nom.mean()
    bus_sizes[:] = 1
    bus_colors = pd.Series(["blue", "red", "green"], index=n.carriers.index)
    n.plot(bus_sizes=bus_sizes, bus_colors=bus_colors, geomap=False)
    plt.close()

    # Retrieving the colors from carriers also should work
    n.carriers["color"] = bus_colors
    n.plot(bus_sizes=bus_sizes)
    plt.close()

    # Retrieving the colors from carriers also should work
    n.carriers["color"] = bus_colors
    n.plot(bus_sizes=bus_sizes)
    plt.close()


def test_plot_split_circles(ac_dc_network):
    n = ac_dc_network

    gen_sizes = n.generators.groupby(["bus", "carrier"]).p_nom.sum()
    gen_sizes[:] = 500
    n.loads.carrier = "load"
    load_sizes = -n.loads_t.p_set.mean().groupby([n.loads.bus, n.loads.carrier]).max()
    bus_sizes = pd.concat((gen_sizes, load_sizes)) / 1e3
    bus_colors = pd.Series(
        ["blue", "red", "green", "orange"], index=list(n.carriers.index) + ["load"]
    )
    n.plot(
        bus_sizes=bus_sizes, bus_colors=bus_colors, bus_split_circles=True, geomap=False
    )
    plt.close()


def test_plot_with_bus_cmap(ac_dc_network):
    n = ac_dc_network

    buses = n.buses.index
    colors = pd.Series(np.random.rand(len(buses)), buses)
    n.plot(bus_colors=colors, bus_cmap="coolwarm", geomap=False)
    plt.close()


def test_plot_with_line_cmap(ac_dc_network):
    n = ac_dc_network

    lines = n.lines.index
    colors = pd.Series(np.random.rand(len(lines)), lines)
    n.plot(line_colors=colors, line_cmap="coolwarm", geomap=False)
    plt.close()


def test_plot_alpha(ac_dc_network):
    n = ac_dc_network

    bus_sizes = n.generators.groupby(["bus", "carrier"]).p_nom.mean()
    bus_sizes[:] = 1
    bus_colors = pd.Series(["blue", "red", "green"], index=n.carriers.index)
    n.plot(
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        geomap=False,
        bus_alpha=0.5,
        line_alpha=0.5,
        link_alpha=0.5,
    )
    plt.close()

    # Retrieving the colors from carriers also should work
    n.carriers["color"] = bus_colors
    n.plot(bus_sizes=bus_sizes)
    plt.close()


def test_plot_line_subset(ac_dc_network):
    n = ac_dc_network

    lines = n.lines.index[:2]
    colors = pd.Series(np.random.rand(len(lines)), lines)
    n.plot(line_colors=colors, line_cmap="coolwarm", geomap=False)
    plt.close()


def test_plot_bus_subset(ac_dc_network):
    n = ac_dc_network

    buses = n.buses.index[:2]
    colors = pd.Series(np.random.rand(len(buses)), buses)
    n.plot(bus_colors=colors, bus_cmap="coolwarm", geomap=False)
    plt.close()

    bus_sizes = n.generators.groupby(["bus", "carrier"]).p_nom.mean()[:3]
    bus_sizes[:] = 1
    bus_colors = pd.Series(["blue", "red", "green"], index=n.carriers.index)
    n.plot(
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

    grouper = n.statistics.groupers.get_bus_and_carrier
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
    bus_colors = pd.Series(["blue", "red", "green"], index=n.carriers.index)

    n.plot(
        bus_sizes=bus_sizes * bus_scale,
        bus_alpha=0.8,
        bus_colors=bus_colors,
        link_widths=branch_widths.get("Link", 0) * branch_scale,
        line_widths=branch_widths.get("Line", 0) * branch_scale,
    )
    plt.close()


def test_plot_layouter(ac_dc_network):
    n = ac_dc_network

    n.plot(layouter=nx.layout.planar_layout, geomap=False)
    plt.close()


def test_plot_map_flow(ac_dc_network):
    n = ac_dc_network

    branches = n.branches()
    flow = pd.Series(range(len(branches)), index=branches.index)
    n.plot(flow=flow, geomap=False)
    plt.close()

    n.lines_t.p0.loc[:, flow.Line.index] = 0
    n.lines_t.p0 += flow.Line
    n.plot(flow="mean", geomap=False)
    plt.close()

    n.plot(flow=n.snapshots[0], geomap=False)
    plt.close()


def test_plot_map_line_colorbar(ac_dc_network):
    n = ac_dc_network

    norm = plt.Normalize(vmin=0, vmax=10)

    n.plot(line_colors=n.lines.index.astype(int), line_cmap="viridis", line_norm=norm)

    plt.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=norm), ax=plt.gca())


def test_plot_map_bus_colorbar(ac_dc_network):
    n = ac_dc_network

    norm = plt.Normalize(vmin=0, vmax=10)

    n.plot(bus_colors=n.buses.x, bus_cmap="viridis", bus_norm=norm)

    plt.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=norm), ax=plt.gca())


def test_plot_legend_lines(ac_dc_network):
    n = ac_dc_network

    fig, ax = plt.subplots()
    n.plot(ax=ax, geomap=False)

    add_legend_lines(
        ax,
        [2, 5],
        ["label a", "label b"],
        patch_kw=dict(alpha=0.5),
        legend_kw=dict(frameon=False),
    )

    plt.close()


def test_plot_legend_patches(ac_dc_network):
    n = ac_dc_network

    fig, ax = plt.subplots()
    n.plot(ax=ax, geomap=False)

    add_legend_patches(
        ax,
        ["r", "g", "b"],
        ["red", "green", "blue"],
        legend_kw=dict(frameon=False),
    )

    plt.close()


def test_plot_legend_circles_no_geomap(ac_dc_network):
    n = ac_dc_network

    fig, ax = plt.subplots()
    n.plot(ax=ax, geomap=False)

    add_legend_circles(ax, 1, "reference size")

    plt.close()


@pytest.mark.skipif(not cartopy_present, reason="Cartopy not installed")
def test_plot_legend_circles_geomap(ac_dc_network):
    n = ac_dc_network

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    n.plot(ax=ax, geomap=True)

    add_legend_circles(ax, [1, 0.5], ["reference A", "reference B"])

    plt.close()
