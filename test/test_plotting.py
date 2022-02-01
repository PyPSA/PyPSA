#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:13:59 2022

@author: fabian
"""

import pytest
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


@pytest.mark.parametrize("geomap", (True, False))
@pytest.mark.parametrize("margin", (None, 0.1))
@pytest.mark.parametrize("jitter", (None, 1))
def test_plot_standard_params(ac_dc_network, geomap, margin, jitter):
    n = ac_dc_network

    n.plot(geomap=geomap, margin=margin, jitter=jitter)


def test_plot_on_axis(ac_dc_network):
    n = ac_dc_network
    fig, ax = plt.subplots()
    n.plot(ax=ax, geomap=False)
    plt.close()

    with pytest.raises(AssertionError):
        n.plot(ax=ax, geomap=True)


def test_plot_bus_circles(ac_dc_network):
    n = ac_dc_network

    bus_sizes = n.generators.groupby(["bus", "carrier"]).p_nom.mean()
    bus_sizes[:] = 1
    bus_colors = pd.Series(["blue", "red", "green"], index=n.carriers.index)
    n.plot(bus_sizes=bus_sizes, bus_colors=bus_colors)
    plt.close()

    # Retrieving the colors from carriers also should work
    n.carriers["color"] = bus_colors
    n.plot(bus_sizes=bus_sizes)
    plt.close()


def test_plot_with_bus_cmap(ac_dc_network):
    n = ac_dc_network

    buses = n.buses.index
    colors = pd.Series(np.random.rand(len(buses)), buses)
    n.plot(bus_colors=colors, bus_cmap="coolwarm")
    plt.close()


def test_plot_with_line_cmap(ac_dc_network):
    n = ac_dc_network

    lines = n.lines.index
    colors = pd.Series(np.random.rand(len(lines)), lines)
    n.plot(line_colors=colors, line_cmap="coolwarm")
    plt.close()


def test_plot_layouter(ac_dc_network):
    n = ac_dc_network

    n.plot(layouter=nx.layout.planar_layout)
    plt.close()


def test_plot_map_flow(ac_dc_network):
    n = ac_dc_network

    branches = n.branches()
    flow = pd.Series(range(len(branches)), index=branches.index)
    n.plot(flow=flow)
    plt.close()

    n.lines_t.p0.loc[:, flow.Line.index] = 0
    n.lines_t.p0 += flow.Line
    n.plot(flow="mean")
    plt.close()

    n.plot(flow=n.snapshots[0])
    plt.close()
