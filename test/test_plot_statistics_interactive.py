"""Tests for interactive statistics plotting."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from pypsa.plot.statistics.charts import ChartGenerator
from pypsa.statistics.expressions import StatisticsAccessor


def test_iplot_exists(ac_dc_network_r):
    """Test that the iplot accessor exists."""
    assert hasattr(ac_dc_network_r.statistics.installed_capacity, "iplot")


@pytest.mark.parametrize(
    "plot_type,expected_trace_type",
    [
        ("bar", go.Bar),
        ("line", go.Scatter),
        ("area", go.Scatter),
    ],
)
def test_iplot_plot_types(ac_dc_network_r, plot_type, expected_trace_type):
    """Test creating different plot types."""
    plot_method = getattr(
        ac_dc_network_r.statistics.installed_capacity.iplot, plot_type
    )
    fig = plot_method()
    assert isinstance(fig, go.Figure)
    assert any(isinstance(trace, expected_trace_type) for trace in fig.data)


@pytest.mark.parametrize(
    "param_name,param_value",
    [
        ("color", "carrier"),
        ("facet_col", "carrier"),
        ("facet_row", "carrier"),
    ],
)
def test_iplot_layout_parameters(ac_dc_network_r, param_name, param_value):
    """Test creating plots with different layout parameters."""
    kwargs = {param_name: param_value}
    fig = ac_dc_network_r.statistics.installed_capacity.iplot.bar(**kwargs)
    assert isinstance(fig, go.Figure)

    if param_name == "color":
        # Plotly Express sets colorway in layout for discrete colors
        assert "colorway" in fig.layout or any(
            hasattr(trace, "marker") and hasattr(trace.marker, "color")
            for trace in fig.data
        )


def test_iplot_facet_parameters(ac_dc_network_r):
    """Test creating a plot with facets."""
    fig = ac_dc_network_r.statistics.installed_capacity.iplot.bar(facet_col="bus")
    assert isinstance(fig, go.Figure)
    # Faceted plots in Plotly Express create multiple subplots (if there are enough data points)
    if len(fig.data) > 1:
        assert "xaxis2" in fig.layout or "yaxis2" in fig.layout


def test_iplot_query_parameter(ac_dc_network_r):
    """Test creating a plot with a query."""
    # Get the data to create a query
    data = ac_dc_network_r.statistics.installed_capacity()
    if isinstance(data, pd.Series):
        data = data.reset_index()

    if "carrier" in data.columns and len(data["carrier"].unique()) > 1:
        test_carrier = data["carrier"].iloc[0]
        fig = ac_dc_network_r.statistics.installed_capacity.iplot.bar(
            query=f"carrier == '{test_carrier}'"
        )
        assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("stacked", [True, False])
def test_iplot_stacked_parameter(ac_dc_network_r, stacked):
    """Test creating stacked and unstacked plots."""
    fig = ac_dc_network_r.statistics.installed_capacity.iplot.bar(stacked=stacked)
    assert isinstance(fig, go.Figure)


def test_iplot_category_orders(ac_dc_network_r):
    """Test creating a plot with specified category orders."""
    # Create plot with specified orders if applicable columns exist
    carriers = ac_dc_network_r.carriers.index.unique().tolist()
    buses = ac_dc_network_r.buses.index.unique().tolist()
    countries = ac_dc_network_r.buses.country.unique().tolist()

    # Create plot with the available orders
    fig = ac_dc_network_r.statistics.installed_capacity.iplot.bar(
        facet_row="bus",
        facet_col="country",
        color="carrier",
        color_order=carriers,
        row_order=buses,
        col_order=countries,
    )
    assert isinstance(fig, go.Figure)


def test_iplot_unstacked_area_plot(ac_dc_network_r):
    """Test creating an unstacked area plot."""
    fig = ac_dc_network_r.statistics.supply.iplot.area(stacked=False)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    "sharex,sharey",
    [
        (False, True),
        (True, False),
    ],
)
def test_iplot_sharex_sharey(ac_dc_network_r, sharex, sharey):
    """Test sharex and sharey parameters for faceted plots."""
    fig = ac_dc_network_r.statistics.installed_capacity.iplot.bar(
        facet_col="country",
        facet_row="bus_carrier",
        sharex=sharex,
        sharey=sharey,
    )
    assert isinstance(fig, go.Figure)

    # Check axis matching behavior if we have multiple axes
    if "xaxis2" in fig.layout and not sharex:
        assert fig.layout.xaxis.matches is None or not fig.layout.xaxis.matches
    if "yaxis2" in fig.layout and not sharey:
        assert fig.layout.yaxis.matches is None or not fig.layout.yaxis.matches


@pytest.mark.parametrize(
    "plot_type", ["", "bar", "line", "area"], ids=["default", "bar", "line", "area"]
)
@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_all_stat_functions_all_plot_types(ac_dc_network_r, plot_type, stat_func):
    """Consolidated test for all statistics functions with all plot types for both network fixtures."""
    plotter = getattr(ac_dc_network_r.statistics, stat_func)

    if plot_type:
        plot_method = getattr(plotter.iplot, plot_type)
        fig = plot_method()
    else:
        fig = plotter.iplot()

    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    "network_fixture",
    ["ac_dc_network_r", "networks_scenario"],
    ids=["single_network", "multiple_networks"],
)
@pytest.mark.parametrize(
    "plot_type", ["", "bar", "line", "area"], ids=["default", "bar", "line", "area"]
)
def test_stat_functions_all_plot_types_with_multi(request, network_fixture, plot_type):
    """Consolidated test for all statistics functions with all plot types for both network fixtures."""
    network = request.getfixturevalue(network_fixture)
    plotter = getattr(network.statistics, "installed_capacity")

    if plot_type:
        plot_method = getattr(plotter.iplot, plot_type)
        fig = plot_method()
    else:
        fig = plotter.iplot()

    assert isinstance(fig, go.Figure)


def test_networks_interactive_query_filtering(networks_scenario):
    """Test query filtering on networks collection."""
    plotter = ChartGenerator(networks_scenario)
    data = networks_scenario.statistics.energy_balance()
    fig = plotter.iplot(
        data,
        "bar",
        x="carrier",
        y="value",
        facet_col="scenario",
        query="value > 1",
    )
    assert isinstance(fig, go.Figure)


def test_networks_interactive_stacking(networks_scenario):
    """Test stacking with networks collection."""
    fig = networks_scenario.statistics.supply.iplot.bar(
        x="carrier", y="value", stacked=True, facet_col="scenario"
    )
    assert isinstance(fig, go.Figure)
