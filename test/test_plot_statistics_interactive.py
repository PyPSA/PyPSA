"""Tests for interactive statistics plotting."""

import pandas as pd
import plotly.graph_objects as go


def test_iplot_exists(ac_dc_network_r):
    """Test that the iplot accessor exists."""
    n = ac_dc_network_r
    assert hasattr(n.statistics.installed_capacity, "iplot")


def test_iplot_bar_plot(ac_dc_network_r):
    """Test creating a bar plot."""
    n = ac_dc_network_r
    fig = n.statistics.installed_capacity.iplot.bar()
    assert isinstance(fig, go.Figure)
    assert any(isinstance(trace, go.Bar) for trace in fig.data)


def test_iplot_line_plot(ac_dc_network_r):
    """Test creating a line plot."""
    n = ac_dc_network_r
    fig = n.statistics.installed_capacity.iplot.line()
    assert isinstance(fig, go.Figure)
    assert any(isinstance(trace, go.Scatter) for trace in fig.data)


def test_iplot_area_plot(ac_dc_network_r):
    """Test creating an area plot."""
    n = ac_dc_network_r
    fig = n.statistics.installed_capacity.iplot.area()
    assert isinstance(fig, go.Figure)
    assert any(isinstance(trace, go.Scatter) for trace in fig.data)


def test_iplot_with_colors(ac_dc_network_r):
    """Test creating a plot with colors from carriers."""
    n = ac_dc_network_r
    fig = n.statistics.installed_capacity.iplot.bar(color="carrier")
    assert isinstance(fig, go.Figure)
    # Plotly Express sets colorway in layout for discrete colors
    assert "colorway" in fig.layout or any(
        hasattr(trace, "marker") and hasattr(trace.marker, "color")
        for trace in fig.data
    )


def test_iplot_different_statistics(ac_dc_network_r):
    """Test creating plots with different statistics."""
    n = ac_dc_network_r
    stats_methods = ["installed_capacity", "optimal_capacity", "supply", "curtailment"]

    for method_name in stats_methods:
        if hasattr(n.statistics, method_name):
            method = getattr(n.statistics, method_name)
            fig = method.iplot(kind="bar")
            assert isinstance(fig, go.Figure)


def test_iplot_facet_parameters(ac_dc_network_r):
    """Test creating a plot with facets."""
    n = ac_dc_network_r
    fig = n.statistics.installed_capacity.iplot.bar(facet_col="bus")
    assert isinstance(fig, go.Figure)
    # Faceted plots in Plotly Express create multiple subplots
    assert "xaxis2" in fig.layout or "yaxis2" in fig.layout


def test_iplot_query_parameter(ac_dc_network_r):
    """Test creating a plot with a query."""
    n = ac_dc_network_r
    # Get the data to create a query
    data = n.statistics.installed_capacity()
    if isinstance(data, pd.Series):
        data = data.reset_index()

    if "carrier" in data.columns and len(data["carrier"].unique()) > 1:
        test_carrier = data["carrier"].iloc[0]
        fig = n.statistics.installed_capacity.iplot.bar(
            query=f"carrier == '{test_carrier}'"
        )
        assert isinstance(fig, go.Figure)


def test_iplot_stacked_parameter(ac_dc_network_r):
    """Test creating a stacked plot."""
    n = ac_dc_network_r
    # Test stacked=True
    fig1 = n.statistics.installed_capacity.iplot.bar(stacked=True)
    assert isinstance(fig1, go.Figure)

    # Test stacked=False
    fig2 = n.statistics.installed_capacity.iplot.bar(stacked=False)
    assert isinstance(fig2, go.Figure)


def test_iplot_category_orders(ac_dc_network_r):
    """Test creating a plot with specified category orders."""
    n = ac_dc_network_r

    # Create plot with specified orders if applicable columns exist
    carriers = n.carriers.index.unique().tolist()
    buses = n.buses.index.unique().tolist()
    countries = n.buses.country.unique().tolist()

    # Create plot with the available orders
    fig = n.statistics.installed_capacity.iplot.bar(
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
    n = ac_dc_network_r
    fig = n.statistics.supply.iplot.area(stacked=False)
    assert isinstance(fig, go.Figure)


def test_iplot_sharex_sharey(ac_dc_network_r):
    """Test sharex and sharey parameters for faceted plots."""
    n = ac_dc_network_r
    fig1 = n.statistics.installed_capacity.iplot.bar(
        facet_col="country",
        facet_row="bus_carrier",
        sharex=False,
        sharey=True,
    )

    # Test with sharex=True, sharey=False
    fig2 = n.statistics.installed_capacity.iplot.bar(
        facet_col="country",
        facet_row="bus_carrier",
        sharex=True,
        sharey=False,
    )

    assert isinstance(fig1, go.Figure)
    assert isinstance(fig2, go.Figure)

    # For sharex=False, the xaxes should not match
    if "xaxis2" in fig1.layout:
        assert fig1.layout.xaxis.matches is None or not fig1.layout.xaxis.matches

    # For sharey=False, the yaxes should not match
    if "yaxis2" in fig2.layout:
        assert fig2.layout.yaxis.matches is None or not fig2.layout.yaxis.matches
