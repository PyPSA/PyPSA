"""Tests for interactive statistics plotting."""

import pandas as pd
import plotly.graph_objects as go
import pytest

import pypsa


@pytest.fixture
def ac_dc_network():
    return pypsa.examples.ac_dc_meshed()


def test_iplot_exists(ac_dc_network):
    """Test that the iplot accessor exists."""
    n = ac_dc_network
    assert hasattr(n.statistics.installed_capacity, "iplot")


def test_iplot_bar_plot(ac_dc_network):
    """Test creating a bar plot."""
    n = ac_dc_network
    fig = n.statistics.installed_capacity.iplot.bar()
    assert isinstance(fig, go.Figure)
    assert any(isinstance(trace, go.Bar) for trace in fig.data)


def test_iplot_line_plot(ac_dc_network):
    """Test creating a line plot."""
    n = ac_dc_network
    fig = n.statistics.installed_capacity.iplot.line()
    assert isinstance(fig, go.Figure)
    assert any(isinstance(trace, go.Scatter) for trace in fig.data)


def test_iplot_area_plot(ac_dc_network):
    """Test creating an area plot."""
    n = ac_dc_network
    fig = n.statistics.installed_capacity.iplot.area()
    assert isinstance(fig, go.Figure)
    assert any(
        isinstance(trace, go.Scatter) and trace.fill is not None for trace in fig.data
    )


def test_iplot_with_colors(ac_dc_network):
    """Test creating a plot with colors from carriers."""
    n = ac_dc_network
    fig = n.statistics.installed_capacity.iplot.bar(color="carrier")
    assert isinstance(fig, go.Figure)
    # Plotly Express sets colorway in layout for discrete colors
    assert "colorway" in fig.layout or any(
        hasattr(trace, "marker") and hasattr(trace.marker, "color")
        for trace in fig.data
    )


def test_iplot_different_statistics(ac_dc_network):
    """Test creating plots with different statistics."""
    n = ac_dc_network
    stats_methods = ["installed_capacity", "optimal_capacity", "supply", "curtailment"]

    for method_name in stats_methods:
        if hasattr(n.statistics, method_name):
            method = getattr(n.statistics, method_name)
            fig = method.iplot(kind="bar")
            assert isinstance(fig, go.Figure)


def test_iplot_facet_parameters(ac_dc_network):
    """Test creating a plot with facets."""
    n = ac_dc_network
    fig = n.statistics.installed_capacity.iplot.bar(facet_col="bus")
    assert isinstance(fig, go.Figure)
    # Faceted plots in Plotly Express create multiple subplots
    assert "xaxis2" in fig.layout or "yaxis2" in fig.layout


def test_iplot_query_parameter(ac_dc_network):
    """Test creating a plot with a query."""
    n = ac_dc_network
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


def test_iplot_stacked_parameter(ac_dc_network):
    """Test creating a stacked plot."""
    n = ac_dc_network
    # Test stacked=True
    fig1 = n.statistics.installed_capacity.iplot.bar(stacked=True)
    assert isinstance(fig1, go.Figure)

    # Test stacked=False
    fig2 = n.statistics.installed_capacity.iplot.bar(stacked=False)
    assert isinstance(fig2, go.Figure)
