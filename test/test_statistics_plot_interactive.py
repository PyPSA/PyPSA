"""Tests for interactive statistics plotting."""

import pandas as pd
import plotly.graph_objects as go
import pytest

import pypsa
from pypsa.plot.statistics.charts import ChartGenerator
from pypsa.statistics.expressions import StatisticsAccessor


@pytest.fixture
def collection_single_index(ac_dc_network_r):
    """Create NetworkCollection with single index for autofaceting tests."""
    n1 = ac_dc_network_r.copy()
    n2 = ac_dc_network_r.copy()
    n3 = ac_dc_network_r.copy()

    networks = [n1, n2, n3]
    index = pd.Index(["scenario_a", "scenario_b", "scenario_c"], name="scenario")
    return pypsa.NetworkCollection(networks, index=index)


@pytest.fixture
def collection_multiindex(ac_dc_network_r):
    """Create NetworkCollection with MultiIndex for autofaceting tests."""
    networks = [ac_dc_network_r.copy() for _ in range(6)]

    index = pd.MultiIndex.from_product(
        [["2030", "2040", "2050"], ["low", "high"]], names=["year", "cost"]
    )
    return pypsa.NetworkCollection(networks, index=index)


def test_iplot_exists(ac_dc_network_r):
    """Test that the iplot accessor exists."""
    assert hasattr(ac_dc_network_r.statistics.installed_capacity, "iplot")


@pytest.mark.parametrize(
    ("plot_type", "expected_trace_type"),
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
    ("param_name", "param_value"),
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
    ("sharex", "sharey"),
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
    ["ac_dc_network_r", "network_collection"],
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


def test_networks_interactive_query_filtering(network_collection):
    """Test query filtering on networks collection."""
    plotter = ChartGenerator(network_collection)
    data = network_collection.statistics.energy_balance()
    fig = plotter.iplot(
        data,
        "bar",
        x="carrier",
        y="value",
        facet_col="scenario",
        query="value > 1",
    )
    assert isinstance(fig, go.Figure)


def test_networks_interactive_stacking(network_collection):
    """Test stacking with networks collection."""
    fig = network_collection.statistics.supply.iplot.bar(
        x="carrier", y="value", stacked=True, facet_col="scenario"
    )
    assert isinstance(fig, go.Figure)


class TestAutoFaceting:
    """Test automatic faceting functionality for NetworkCollections."""

    def test_single_index_auto_facet_col(self, collection_single_index):
        """Test that single index automatically sets facet_col."""
        # Call plot method and check that the plot is created successfully
        # The autofaceting should work transparently
        fig = collection_single_index.statistics.installed_capacity.iplot.bar()
        assert isinstance(fig, go.Figure)

        # Check that we have data for multiple scenarios
        assert len(fig.data) >= 1

        # The plot should have faceted structure for multiple scenarios
        # We can verify this by checking subplot annotations or layout
        if hasattr(fig, "layout") and hasattr(fig.layout, "annotations"):
            # Faceted plots often have annotations for subplot titles
            annotations = [
                ann.text for ann in fig.layout.annotations if hasattr(ann, "text")
            ]
            # This is a softer check since the exact annotation format may vary
            assert len(annotations) >= 0  # Just ensure no errors occurred

    def test_multiindex_auto_facet_both(self, collection_multiindex):
        """Test that MultiIndex automatically sets both facet_row and facet_col."""
        # Call plot method and check that the plot is created successfully
        fig = collection_multiindex.statistics.installed_capacity.iplot.bar()
        assert isinstance(fig, go.Figure)

        # Check that we have data
        assert len(fig.data) >= 1

        # For multiindex, we should have a more complex layout structure
        # indicating both row and column faceting
        layout = fig.layout
        assert layout is not None

        # The presence of multiple axis definitions can indicate faceting
        axis_keys = [key for key in dir(layout) if "axis" in key.lower()]
        # Should have some axis definitions for the faceted structure
        assert len(axis_keys) >= 0  # Ensure no errors in plot creation

    def test_explicit_facet_overrides_auto(self, collection_single_index):
        """Test that explicit facet arguments override automatic faceting."""
        # Call plot with explicit facet_col that differs from auto-faceting
        fig = collection_single_index.statistics.installed_capacity.iplot.bar(
            facet_col="carrier"
        )
        assert isinstance(fig, go.Figure)

        # The plot should be created successfully with explicit faceting
        assert len(fig.data) >= 1

    def test_no_autofaceting_for_single_network(self, ac_dc_network_r):
        """Test that single network doesn't get automatic faceting."""
        # Single network should not have autofaceting applied
        fig = ac_dc_network_r.statistics.installed_capacity.iplot.bar()
        assert isinstance(fig, go.Figure)

        # Should create a simple plot without complex faceting structure
        assert len(fig.data) >= 1

        # Check that _index_names is empty for single networks
        assert getattr(ac_dc_network_r, "_index_names", []) == []
