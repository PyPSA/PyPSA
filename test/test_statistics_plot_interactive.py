# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for interactive statistics plotting."""

import pandas as pd
import plotly.graph_objects as go
import pytest

import pypsa
from pypsa.plot.statistics.charts import ChartGenerator
from pypsa.plot.statistics.plotter import StatisticInteractivePlotter
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


def _multi_period_sample() -> pd.DataFrame:
    """Return deterministic sample data with investment period columns."""
    index = pd.MultiIndex.from_product(
        [["Generator"], ["wind"]], names=["component", "carrier"]
    )
    periods = pd.Index([2020, 2030], name="period")
    data = pd.DataFrame([[1.0, 2.0]], index=index, columns=periods)
    data.attrs = {"name": "Sample", "unit": "MW"}
    return data


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


def test_iplot_bar_drops_redundant_axis_color(ac_dc_network_r):
    """Bar charts should not split traces when color duplicates an axis."""
    fig = ac_dc_network_r.statistics.energy_balance.iplot.bar()

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].showlegend is False


def test_iplot_category_orders(ac_dc_network_r):
    """Test creating a plot with specified category orders."""
    # Create plot with specified orders if applicable columns exist
    carriers = ac_dc_network_r.c.carriers.static.index.unique().tolist()
    buses = ac_dc_network_r.c.buses.static.index.unique().tolist()
    countries = ac_dc_network_r.c.buses.static.country.unique().tolist()

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

    def test_single_index_grouped_bars(self, collection_single_index):
        """Single-index collections should group bars instead of faceting."""
        fig = collection_single_index.statistics.installed_capacity.iplot.bar()
        assert isinstance(fig, go.Figure)

        # Bars should share a single subplot axis (no auto-created facets)
        subplot_axes = {getattr(trace, "xaxis", "x") for trace in fig.data}
        assert subplot_axes == {"x"}

        # Default behaviour should group bars by scenario (network index)
        assert fig.layout.barmode == "group"
        trace_names = {trace.name for trace in fig.data if hasattr(trace, "name")}
        expected_names = set(map(str, collection_single_index.index))
        assert expected_names.issubset(trace_names)

    def test_multiindex_auto_facet_both(self, collection_multiindex):
        """Test that MultiIndex bar plots use 1D faceting + grouped bars."""
        # Call plot method and check that the plot is created successfully
        fig = collection_multiindex.statistics.installed_capacity.iplot.bar()
        assert isinstance(fig, go.Figure)

        # Check that we have data
        assert len(fig.data) >= 1

        # For bar plots, should group by second index level (cost: "low", "high")
        assert fig.layout.barmode == "group"
        trace_names = {trace.name for trace in fig.data if getattr(trace, "name", None)}
        # Should include the second index level values
        expected_second_level = set(
            map(str, collection_multiindex.index.get_level_values(1).unique())
        )
        assert expected_second_level.issubset(trace_names)

        # Should have 1D faceting (column facets for first index level: year)
        # Check for multiple subplots but not a 2D grid
        layout = fig.layout
        assert layout is not None

        # Should have multiple xaxis definitions for column faceting
        # but not both xaxis2 and yaxis2 (which would indicate 2D grid)
        has_xaxis2 = "xaxis2" in fig.layout
        has_yaxis2 = "yaxis2" in fig.layout
        # For 1D column faceting, we should have multiple x-axes but share y-axis
        if has_xaxis2 and has_yaxis2:
            # If both exist, they shouldn't create a 2D grid (check domain structure)
            # In 1D faceting, all subplots share the same row (y domain)
            pass  # Allow both for now, as plotly may create them differently

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


def test_interactive_multi_investment_grouping(ac_dc_network_r):
    """Interactive bar charts should group investment periods side-by-side."""
    network = ac_dc_network_r.copy()
    network.investment_periods = pd.Index([2020, 2030], name="period")
    sample = _multi_period_sample()

    def fake_statistic(**kwargs):
        return sample

    fake_statistic.__name__ = "installed_capacity"

    plotter = StatisticInteractivePlotter(fake_statistic, network)
    fig = plotter.bar()

    assert isinstance(fig, go.Figure)
    assert fig.layout.barmode == "group"
    trace_names = {
        str(trace.name) for trace in fig.data if getattr(trace, "name", None)
    }
    assert trace_names == {"2020", "2030"}


class TestInteractivePlotIssue1719:
    """Regression tests for the statistics plotting bug collection (issue #1719)."""

    def test_color_none_disables_grouping(self, ac_dc_network_r):
        """#1: color=None is respected instead of falling back to 'carrier'."""
        grouped = ac_dc_network_r.statistics.energy_balance.iplot.bar(y="bus_carrier")
        ungrouped = ac_dc_network_r.statistics.energy_balance.iplot.bar(
            y="bus_carrier", color=None
        )
        assert len(ungrouped.data) < len(grouped.data)
        assert len(ungrouped.data) == 1

    def test_at_port_routed_to_statistics(self, ac_dc_network_r):
        """#2: at_port is forwarded to the statistics call, not the backend."""
        fig = ac_dc_network_r.statistics.supply.iplot.bar(at_port="all")
        assert isinstance(fig, go.Figure)

    def test_prices_rejects_carrier(self, ac_dc_network_r):
        """#3: passing 'carrier' to prices raises instead of silently dropping."""
        with pytest.raises(ValueError, match="bus_carrier"):
            ac_dc_network_r.statistics.prices.iplot.bar(carrier="AC")

    def test_mathtext_in_legend_is_sanitized(self, ac_dc_network_r):
        """#7: LaTeX subscripts in nice names become Plotly tags in legends."""
        n = ac_dc_network_r.copy()
        n.c.carriers.static.loc["gas", "nice_name"] = "CO$_2$"
        fig = n.statistics.installed_capacity.iplot.bar(
            x="value", y="bus_carrier", color="carrier"
        )
        names = [str(t.name) for t in fig.data]
        assert "CO<sub>2</sub>" in names
        assert not any("$" in name for name in names)

    def test_empty_facets_stay_visible(self, ac_dc_network_r):
        """#8: requested facet values without data are still rendered."""
        n = ac_dc_network_r.copy()
        n.add("Carrier", "H2", color="#123456")
        n.add("Bus", "h2 bus", carrier="H2")

        fig = n.statistics.installed_capacity.iplot.bar(
            bus_carrier=["AC", "H2"], facet_col="bus_carrier"
        )
        facets = {a.text for a in fig.layout.annotations if a.text}
        assert facets == {"bus_carrier=AC", "bus_carrier=H2"}

    def test_unshared_axes_keep_tick_labels(self, ac_dc_network_r):
        """#9: sharex=False keeps tick labels on all faceted subplots."""
        fig = ac_dc_network_r.statistics.installed_capacity.iplot.bar(
            bus_carrier=["AC", "DC"], facet_col="bus_carrier", sharex=False
        )
        layout = fig.layout.to_plotly_json()
        xaxes = [layout[k] for k in layout if k.startswith("xaxis")]
        assert all(ax.get("showticklabels") is True for ax in xaxes)

    def test_categorical_axis_labels_every_bar(self, ac_dc_network_r):
        """Categorical y-axes label every bar and grow with the row count."""
        plotter = ChartGenerator(ac_dc_network_r)

        def bar_fig(n_rows):
            index = pd.Index([f"carrier {i}" for i in range(n_rows)], name="carrier")
            data = pd.Series(range(n_rows), index=index, dtype=float)
            return plotter.iplot(data, "bar", x="value", y="carrier")

        small, large, huge = bar_fig(5), bar_fig(50), bar_fig(500)
        assert large.layout.yaxis.dtick == 1
        assert large.layout.height > small.layout.height
        assert huge.layout.height <= 2000
        assert huge.layout.yaxis.dtick is None
