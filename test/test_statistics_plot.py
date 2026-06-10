# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from pypsa.consistency import ConsistencyError
from pypsa.plot.statistics.base import UNSET
from pypsa.plot.statistics.charts import (
    CHART_TYPES,
    ChartGenerator,
    adjust_collection_bar_defaults,
    prepare_bar_data,
)
from pypsa.plot.statistics.maps import MapPlotGenerator
from pypsa.plot.statistics.plotter import StatisticPlotter
from pypsa.statistics.expressions import StatisticsAccessor

# Set random seed for reproducibility
np.random.seed(42)  # noqa: NPY002

plt.rcdefaults()
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["figure.dpi"] = 100


def _multi_period_sample() -> pd.DataFrame:
    """Return deterministic sample data with investment period columns."""
    index = pd.MultiIndex.from_product(
        [["Generator"], ["wind"]], names=["component", "carrier"]
    )
    periods = pd.Index([2020, 2030], name="period")
    data = pd.DataFrame([[1.0, 2.0]], index=index, columns=periods)
    data.attrs = {"name": "Sample", "unit": "MW"}
    return data


@pytest.mark.skipif(
    sys.version_info < (3, 13) or sys.platform not in ["darwin"],
    reason="Run only once for stability.",
)
@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
@pytest.mark.mpl_image_compare(tolerance=40)
def test_simple_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    fig, _, _ = plotter.plot()

    return fig


@pytest.mark.skipif(
    sys.version_info < (3, 13) or sys.platform not in ["darwin"],
    reason="Run only once for stability.",
)
@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
@pytest.mark.parametrize("kind", CHART_TYPES + ["map"])
@pytest.mark.mpl_image_compare(tolerance=40)
def test_plot_types(ac_dc_network_r, stat_func, kind):
    if kind == "map" and stat_func == "prices":
        pytest.skip("Map plotting for 'prices' is not implemented.")
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    fig, *_ = plotter.plot(kind=kind)

    return fig


def test_to_long_format_static(ac_dc_network_r):
    """Test the _to_long_format method with optimal_capacity data."""
    # Create the accessor instance
    accessor = ChartGenerator(ac_dc_network_r)

    # Get optimal capacity data from statistics
    data = ac_dc_network_r.statistics.optimal_capacity()

    # Convert to long format
    long_data = accessor._to_long_format(data)

    # Check the output structure
    assert isinstance(long_data, pd.DataFrame)
    assert set(long_data.columns) == {"component", "carrier", "value"}


def test_to_long_format_dynamic(ac_dc_network_r):
    """Test the _to_long_format method with installed_capacity data."""
    # Create the accessor instance
    accessor = ChartGenerator(ac_dc_network_r)

    # Get installed capacity data from statistics
    data = ac_dc_network_r.statistics.energy_balance()

    # Convert to long format
    long_data = accessor._to_long_format(data)

    # Check the output structure
    assert isinstance(long_data, pd.DataFrame)
    assert set(long_data.columns) == {"component", "carrier", "bus_carrier", "value"}


def test_to_long_format_dynamic_multi(network_collection):
    """Test the _to_long_format method with installed_capacity data."""
    # Create the accessor instance
    accessor = ChartGenerator(network_collection)

    # Get installed capacity data from statistics
    data = network_collection.statistics.energy_balance()

    # Convert to long format
    long_data = accessor._to_long_format(data)

    # Check the output structure
    assert isinstance(long_data, pd.DataFrame)
    assert set(long_data.columns) == {
        "component",
        "carrier",
        "bus_carrier",
        "value",
    }.union(network_collection.index.names)


def test_derive_statistic_parameters(ac_dc_network_r):
    """Test derivation of statistic parameters"""
    # TODO rewrite once function is updated
    plotter = ChartGenerator(ac_dc_network_r)

    # Test with default parameters
    stats_kwargs = plotter.derive_statistic_parameters("carrier", "value", "carrier")
    assert stats_kwargs["groupby"] == ["carrier"]
    assert stats_kwargs["aggregate_across_components"]

    # Test with default parameters
    stats_kwargs = plotter.derive_statistic_parameters(
        "carrier", "value", "bus_carrier"
    )
    assert set(stats_kwargs["groupby"]) == {"bus_carrier", "carrier"}
    assert stats_kwargs["aggregate_across_components"]


def test_get_carrier_colors_and_labels(ac_dc_network_r):
    """Test carrier colors and labels retrieval"""
    plotter = ChartGenerator(ac_dc_network_r)

    colors = plotter.get_carrier_colors()
    assert isinstance(colors, dict)
    assert "-" in colors
    assert None in colors

    labels = plotter.get_carrier_labels()
    assert isinstance(labels, pd.Series)

    # Test with nice_names=False
    labels_raw = plotter.get_carrier_labels(nice_names=False)
    assert isinstance(labels_raw, pd.Series)
    assert (labels_raw.index.values == labels.values).all()


def test_query_filtering(ac_dc_network_r):
    """Test query filtering in plots"""
    plotter = ChartGenerator(ac_dc_network_r)
    data = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="carrier"))

    fig, ax, g = plotter.plot(data, "bar", x="carrier", y="value", query="value > 1")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)
    plt.close(fig)


def test_consistency_checks(ac_dc_network_r):
    """Test plotting consistency checks"""
    plotter = ChartGenerator(ac_dc_network_r)
    n = ac_dc_network_r.copy()
    plotter = ChartGenerator(n)
    n.c.carriers.static.color = pd.Series()
    # Test with missing carrier colors
    with pytest.raises(ConsistencyError):
        plotter.plot(data=pd.DataFrame(), kind="area", x="carrier", y="value")


def test_stacking(ac_dc_network_r):
    """Test stacking options in bar plots"""
    n = ac_dc_network_r
    fig, ax, g = n.statistics.supply.plot.bar(x="carrier", y="value", stacked=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)
    plt.close(fig)


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_networks_simple_plot(network_collection, stat_func):
    plotter = getattr(network_collection.statistics, stat_func)
    fig, ax, g = plotter.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_networks_bar_plot(network_collection, stat_func):
    plotter = getattr(network_collection.statistics, stat_func)
    fig, ax, g = plotter.plot.bar(facet_col="scenario")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)
    plt.close(fig)


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_networks_line_plot(network_collection, stat_func):
    plotter = getattr(network_collection.statistics, stat_func)
    fig, ax, g = plotter.plot.line(facet_col="scenario")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)
    plt.close(fig)


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_networks_area_plot(network_collection, stat_func):
    plotter = getattr(network_collection.statistics, stat_func)
    fig, ax, g = plotter.plot.area(facet_col="scenario")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)
    plt.close(fig)


def test_prepare_bar_data_stacks_periods(ac_dc_network_r):
    network = ac_dc_network_r.copy()
    periods = pd.Index([2020, 2030], name="period")
    network.investment_periods = periods

    data = _multi_period_sample()
    stacked = prepare_bar_data(network, "bar", data)

    assert isinstance(stacked, pd.Series)
    assert "period" in stacked.index.names
    assert set(stacked.index.get_level_values("period")) == {"2020", "2030"}
    assert stacked.attrs == data.attrs


def test_adjust_defaults_for_multi_investment(ac_dc_network_r):
    network = ac_dc_network_r.copy()
    periods = pd.Index([2020, 2030], name="period")
    network.investment_periods = periods

    color, stacked, order = adjust_collection_bar_defaults(
        network, "bar", UNSET, True, None
    )

    assert color == "period"
    assert stacked is False
    assert list(order) == list(periods)


def test_statistic_plotter_multi_investment_defaults(ac_dc_network_r):
    network = ac_dc_network_r.copy()
    network.investment_periods = pd.Index([2020, 2030], name="period")
    sample = _multi_period_sample()

    def fake_statistic(**kwargs):
        return sample

    fake_statistic.__name__ = "installed_capacity"

    plotter = StatisticPlotter(fake_statistic, network)
    fig, ax, g = plotter.bar()

    assert "period" in g.data.columns
    assert set(g.data["period"].unique()) == {"2020", "2030"}
    plt.close(fig)


def test_networks_query_filtering(network_collection):
    plotter = ChartGenerator(network_collection)
    data = network_collection.statistics.energy_balance()
    fig, ax, g = plotter.plot(
        data, "bar", x="carrier", y="value", facet_col="scenario", query="value > 1"
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)


def test_networks_stacking(network_collection):
    fig, ax, g = network_collection.statistics.supply.plot.bar(
        x="carrier", y="value", stacked=True, facet_col="scenario"
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)


def test_networks_plot_map(network_collection):
    with pytest.raises(NotImplementedError):
        network_collection.statistics.energy_balance.plot.map()


def test_statistics_map_transmission_flow_bus_carrier_non_zero(
    ac_dc_network_r, monkeypatch
):
    captured: dict[str, pd.Series | int] = {}

    def fake_draw_map(self, **kwargs):
        captured["line_flow"] = kwargs.get("line_flow", 0)
        captured["link_flow"] = kwargs.get("link_flow", 0)

    monkeypatch.setattr(MapPlotGenerator, "draw_map", fake_draw_map)

    ac_dc_network_r.statistics.energy_balance.plot.map(
        bus_carrier="AC",
        transmission_flow=True,
        geomap=False,
        draw_legend_circles=False,
        draw_legend_lines=False,
        draw_legend_arrows=False,
        draw_legend_patches=False,
    )

    line_flow = captured.get("line_flow", 0)
    link_flow = captured.get("link_flow", 0)

    line_non_zero = isinstance(line_flow, pd.Series) and (line_flow.abs() > 0).any()
    link_non_zero = isinstance(link_flow, pd.Series) and (link_flow.abs() > 0).any()

    assert line_non_zero or link_non_zero


class TestStatisticsPlotIssue1719:
    """Regression tests for the statistics plotting bug collection (issue #1719)."""

    @pytest.mark.parametrize("kind", ["box", "violin"])
    def test_distribution_defaults_to_time_series(self, ac_dc_network_r, kind):
        """#4: distribution plots use the full time series, not its aggregate."""
        plotter = ChartGenerator(ac_dc_network_r)
        derived = plotter.derive_statistic_parameters(
            "carrier", "value", method_name="capacity_factor", chart_type=kind
        )
        assert derived["groupby_time"] is False

        fig, ax, _ = getattr(ac_dc_network_r.statistics.capacity_factor.plot, kind)()
        # A real distribution draws multiple artists per category (violins as
        # collections, boxplots as patches), not a single flat mark.
        assert len(ax.collections) + len(ax.patches) > 0
        plt.close(fig)

    def test_static_statistic_keeps_scalar_for_distribution(self, ac_dc_network_r):
        """#4: static statistics without time support are left untouched."""
        plotter = ChartGenerator(ac_dc_network_r)
        derived = plotter.derive_statistic_parameters(
            "carrier", "value", method_name="installed_capacity", chart_type="box"
        )
        assert "groupby_time" not in derived

    def test_box_height_scales_with_row_count(self, ac_dc_network_r):
        """#5: box height grows with the number of categorical (name) rows."""
        plotter = ChartGenerator(ac_dc_network_r)

        def box_height(n_rows):
            data = pd.DataFrame(
                np.random.rand(n_rows, 8),  # noqa: NPY002
                index=pd.Index([f"bus {i}" for i in range(n_rows)], name="name"),
            )
            fig, *_ = plotter.plot(data, "box", x="value", y="name")
            height = fig.get_figheight()
            plt.close(fig)
            return height

        # Height grows with the number of rows on large networks, capped at 30in.
        assert box_height(60) > box_height(20) > box_height(5)
        assert box_height(500) == 30

    def test_box_height_scales_per_facet(self, ac_dc_network_r):
        """Faceted boxes scale height by rows per facet, not total rows."""
        plotter = ChartGenerator(ac_dc_network_r)
        names = [f"bus {i}" for i in range(60)]
        groups = ["a"] * 20 + ["b"] * 20 + ["c"] * 20
        index = pd.MultiIndex.from_arrays([names, groups], names=["name", "group"])
        data = pd.Series(np.random.rand(60), index=index)  # noqa: NPY002

        fig, *_ = plotter.plot(data, "box", x="value", y="name", facet_row="group")
        per_facet_height = 0.3 * 20 + 1
        assert fig.get_figheight() == pytest.approx(3 * per_facet_height)
        plt.close(fig)

    def test_prices_facet_by_bus_attribute(self, ac_dc_network_r):
        """Prices plots support faceting and grouping by static bus attributes."""
        n = ac_dc_network_r.copy()
        static = n.c.buses.static
        static["country"] = ["UK" if i % 2 else "DE" for i in range(len(static))]

        fig, ax, g = n.statistics.prices.plot.box(bus_carrier="AC", facet_row="country")
        countries = static.loc[static.carrier == "AC", "country"]
        assert g.axes.shape[0] == countries.nunique()
        plt.close(fig)

        fig, ax, g = n.statistics.prices.plot.bar(y="bus_carrier")
        labels = {t.get_text() for t in ax.get_yticklabels()}
        assert labels == set(static.carrier.unique())
        plt.close(fig)

    @pytest.mark.parametrize("kind", ["line", "scatter"])
    def test_line_scatter_use_carrier_palette(self, ac_dc_network_r, kind):
        """#6: line/scatter honor carrier colors via the palette."""
        import matplotlib.colors as mcolors

        n = ac_dc_network_r.copy()
        n.c.carriers.static.loc["gas", "color"] = "#ff0000"
        n.c.carriers.static.loc["wind", "color"] = "#0000ff"

        plotter = getattr(n.statistics.installed_capacity.plot, kind)
        fig, ax, _ = plotter(carrier=["gas", "wind"], color="carrier")
        used = {
            line.get_label(): mcolors.to_hex(line.get_color())
            for line in ax.lines
            if not line.get_label().startswith("_")
        }
        assert used.get("gas") == "#ff0000"
        assert used.get("wind") == "#0000ff"
        plt.close(fig)

    def test_col_wrap_wraps_static_facets(self, ac_dc_network_r):
        """#11: col_wrap wraps facets instead of crashing."""
        fig, _, g = ac_dc_network_r.statistics.installed_capacity.plot.bar(
            facet_col="bus_carrier", col_wrap=1
        )
        assert g.axes.ndim == 1
        assert len(g.axes) == 2
        plt.close(fig)
