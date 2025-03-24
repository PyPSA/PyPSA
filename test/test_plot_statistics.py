import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from pypsa.consistency import ConsistencyError
from pypsa.plot.statistics.charts import ChartGenerator
from pypsa.statistics.expressions import StatisticsAccessor

# Set random seed for reproducibility
np.random.seed(42)  # noqa: NPY002

plt.rcdefaults()
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["figure.dpi"] = 100


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
@pytest.mark.mpl_image_compare(tolerance=20)
def test_simple_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    fig, _, _ = plotter.plot()

    return fig


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
@pytest.mark.mpl_image_compare(tolerance=20)
def test_bar_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    fig, _, _ = plotter.plot.bar()

    return fig


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
@pytest.mark.mpl_image_compare(tolerance=20)
def test_line_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    fig, _, _ = plotter.plot.line()

    return fig


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
@pytest.mark.mpl_image_compare(tolerance=20)
def test_area_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    fig, _, _ = plotter.plot.area()

    return fig


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
# @pytest.mark.mpl_image_compare(tolerance=20) # TODO find better way to compare
def test_map_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)

    fig, _ = plotter.plot.map()

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


def test_consistency_checks(ac_dc_network_r):
    """Test plotting consistency checks"""
    plotter = ChartGenerator(ac_dc_network_r)

    # Test with missing carrier colors
    with pytest.raises(ConsistencyError):
        n = ac_dc_network_r.copy()
        plotter = ChartGenerator(n)
        n.carriers.color = pd.Series()
        plotter.plot(data=pd.DataFrame(), kind="area", x="carrier", y="value")


def test_stacking(ac_dc_network_r):
    """Test stacking options in bar plots"""
    n = ac_dc_network_r
    fig, ax, g = n.statistics.supply.plot.bar(x="carrier", y="value", stacked=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)
