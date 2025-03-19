import hashlib
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns

from pypsa.consistency import ConsistencyError
from pypsa.plot.statistics.charts import (
    AreaPlotGenerator,
    BarPlotGenerator,
    LinePlotGenerator,
)
from pypsa.statistics import StatisticsAccessor

PLOT_HASHES = {
    "capex": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "installed_capex": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "expanded_capex": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "optimal_capacity": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "installed_capacity": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "expanded_capacity": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "opex": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "supply": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "withdrawal": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "transmission": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "energy_balance": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "curtailment": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "capacity_factor": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "revenue": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
    "market_value": {
        "plot": "xxx",
        "bar": "xxx",
        "line": "xxx",
        "area": "xxx",
        "map": "xxx",
    },
}


def get_object_hash(obj):
    """Generate a hash for any picklable Python object."""
    pickled_obj = pickle.dumps(obj)
    return hashlib.md5(pickled_obj).hexdigest()


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_simple_plot(pytestconfig, ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    plot = plotter.plot()

    plot_hash = get_object_hash(plot)
    assert plot_hash == PLOT_HASHES[stat_func]["plot"], (
        f"Plot hash mismatch for {stat_func}. If this is expected, "
        "update the PLOT_HASHES dictionary."
    )

    if pytestconfig.getoption("--save-plots"):
        Path("test_plots_output").mkdir(exist_ok=True)
        plot.save("test_plots_output/" + stat_func + "-simple.png")

    plt.close()


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_bar_plot(pytestconfig, ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    plot = plotter.plot.bar()

    plot_hash = get_object_hash(plot)
    assert plot_hash == PLOT_HASHES[stat_func]["plot"], (
        f"Plot hash mismatch for {stat_func}. If this is expected, "
        "update the PLOT_HASHES dictionary."
    )

    if pytestconfig.getoption("--save-plots"):
        Path("test_plots_output").mkdir(exist_ok=True)
        plot.save("test_plots_output/" + stat_func + "-bar.png")

    plt.close()


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_line_plot(pytestconfig, ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    plot = plotter.plot.line()

    plot_hash = get_object_hash(plot)
    assert plot_hash == PLOT_HASHES[stat_func]["plot"], (
        f"Plot hash mismatch for {stat_func}. If this is expected, "
        "update the PLOT_HASHES dictionary."
    )

    if pytestconfig.getoption("--save-plots"):
        Path("test_plots_output").mkdir(exist_ok=True)
        plot.save("test_plots_output/" + stat_func + "-line.png")
    plt.close()


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_area_plot(pytestconfig, ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    plot = plotter.plot.area()

    plot_hash = get_object_hash(plot)
    assert plot_hash == PLOT_HASHES[stat_func]["plot"], (
        f"Plot hash mismatch for {stat_func}. If this is expected, "
        "update the PLOT_HASHES dictionary."
    )

    if pytestconfig.getoption("--save-plots"):
        Path("test_plots_output").mkdir(exist_ok=True)
        plot.save("test_plots_output/" + stat_func + "-area.png")
    plt.close()


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_map_plot(pytestconfig, ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    plot = plotter.plot.area()

    plot_hash = get_object_hash(plot)
    assert plot_hash == PLOT_HASHES[stat_func]["plot"], (
        f"Plot hash mismatch for {stat_func}. If this is expected, "
        "update the PLOT_HASHES dictionary."
    )

    if pytestconfig.getoption("--save-plots"):
        Path("test_plots_output").mkdir(exist_ok=True)
        plot.save("test_plots_output/" + stat_func + "-map.png")
    plt.close()


def test_to_long_format_static(ac_dc_network_r):
    """Test the _to_long_format method with optimal_capacity data."""
    # Create the accessor instance
    accessor = BarPlotGenerator(ac_dc_network_r)

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
    accessor = BarPlotGenerator(ac_dc_network_r)

    # Get installed capacity data from statistics
    data = ac_dc_network_r.statistics.energy_balance()

    # Convert to long format
    long_data = accessor._to_long_format(data)

    # Check the output structure
    assert isinstance(long_data, pd.DataFrame)
    assert set(long_data.columns) == {"component", "carrier", "bus_carrier", "value"}


def test_bar_plotter_validation(ac_dc_network_r):
    """Test BarPlotGenerator validation logic"""
    plotter = BarPlotGenerator(ac_dc_network_r)

    # Test valid data
    valid_data = pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2]})
    assert plotter._validate(valid_data).equals(valid_data)


def test_line_plotter_validation(ac_dc_network_r):
    """Test LinePlotGenerator validation logic"""
    plotter = LinePlotGenerator(ac_dc_network_r)

    # Test with snapshot column
    data = pd.DataFrame({"snapshot": ["2025-01-01", "2025-01-02"], "value": [1, 2]})
    validated = plotter._validate(data)
    assert pd.api.types.is_datetime64_any_dtype(validated["snapshot"])

    # Test without snapshot column
    data = pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2]})
    assert plotter._validate(data).equals(data)


def test_area_plotter_validation(ac_dc_network_r):
    """Test AreaPlotGenerator validation logic"""
    plotter = AreaPlotGenerator(ac_dc_network_r)

    # Test with snapshot column
    data = pd.DataFrame({"snapshot": ["2025-01-01", "2025-01-02"], "value": [1, 2]})
    validated = plotter._validate(data)
    assert pd.api.types.is_datetime64_any_dtype(validated["snapshot"])

    # Test without snapshot column
    data = pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2]})
    assert plotter._validate(data).equals(data)


# def test_derive_statistic_parameters(ac_dc_network_r):
#     """Test derivation of statistic parameters"""
#     # TODO rewrite once function is updated
#     plotter = ChartGenerator(ac_dc_network_r)

#     # Test with default parameters
#     groupby, agg_comp, agg_time = plotter.derive_statistic_parameters(
#         "carrier", "value", "carrier"
#     )
#     assert isinstance(groupby, list)
#     assert isinstance(agg_comp, bool)
#     assert isinstance(agg_time, bool | str)

#     # Test with custom parameters
#     stats_opts = {
#         "groupby": ["carrier"],
#         "aggregate_across_components": True,
#         "aggregate_time": "mean",
#     }
#     groupby, agg_comp, agg_time = plotter.derive_statistic_parameters(
#         "carrier", "value", stats_opts=stats_opts
#     )
#     assert groupby == ["carrier"]
#     assert agg_comp is True
#     assert agg_time == "mean"


def test_get_carrier_colors_and_labels(ac_dc_network_r):
    """Test carrier colors and labels retrieval"""
    plotter = BarPlotGenerator(ac_dc_network_r)

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
    plotter = BarPlotGenerator(ac_dc_network_r)
    data = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="carrier"))

    fig, ax, g = plotter._base_plot(
        data, "bar", x="carrier", y="value", query="value > 1"
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)


def test_consistency_checks(ac_dc_network_r):
    """Test plotting consistency checks"""
    plotter = BarPlotGenerator(ac_dc_network_r)

    # Test with missing carrier colors
    with pytest.raises(ConsistencyError):
        n = ac_dc_network_r.copy()
        plotter = BarPlotGenerator(n)
        n.carriers.color = pd.Series()
        plotter._base_plot(data=pd.DataFrame(), kind="area", x="carrier", y="value")


def test_stacking(ac_dc_network_r):
    """Test stacking options in bar plots"""
    n = ac_dc_network_r
    fig, ax, g = n.statistics.supply.plot.bar(x="carrier", y="value", stacked=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert isinstance(g, sns.FacetGrid)


def test_line_plot_resampling(ac_dc_network_r):
    """Test resampling functionality in line plots"""
    n = ac_dc_network_r
    n.statistics.supply.plot.line(resample="1D", x="snapshot")
