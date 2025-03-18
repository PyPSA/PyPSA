import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn.objects as so

from pypsa.consistency import ConsistencyError
from pypsa.plot.statistics.charts import (
    AreaPlotGenerator,
    BarPlotGenerator,
    LinePlotGenerator,
)
from pypsa.statistics.expressions import StatisticsAccessor


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_bar_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    plotter.bar()
    plt.close()


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_line_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    plotter.line()
    plt.close()


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_area_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    plotter.area()
    plt.close()


@pytest.mark.parametrize("stat_func", StatisticsAccessor._methods)
def test_map_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.statistics, stat_func)
    plotter.area()
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


def test_process_data_for_stacking(ac_dc_network_r):
    """Test the _process_data_for_stacking method"""
    plotter = BarPlotGenerator(ac_dc_network_r)

    # Test with positive and negative values
    data = pd.DataFrame(
        {
            "carrier": ["a", "a", "b", "b"],
            "value": [1, -1, 2, -2],
        }
    )
    stacked_data = plotter._process_data_for_stacking(data, "carrier")
    assert len(stacked_data) == 4
    assert set(stacked_data["value"]) == {1, -1, 2, -2, -1, -2}


def test_base_plot_color_scale(ac_dc_network_r):
    """Test color scale application in base plot"""
    plotter = BarPlotGenerator(ac_dc_network_r)
    data = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="carrier"))
    plot = plotter._base_plot(data, x="carrier", y="value", color="carrier")
    assert isinstance(plot, so.Plot)
    assert plot._scales["color"] is not None


def test_base_plot_faceting(ac_dc_network_r):
    """Test faceting functionality in base plot"""
    plotter = BarPlotGenerator(ac_dc_network_r)
    data = (
        pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2], "region": ["r1", "r2"]})
        .set_index(["carrier", "region"])
        .value
    )
    plot = plotter._base_plot(data, x="carrier", y="value", facet_col="region")
    assert isinstance(plot, so.Plot)
    assert plot._facet_spec is not None


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

    colors = plotter._get_carrier_colors()
    assert isinstance(colors, dict)
    assert "-" in colors
    assert None in colors

    labels = plotter._get_carrier_labels()
    assert isinstance(labels, dict)

    # Test with nice_names=False
    labels_raw = plotter._get_carrier_labels(nice_names=False)
    assert isinstance(labels_raw, dict)
    assert len(labels_raw) == 0


def test_query_filtering(ac_dc_network_r):
    """Test query filtering in plots"""
    plotter = BarPlotGenerator(ac_dc_network_r)
    data = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="carrier"))

    filtered_plot = plotter._base_plot(data, x="carrier", y="value", query="value > 1")
    assert isinstance(filtered_plot, so.Plot)


def test_consistency_checks(ac_dc_network_r):
    """Test plotting consistency checks"""
    plotter = BarPlotGenerator(ac_dc_network_r)

    # Test with missing carrier colors
    with pytest.raises(ConsistencyError):
        n = ac_dc_network_r.copy()
        plotter = BarPlotGenerator(n)
        n.carriers.color = pd.Series()
        plotter._base_plot(data=pd.DataFrame(), x="carrier", y="value")


def test_stacking_and_dodging(ac_dc_network_r):
    """Test stacking and dodging options in bar plots"""
    n = ac_dc_network_r
    stacked_plot = n.statistics.supply.bar(x="carrier", y="value", stacked=True)
    assert isinstance(stacked_plot, so.Plot)

    # Test dodged plot
    dodged_plot = n.statistics.supply.bar(x="carrier", y="value", dodged=True)
    assert isinstance(dodged_plot, so.Plot)


def test_line_plot_resampling(ac_dc_network_r):
    """Test resampling functionality in line plots"""
    n = ac_dc_network_r
    n.statistics.supply.line(resample="1D", x="snapshot")
