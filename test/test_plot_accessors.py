import pandas as pd
import pytest
import seaborn.objects as so

from pypsa.consistency import ConsistencyError
from pypsa.plot.accessors import (
    AreaPlotAccessor,
    BarPlotAccessor,
    ChartPlotTypeAccessor,
    LinePlotAccessor,
)


def test_to_long_format_static(ac_dc_network_r):
    """Test the _to_long_format method with optimal_capacity data."""
    # Create the accessor instance
    accessor = ChartPlotTypeAccessor(ac_dc_network_r)

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
    accessor = ChartPlotTypeAccessor(ac_dc_network_r)

    # Get installed capacity data from statistics
    data = ac_dc_network_r.statistics.energy_balance()

    # Convert to long format
    long_data = accessor._to_long_format(data)

    # Check the output structure
    assert isinstance(long_data, pd.DataFrame)
    assert set(long_data.columns) == {"component", "carrier", "bus_carrier", "value"}


def test_bar_plotter_validation(ac_dc_network_r):
    """Test BarPlotAccessor validation logic"""
    plotter = BarPlotAccessor(ac_dc_network_r)

    # Test valid data
    valid_data = pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2]})
    assert plotter._validate(valid_data).equals(valid_data)


def test_line_plotter_validation(ac_dc_network_r):
    """Test LinePlotAccessor validation logic"""
    plotter = LinePlotAccessor(ac_dc_network_r)

    # Test with snapshot column
    data = pd.DataFrame({"snapshot": ["2025-01-01", "2025-01-02"], "value": [1, 2]})
    validated = plotter._validate(data)
    assert pd.api.types.is_datetime64_any_dtype(validated["snapshot"])

    # Test without snapshot column
    data = pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2]})
    assert plotter._validate(data).equals(data)


def test_area_plotter_validation(ac_dc_network_r):
    """Test AreaPlotAccessor validation logic"""
    plotter = AreaPlotAccessor(ac_dc_network_r)

    # Test with snapshot column
    data = pd.DataFrame({"snapshot": ["2025-01-01", "2025-01-02"], "value": [1, 2]})
    validated = plotter._validate(data)
    assert pd.api.types.is_datetime64_any_dtype(validated["snapshot"])

    # Test without snapshot column
    data = pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2]})
    assert plotter._validate(data).equals(data)


def test_plot_accessor_creation(ac_dc_network_r):
    """Test creation of plot accessor and its components"""
    plot = ac_dc_network_r.plot

    assert hasattr(plot, "bar")
    assert hasattr(plot, "line")
    assert hasattr(plot, "area")
    assert hasattr(plot, "map")

    assert isinstance(plot.bar, BarPlotAccessor)
    assert isinstance(plot.line, LinePlotAccessor)
    assert isinstance(plot.area, AreaPlotAccessor)


def test_process_data_for_stacking(ac_dc_network_r):
    """Test the _process_data_for_stacking method"""
    plotter = ChartPlotTypeAccessor(ac_dc_network_r)

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
    plotter = ChartPlotTypeAccessor(ac_dc_network_r)
    data = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="carrier"))
    plot = plotter._base_plot(data, x="carrier", y="value", color="carrier")
    assert isinstance(plot, so.Plot)
    assert plot._scales["color"] is not None


def test_base_plot_faceting(ac_dc_network_r):
    """Test faceting functionality in base plot"""
    plotter = ChartPlotTypeAccessor(ac_dc_network_r)
    data = (
        pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2], "region": ["r1", "r2"]})
        .set_index(["carrier", "region"])
        .value
    )
    plot = plotter._base_plot(data, x="carrier", y="value", col="region")
    assert isinstance(plot, so.Plot)
    assert plot._facet_spec is not None


def test_derive_statistic_parameters(ac_dc_network_r):
    """Test derivation of statistic parameters"""
    plotter = ChartPlotTypeAccessor(ac_dc_network_r)

    # Test with default parameters
    groupby, agg_comp, agg_time = plotter._derive_statistic_parameters(
        "carrier", "value", "carrier"
    )
    assert isinstance(groupby, list)
    assert isinstance(agg_comp, bool)
    assert isinstance(agg_time, bool | str)

    # Test with custom parameters
    stats_opts = {
        "groupby": ["carrier"],
        "aggregate_across_components": True,
        "aggregate_time": "mean",
    }
    groupby, agg_comp, agg_time = plotter._derive_statistic_parameters(
        "carrier", "value", stats_opts=stats_opts
    )
    assert groupby == ["carrier"]
    assert agg_comp is True
    assert agg_time == "mean"


def test_get_carrier_colors_and_labels(ac_dc_network_r):
    """Test carrier colors and labels retrieval"""
    plotter = ChartPlotTypeAccessor(ac_dc_network_r)

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


def test_validate_data_requirements(ac_dc_network_r):
    """Test data validation requirements"""
    plotter = ChartPlotTypeAccessor(ac_dc_network_r)

    # Test with missing value column
    invalid_data = pd.DataFrame({"carrier": ["a", "b"]})
    with pytest.raises(ValueError, match="Data must contain 'value' column"):
        plotter._validate(invalid_data)

    # Test with valid data
    valid_data = pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2]})
    validated = plotter._validate(valid_data)
    assert "value" in validated.columns
    assert isinstance(validated["carrier"].dtype, pd.CategoricalDtype)


def test_query_filtering(ac_dc_network_r):
    """Test query filtering in plots"""
    plotter = ChartPlotTypeAccessor(ac_dc_network_r)
    data = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="carrier"))

    filtered_plot = plotter._base_plot(data, x="carrier", y="value", query="value > 1")
    assert isinstance(filtered_plot, so.Plot)


def test_consistency_checks(ac_dc_network_r):
    """Test plotting consistency checks"""
    plotter = ChartPlotTypeAccessor(ac_dc_network_r)

    # Test with missing carrier colors
    with pytest.raises(ConsistencyError):
        n = ac_dc_network_r.copy()
        plotter = ChartPlotTypeAccessor(n)
        n.carriers.color = pd.Series()
        plotter._base_plot(pd.DataFrame(), "carrier", "value")


def test_plot_methods(ac_dc_network_r):
    """Test main plotting methods don't raise errors"""
    plot = ac_dc_network_r.plot

    # Test bar plot methods
    plot.bar.optimal_capacity()
    plot.bar.installed_capacity()
    plot.bar.energy_balance()
    plot.bar.capex()
    plot.bar.opex()
    plot.bar.revenue()
    plot.bar.market_value()

    # Test line plot methods
    plot.line.supply()
    plot.line.withdrawal()
    plot.line.energy_balance()
    plot.line.capacity_factor()
    plot.line.curtailment()
    plot.line.transmission()

    # Test area plot methods
    plot.area.supply()
    plot.area.withdrawal()
    plot.area.energy_balance()
    plot.area.capacity_factor()
    plot.area.curtailment()
    plot.area.transmission()


def test_time_aggregation_behavior(ac_dc_network_r):
    """Test time aggregation behavior for different plotters"""
    n = ac_dc_network_r

    assert n.plot.bar._time_aggregation == "sum"
    assert n.plot.line._time_aggregation is False
    assert n.plot.area._time_aggregation is False


def test_stacking_and_dodging(ac_dc_network_r):
    """Test stacking and dodging options in bar plots"""
    n = ac_dc_network_r
    stacked_plot = n.plot.bar.supply(x="carrier", y="value", stacked=True)
    assert isinstance(stacked_plot, so.Plot)

    # Test dodged plot
    dodged_plot = n.plot.bar.supply(x="carrier", y="value", dodged=True)
    assert isinstance(dodged_plot, so.Plot)


def test_line_plot_resampling(ac_dc_network_r):
    """Test resampling functionality in line plots"""
    n = ac_dc_network_r
    n.plot.line.supply(resample="1D", x="snapshot")


def test_plotter_default_x_values(ac_dc_network_r):
    """Test static and dynamic default x-values for different plotters"""
    n = ac_dc_network_r

    # Test class defaults
    assert n.plot.bar._default_static_x == "carrier"
    assert n.plot.bar._default_dynamic_x == "carrier"

    assert n.plot.line._default_static_x == "carrier"
    assert n.plot.line._default_dynamic_x == "snapshot"

    assert n.plot.area._default_static_x == "carrier"
    assert n.plot.area._default_dynamic_x == "snapshot"

    # Test defaults for static plots
    static_bar = n.plot.bar.optimal_capacity()
    static_line = n.plot.line.optimal_capacity()
    static_area = n.plot.area.optimal_capacity()

    assert isinstance(static_bar, so.Plot)
    assert isinstance(static_line, so.Plot)
    assert isinstance(static_area, so.Plot)

    # Test defaults for dynamic plots
    dynamic_bar = n.plot.bar.supply()
    dynamic_line = n.plot.line.supply()
    dynamic_area = n.plot.area.supply()

    assert isinstance(dynamic_bar, so.Plot)
    assert isinstance(dynamic_line, so.Plot)
    assert isinstance(dynamic_area, so.Plot)
