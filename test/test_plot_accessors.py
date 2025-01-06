import pandas as pd
import seaborn.objects as so

from pypsa.plot.accessors import (
    AreaPlotter,
    BarPlotter,
    BasePlotTypeAccessor,
    LinePlotter,
)


def test_to_long_format_static(ac_dc_network_r):
    """Test the _to_long_format method with optimal_capacity data."""
    # Create the accessor instance
    accessor = BasePlotTypeAccessor(ac_dc_network_r)

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
    accessor = BasePlotTypeAccessor(ac_dc_network_r)

    # Get installed capacity data from statistics
    data = ac_dc_network_r.statistics.energy_balance()

    # Convert to long format
    long_data = accessor._to_long_format(data)

    # Check the output structure
    assert isinstance(long_data, pd.DataFrame)
    assert set(long_data.columns) == {"component", "carrier", "bus_carrier", "value"}


def test_bar_plotter_validation(ac_dc_network_r):
    """Test BarPlotter validation logic"""
    plotter = BarPlotter(ac_dc_network_r)

    # Test valid data
    valid_data = pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2]})
    assert plotter._validate(valid_data).equals(valid_data)


def test_line_plotter_validation(ac_dc_network_r):
    """Test LinePlotter validation logic"""
    plotter = LinePlotter(ac_dc_network_r)

    # Test with snapshot column
    data = pd.DataFrame({"snapshot": ["2025-01-01", "2025-01-02"], "value": [1, 2]})
    validated = plotter._validate(data)
    assert pd.api.types.is_datetime64_any_dtype(validated["snapshot"])

    # Test without snapshot column
    data = pd.DataFrame({"carrier": ["a", "b"], "value": [1, 2]})
    assert plotter._validate(data).equals(data)


def test_area_plotter_validation(ac_dc_network_r):
    """Test AreaPlotter validation logic"""
    plotter = AreaPlotter(ac_dc_network_r)

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
    assert hasattr(plot, "maps")

    assert isinstance(plot.bar, BarPlotter)
    assert isinstance(plot.line, LinePlotter)
    assert isinstance(plot.area, AreaPlotter)


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


def test_plot_methods_with_bus_carrier(ac_dc_network_r):
    """Test plotting methods with bus_carrier filtering"""
    plot = ac_dc_network_r.plot

    # Test with AC bus carrier
    plot.bar.optimal_capacity(bus_carrier="AC")
    plot.line.supply(bus_carrier="AC")
    plot.area.energy_balance(bus_carrier="AC")

    # Test with DC bus carrier
    plot.bar.installed_capacity(bus_carrier="DC")
    plot.line.withdrawal(bus_carrier="DC")
    plot.area.transmission(bus_carrier="DC")


def test_plot_methods_with_groupby(ac_dc_network_r):
    """Test plotting methods with different groupby options"""
    plot = ac_dc_network_r.plot

    # Test single groupby
    plot.bar.optimal_capacity(groupby=["carrier"])
    plot.line.supply(groupby=["bus_carrier"])
    plot.area.energy_balance(groupby=["carrier", "bus_carrier"])

    # Test multiple groupby
    plot.bar.installed_capacity(groupby=["carrier", "bus_carrier"])
    plot.line.withdrawal(groupby=["carrier", "bus_carrier"])
    plot.area.transmission(groupby=["bus_carrier"])


def test_plot_methods_with_aggregation(ac_dc_network_r):
    """Test plotting methods with aggregation options"""
    plot = ac_dc_network_r.plot

    # Test time aggregation
    plot.line.supply(aggregate_time="mean")
    plot.area.withdrawal(aggregate_time="sum")

    # Test component aggregation
    plot.bar.optimal_capacity(aggregate_across_components=False)
    plot.line.energy_balance(aggregate_across_components=True)


def test_plot_methods_with_nice_names(ac_dc_network_r):
    """Test plotting methods with nice names"""
    plot = ac_dc_network_r.plot

    # Test with nice names
    plot.bar.optimal_capacity(nice_names=True)
    plot.line.supply(nice_names=True)
    plot.area.energy_balance(nice_names=True)

    # Test without nice names
    plot.bar.installed_capacity(nice_names=False)
    plot.line.withdrawal(nice_names=False)
    plot.area.transmission(nice_names=False)


def test_plot_methods_with_plot_kwargs(ac_dc_network_r):
    """Test plotting methods with additional plot kwargs"""
    plot = ac_dc_network_r.plot

    # Test bar plot kwargs
    plot.bar.optimal_capacity(stacked=True, orientation="horizontal")
    plot.bar.installed_capacity(stacked=False, orientation="vertical")

    # Test line plot kwargs
    plot.line.supply(resample="h")  # Hourly resampling
    plot.line.withdrawal(resample="D")  # Daily resampling

    # Test area plot kwargs
    plot.area.energy_balance(stacked=True)
    plot.area.transmission(stacked=False)


def test_base_plot_type_accessor_initialization(ac_dc_network_r):
    """Test BasePlotTypeAccessor initialization"""
    accessor = BasePlotTypeAccessor(ac_dc_network_r)

    assert hasattr(accessor, "_network")
    assert hasattr(accessor, "_statistics")
    assert accessor._time_aggregation is False


def test_to_long_format_series(ac_dc_network_r):
    """Test _to_long_format with Series input"""
    accessor = BasePlotTypeAccessor(ac_dc_network_r)

    # Create test series with multiindex
    index = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["carrier", "bus"])
    series = pd.Series([10, 20], index=index)

    result = accessor._to_long_format(series)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"carrier", "bus", "value"}


def test_check_plotting_consistency(ac_dc_network_r):
    """Test _check_plotting_consistency method"""
    accessor = BasePlotTypeAccessor(ac_dc_network_r)

    # Should not raise errors with valid network
    accessor._check_plotting_consistency()


def test_get_carrier_colors(ac_dc_network_r):
    """Test _get_carrier_colors method"""
    accessor = BasePlotTypeAccessor(ac_dc_network_r)

    # Test with carrier data
    colors = accessor._get_carrier_colors()

    assert isinstance(colors, dict)
    assert "wind" in colors
    assert "gas" in colors
    assert "-" in colors  # Test default gray color


def test_get_carrier_labels(ac_dc_network_r):
    """Test _get_carrier_labels method"""
    accessor = BasePlotTypeAccessor(ac_dc_network_r)

    # Test with nice names
    labels = accessor._get_carrier_labels(nice_names=True)
    assert isinstance(labels, dict)

    # Test without nice names
    labels = accessor._get_carrier_labels(nice_names=False)
    assert labels == {}


def test_create_base_plot(ac_dc_network_r):
    """Test _create_base_plot method"""
    accessor = BasePlotTypeAccessor(ac_dc_network_r)

    # Test with simple data
    data = pd.DataFrame(
        {"carrier": ["wind", "gas"], "value": [10, 20], "bus_carrier": ["AC", "AC"]}
    )

    plot = accessor._create_base_plot(data, x="carrier", y="value")
    assert isinstance(plot, so.Plot)


def test_bar_plotter_plot(ac_dc_network_r):
    """Test BarPlotter._plot method"""
    plotter = BarPlotter(ac_dc_network_r)

    # Test with simple data
    data = pd.DataFrame({"carrier": ["wind", "gas"], "value": [10, 20]})

    plot = plotter._base_plot(data)
    assert plot is not None


def test_line_plotter_plot(ac_dc_network_r):
    """Test LinePlotter._plot method"""
    plotter = LinePlotter(ac_dc_network_r)

    # Test with time series data
    data = pd.DataFrame(
        {
            "snapshot": pd.date_range("2023-01-01", periods=2),
            "carrier": ["wind", "gas"],
            "value": [10, 20],
        }
    )

    plot = plotter._base_plot(data)
    assert plot is not None


def test_area_plotter_plot(ac_dc_network_r):
    """Test AreaPlotter._plot method"""
    plotter = AreaPlotter(ac_dc_network_r)

    # Test with time series data
    data = pd.DataFrame(
        {
            "snapshot": pd.date_range("2023-01-01", periods=2),
            "carrier": ["wind", "gas"],
            "value": [10, 20],
        }
    )

    plot = plotter._base_plot(data)
    assert plot is not None


def test_plot_accessor_call(ac_dc_network_r):
    """Test PlotAccessor.__call__ method"""
    plot = ac_dc_network_r.plot

    # Test map plotting
    result = plot()
    assert result is not None
