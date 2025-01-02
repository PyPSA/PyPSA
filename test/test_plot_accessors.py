import pandas as pd

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

    # Test line plot methods
    plot.line.supply()
    plot.line.withdrawal()
    plot.line.energy_balance()

    # Test area plot methods
    plot.area.supply()
    plot.area.withdrawal()
    plot.area.energy_balance()
