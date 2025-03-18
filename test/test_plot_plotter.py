import matplotlib.pyplot as plt
import pytest

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
