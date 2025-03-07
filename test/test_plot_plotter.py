import matplotlib.pyplot as plt
import pytest

from pypsa.plot.accessor import PlotAccessor


@pytest.mark.parametrize("stat_func", PlotAccessor._methods)
def test_bar_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.plot, stat_func)
    plotter.bar()
    plt.close()

    with pytest.raises(TypeError):
        plotter().bar()


@pytest.mark.parametrize("stat_func", PlotAccessor._methods)
def test_line_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.plot, stat_func)
    plotter.line()
    plt.close()

    with pytest.raises(TypeError):
        plotter().line()


@pytest.mark.parametrize("stat_func", PlotAccessor._methods)
def test_area_plot(ac_dc_network_r, stat_func):
    plotter = getattr(ac_dc_network_r.plot, stat_func)
    plotter.area()
    plt.close()

    with pytest.raises(TypeError):
        plotter().area()
