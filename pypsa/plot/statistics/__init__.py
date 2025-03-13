"""
Statistic plots for PyPSA.

This module contains all plotting functions which can be used on top of the
statistics functions of :mod:`pypsa.statistics`.
"""

from pypsa.plot.statistics.charts import (
    AreaPlotGenerator,
    BarPlotGenerator,
    LinePlotGenerator,
)
from pypsa.plot.statistics.maps import MapPlotGenerator

__all__ = [
    "BarPlotGenerator",
    "LinePlotGenerator",
    "AreaPlotGenerator",
    "MapPlotGenerator",
]
