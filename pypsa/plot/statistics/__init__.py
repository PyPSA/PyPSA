"""
Statistic plots for PyPSA.

This module contains all plotting functions which can be used on top of the
statistics functions of :mod:`pypsa.statistics`.
"""

from pypsa.plot.maps import MapPlotGenerator
from pypsa.plot.statistics.charts import (
    AreaPlotGenerator,
    BarPlotGenerator,
    LinePlotGenerator,
)

__all__ = [
    "BarPlotGenerator",
    "LinePlotGenerator",
    "AreaPlotGenerator",
    "MapPlotGenerator",
]
