# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Statistic plots for PyPSA.

This module contains all plotting functions which can be used on top of the
statistics functions of [`pypsa.Network.statistics`][].
"""

from pypsa.plot.statistics.charts import ChartGenerator
from pypsa.plot.statistics.maps import MapPlotGenerator

__all__ = [
    "ChartGenerator",
    "MapPlotGenerator",
]
