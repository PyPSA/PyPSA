from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

from pypsa.plot.maps import plot
from pypsa.plot.statistics.charts import (
    AreaPlotAccessor,
    BarPlotAccessor,
    ChartPlotTypeAccessor,
    LinePlotAccessor,
)
from pypsa.plot.statistics.maps import MapPlotAccessor

if TYPE_CHECKING:
    from pypsa import Network


class PlotAccessor:
    """
    Main plot accessor providing access to different plot types
    """

    _n: Network
    _base: ChartPlotTypeAccessor
    map: MapPlotAccessor
    bar: BarPlotAccessor
    line: LinePlotAccessor
    area: AreaPlotAccessor

    def __init__(self: PlotAccessor, n: Network) -> None:
        self._n = n
        self._base = ChartPlotTypeAccessor(n)
        self.map = MapPlotAccessor(n)
        self.bar = BarPlotAccessor(n)
        self.line = LinePlotAccessor(n)
        self.area = AreaPlotAccessor(n)

    @wraps(plot)
    def __call__(self: PlotAccessor, *args: Any, **kwargs: Any) -> Any:
        """Default plot method (map)"""
        return plot(self._n, *args, **kwargs)
