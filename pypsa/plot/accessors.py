from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import pandas as pd

from pypsa import Network


class BasePlotTypeAccessor:
    """Base class containing shared statistics methods"""

    _network: Network
    _statistics: Any
    _time_aggregation: str | bool

    def __init__(self: BasePlotTypeAccessor, n: Network) -> None:
        self._network = n
        self._statistics = n.statistics
        self._time_aggregation = False

    def _to_long_format(
        self: BasePlotTypeAccessor, data: pd.DataFrame | pd.Series
    ) -> pd.DataFrame:
        """
        Convert data to long format suitable for plotting.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series
            Input data from statistics functions, typically with multiindex

        Returns
        -------
        pd.DataFrame
            Long format DataFrame with multiindex levels as columns and values in 'value' column
        """
        if isinstance(data, pd.Series):
            return data.rename("value").reset_index()
        else:
            return data.fillna(0).melt(ignore_index=False).reset_index()

    def _validate(self: BasePlotTypeAccessor, data: pd.DataFrame) -> pd.DataFrame:
        """Validation method to be implemented by subclasses"""
        raise NotImplementedError

    def _plot(self: BasePlotTypeAccessor, data: pd.DataFrame, **kwargs: Any) -> Any:
        """Plot method to be implemented by subclasses"""
        raise NotImplementedError

    def _process_data(
        self: BasePlotTypeAccessor, data: pd.DataFrame, **kwargs: Any
    ) -> Any:
        """Common data processing pipeline"""
        data = self._validate(data)
        return self._plot(data, **kwargs)

    # Shared statistics methods
    def optimal_capacity(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot optimal capacity"""
        data = self._statistics.optimal_capacity(groupby=groupby)
        return self._process_data(data, **kwargs)

    def energy_balance(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot energy balance"""
        data = self._statistics.energy_balance(
            groupby=groupby, aggregate_time=self._time_aggregation
        )
        return self._process_data(data, **kwargs)

    def supply(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot supply data"""
        if self._time_aggregation is False:
            data = self._statistics.supply(groupby=groupby, aggregate_time=False)
            return self._process_data(data, **kwargs)
        raise ValueError(f"{self.__class__.__name__} cannot handle time series data")

    # ... other shared statistics methods


class MapPlotAccessor(BasePlotTypeAccessor):
    """Map-specific plot implementation"""

    _default_projection: ClassVar[str] = "euro_scope"

    def __init__(self: MapPlotAccessor, n: Network) -> None:
        super().__init__(n)
        self._time_aggregation = "sum"

    def _validate(self: MapPlotAccessor, data: pd.DataFrame) -> pd.DataFrame:
        """Implement map-specific data validation"""
        if not all(
            hasattr(self._network.buses[comp], "x")
            for comp in data.index.get_level_values("component")
        ):
            raise ValueError("Not all components have spatial coordinates")
        return data

    def _plot(
        self: MapPlotAccessor,
        data: pd.DataFrame,
        projection: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Implement map plotting logic"""
        projection = projection or self._default_projection
        # Map plotting implementation
        pass


class BarPlotAccessor(BasePlotTypeAccessor):
    """Bar plot-specific implementation"""

    _default_orientation: ClassVar[str] = "vertical"

    def __init__(self: BarPlotAccessor, n: Network) -> None:
        super().__init__(n)
        self._time_aggregation = "sum"

    def _validate(self: BarPlotAccessor, data: pd.DataFrame) -> pd.DataFrame:
        """Implement bar-specific data validation"""
        if data.index.nlevels < 1:
            raise ValueError("Data must have at least one index level for bar plots")
        return data

    def _plot(
        self: BarPlotAccessor,
        data: pd.DataFrame,
        stacked: bool = False,
        orientation: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Implement bar plotting logic"""
        orientation = orientation or self._default_orientation
        # Bar plotting implementation
        pass


class LinePlotAccessor(BasePlotTypeAccessor):
    """Line plot-specific implementation"""

    _default_resample: ClassVar[str | None] = None

    def __init__(self: LinePlotAccessor, n: Network) -> None:
        super().__init__(n)
        self._time_aggregation = False

    def _validate(self: LinePlotAccessor, data: pd.DataFrame) -> pd.DataFrame:
        """Implement time series data validation"""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index for time series plots")
        return data

    def _plot(
        self: LinePlotAccessor,
        data: pd.DataFrame,
        resample: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Implement line plotting logic"""
        resample = resample or self._default_resample
        if resample:
            data = data.resample(resample).mean()
        # Line plotting implementation
        pass


class AreaPlotAccessor(BasePlotTypeAccessor):
    """Area plot-specific implementation"""

    _default_resample: ClassVar[str | None] = None
    _default_stacked: ClassVar[bool] = True

    def __init__(self: AreaPlotAccessor, n: Network) -> None:
        super().__init__(n)
        self._time_aggregation = False

    def _validate(self: AreaPlotAccessor, data: pd.DataFrame) -> pd.DataFrame:
        """Implement time series data validation"""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index for time series plots")
        return data

    def _plot(
        self: AreaPlotAccessor,
        data: pd.DataFrame,
        resample: str | None = None,
        stacked: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Implement area plotting logic"""
        resample = resample or self._default_resample
        stacked = stacked if stacked is not None else self._default_stacked
        if resample:
            data = data.resample(resample).mean()
        # Area plotting implementation
        pass


class PlotAccessor:
    """Main plot accessor providing access to different plot types"""

    _network: Network
    maps: MapPlotAccessor
    bar: BarPlotAccessor
    line: LinePlotAccessor
    area: AreaPlotAccessor

    def __init__(self: PlotAccessor, n: Network) -> None:
        self._network = n
        self.maps = MapPlotAccessor(n)
        self.bar = BarPlotAccessor(n)
        self.line = LinePlotAccessor(n)
        self.area = AreaPlotAccessor(n)
