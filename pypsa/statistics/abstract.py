"""
Statistics Accessor.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypsa import Network

import warnings

import pandas as pd

from pypsa.statistics.grouping import deprecated_groupers, groupers

logger = logging.getLogger(__name__)


class Parameters:
    """
    Container for all the parameters.

    Attributes
    ----------
        drop_zero (bool): Flag indicating whether to drop zero values in statistic metrics.
        nice_names (bool): Flag indicating whether to use nice names in statistic metrics.
        round (int): Number of decimal places to round the values to in statistic metrics.

    Methods
    -------
        set_parameters(**kwargs): Sets the values of the parameters based on the provided keyword arguments.
    """

    PARAMETER_TYPES = {
        "drop_zero": bool,
        "nice_names": bool,
        "round": int,
    }

    def __init__(self) -> None:
        self.drop_zero = True
        self.nice_names = True
        self.round = 5

    def __repr__(self) -> str:
        param_str = ", ".join(
            f"{key}={getattr(self, key)}" for key in self.PARAMETER_TYPES
        )
        return f"Parameters({param_str})"

    def set_parameters(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            expected_type = self.PARAMETER_TYPES.get(key)
            if expected_type is None:
                raise ValueError(
                    f"Invalid parameter name: {key} \n Possible parameters are {list(self.PARAMETER_TYPES.keys())}"
                )
            elif not isinstance(value, expected_type):
                raise ValueError(
                    f"Invalid type for parameter {key}: expected {expected_type.__name__}, got {type(value).__name__}"
                )
            else:
                setattr(self, key, value)


class AbstractStatisticsAccessor(ABC):
    """
    Abstract accessor to calculate different statistical values.
    """

    def __init__(self, n: Network) -> None:
        self.n = n
        self.groupers = deprecated_groupers
        self.parameters = Parameters()

    def set_parameters(self, **kwargs: Any) -> None:
        """
        Setting the parameters for the statistics accessor.

        To see the list of parameters, one can simply call `n.statistics.parameters`.
        """
        self.parameters.set_parameters(**kwargs)

    def _get_grouping(
        self,
        n: Network,
        c: str,
        groupby: Callable | Sequence[str] | str | bool,
        port: str | None = None,
        nice_names: bool = False,
    ) -> dict:
        by = None
        level = None
        if callable(groupby):
            try:
                by = groupby(n, c, port=port, nice_names=nice_names)
            except TypeError:
                by = groupby(n, c, nice_names=nice_names)
        elif isinstance(groupby, (str | list)):
            by = groupers[groupby](n, c, port=port, nice_names=nice_names)
        elif groupby is not False:
            msg = f"Argument `groupby` must be a function, list, string, False or dict, got {repr(groupby)}."
            raise ValueError(msg)
        return dict(by=by, level=level)

    @property
    def is_multi_indexed(self) -> bool:
        return isinstance(self.n.snapshots, pd.MultiIndex)

    @classmethod
    def _aggregate_timeseries(
        cls, obj: Any, weights: pd.Series, agg: str | Callable | bool = "sum"
    ) -> Any:
        """
        Calculate the weighted sum or average of a DataFrame or Series.
        """
        if not agg:
            return obj.T if isinstance(obj, pd.DataFrame) else obj

        if agg == "mean":
            if isinstance(weights.index, pd.MultiIndex):
                weights = weights.groupby(level=0).transform(lambda w: w / w.sum())
            else:
                weights = weights / weights.sum()
            agg = "sum"

        return cls._aggregate_with_weights(obj, weights, agg)

    # The following methods are implemented in the concrete classes
    @abstractmethod
    def _aggregate_with_weights(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _aggregate_components_groupby(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _aggregate_components_concat_values(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _aggregate_components_concat_data(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _aggregate_across_components(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _get_component_index(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _concat_periods(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def _aggregate_components(
        self,
        func: Callable,
        agg: Callable | str = "sum",
        comps: Collection[str] | str | None = None,
        groupby: str | Sequence[str] | Callable = "carrier",
        aggregate_across_components: bool = False,
        at_port: str | Sequence[str] | bool | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = True,
    ) -> pd.Series | pd.DataFrame:
        """
        Apply a function and group the result for a collection of components.
        """
        d = {}
        n = self.n

        if is_one_component := isinstance(comps, str):
            comps = [comps]
        if comps is None:
            comps = n.branch_components | n.one_port_components
        if nice_names is None:
            nice_names = self.parameters.nice_names
        for c in comps:
            if n.static(c).empty:
                continue

            ports = [str(col)[3:] for col in n.static(c) if str(col).startswith("bus")]
            if not at_port:
                ports = [ports[0]]

            values = []
            for port in ports:
                vals = func(n, c, port)
                if self._aggregate_components_skip_iteration(vals):
                    continue

                vals = self._filter_active_assets(n, c, vals)  # for multiinvest
                vals = self._filter_bus_carrier(n, c, port, bus_carrier, vals)

                if self._aggregate_components_skip_iteration(vals):
                    continue

                if groupby is not False:
                    if groupby is None:
                        warnings.warn(
                            "Passing `groupby=None` is deprecated. Drop the "
                            "argument to get the default grouping (by carrier), which "
                            "was also the previous default behavior.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        groupby = "carrier"
                    grouping = self._get_grouping(
                        n, c, groupby, port=port, nice_names=nice_names
                    )
                    vals = self._aggregate_components_groupby(vals, grouping, agg)
                values.append(vals)

            if not values:
                continue

            df = self._aggregate_components_concat_values(values, agg)

            d[c] = df

        df = self._aggregate_components_concat_data(d, is_one_component)

        if aggregate_across_components:
            df = self._aggregate_across_components(df, agg)

        return df

    def _aggregate_components_skip_iteration(self, vals: Any) -> bool:
        return False

    def _filter_active_assets(self, n: Network, c: str, obj: Any) -> Any:
        """
        For static values iterate over periods and concat values.
        """
        if isinstance(obj, pd.DataFrame) or "snapshot" in getattr(obj, "dims", []):
            return obj
        idx = self._get_component_index(obj, c)
        if not self.is_multi_indexed:
            mask = n.get_active_assets(c)
            idx = mask.index[mask].intersection(idx)
            return obj.loc[idx]

        per_period = {}
        for p in n.investment_periods:
            mask = n.get_active_assets(c, p)
            idx = mask.index[mask].intersection(idx)
            per_period[p] = obj.loc[idx]

        return self._concat_periods(per_period, c)

    def _filter_bus_carrier(
        self,
        n: Network,
        c: str,
        port: str,
        bus_carrier: str | Sequence[str] | None,
        obj: Any,
    ) -> Any:
        """
        Filter the DataFrame for components which are connected to a bus with
        carrier `bus_carrier`.
        """
        if bus_carrier is None:
            return obj

        idx = self._get_component_index(obj, c)
        ports = n.static(c).loc[idx, f"bus{port}"]
        port_carriers = ports.map(n.buses.carrier)
        if isinstance(bus_carrier, str):
            if bus_carrier in n.buses.carrier.unique():
                mask = port_carriers == bus_carrier
            else:
                mask = port_carriers.str.contains(bus_carrier)
        elif isinstance(bus_carrier, list):
            mask = port_carriers.isin(bus_carrier)
        else:
            raise ValueError(
                f"Argument `bus_carrier` must be a string or list, got {type(bus_carrier)}"
            )
        # links may have empty ports which results in NaNs
        mask = mask.where(mask.notnull(), False)
        return obj.loc[ports.index[mask]]
