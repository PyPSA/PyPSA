# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Statistics Accessor."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from pypsa._options import options
from pypsa.common import normalize_carrier_nice_names
from pypsa.constants import RE_PORTS
from pypsa.statistics.grouping import groupers

if TYPE_CHECKING:
    from pypsa import Network, NetworkCollection

logger = logging.getLogger(__name__)


class AbstractStatisticsAccessor(ABC):
    """Abstract accessor to calculate different statistical values."""

    def __init__(self, n: Network | NetworkCollection) -> None:
        """Initialize the statistics accessor."""
        self._n = n

    @property
    def n(self) -> Network | NetworkCollection:
        """Get the network instance."""
        warnings.warn(
            "Accessing the network instance via `n` is deprecated. Use the network instance directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._n

    def _get_grouping(
        self,
        n: Network | NetworkCollection,
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
            msg = f"Argument `groupby` must be a string, list, callable, or False, got {repr(groupby)}."
            raise ValueError(msg)
        return {"by": by, "level": level}

    @property
    def is_multi_indexed(self) -> bool:
        """Check if the snapshots are multi-indexed.

        Returns
        -------
        bool
            True if the snapshots are multi-indexed, False otherwise.

        """
        # TODO could be moved to Network
        return isinstance(self._n.snapshots, pd.MultiIndex)

    @classmethod
    def _aggregate_timeseries(
        cls, obj: Any, weights: pd.Series, agg: str | Callable | bool = "sum"
    ) -> Any:
        """Calculate the weighted sum or average of a DataFrame or Series."""
        if not agg:
            return obj.T if isinstance(obj, pd.DataFrame) else obj

        if agg is True:
            agg = "sum"

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
    def _apply_option_kwargs(self, *args: Any, **kwargs: Any) -> Any:
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
        components: Collection[str] | str | None = None,
        groupby: str | Sequence[str] | Callable | Literal[False] = "carrier",
        aggregate_across_components: bool = False,
        at_port: str | Sequence[str] | bool | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = True,
        drop_zero: bool | None = None,
        round: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Apply a function and group the result for a collection of components."""
        d = {}
        n = self._n

        if is_one_component := isinstance(components, str):
            components = [components]
        if components is None:
            components = sorted(n.branch_components | n.one_port_components)
        if nice_names is None:
            # TODO move to _apply_option_kwargs
            nice_names = options.params.statistics.nice_names
        # Normalize bus_carrier and carrier if nice_names was passed to allow using
        # nice names to filter
        if nice_names:
            nice_name_series = n.c.carriers.static.nice_name
            bus_carrier = normalize_carrier_nice_names(nice_name_series, bus_carrier)
            carrier = normalize_carrier_nice_names(nice_name_series, carrier)
        for c in components:
            if n.c[c].static.empty:
                continue

            ports = [
                match.group(1)
                for col in n.c[c].static
                if (match := RE_PORTS.search(str(col)))
            ]
            if not at_port:
                ports = [ports[0]]

            values = []
            for port in ports:
                vals = func(n, c, port)
                if self._aggregate_components_skip_iteration(vals):
                    continue

                vals = self._filter_active_assets(n, c, vals)  # for multiinvest
                vals = self._filter_bus_carrier(n, c, port, bus_carrier, vals)
                vals = self._filter_carrier(n, c, carrier, vals)

                if self._aggregate_components_skip_iteration(vals):
                    continue

                if groupby is not False:
                    if groupby is None:
                        warnings.warn(
                            "Passing `groupby=None` is deprecated. Drop the "
                            "argument to get the default grouping (by carrier), which "
                            "was also the previous default behavior. Deprecated in "
                            "version 0.34 and will be removed in version 1.0.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        groupby = "carrier"
                    grouping = self._get_grouping(
                        n, c, groupby, port=port, nice_names=nice_names
                    )
                    vals = self._aggregate_components_groupby(vals, grouping, agg, c)
                # Avoid having 'component' as index name in multiindex
                elif isinstance(vals, pd.DataFrame | pd.Series):
                    vals = vals.rename_axis(c, axis=0)
                values.append(vals)

            if not values:
                continue

            df = self._aggregate_components_concat_values(values, agg)

            d[c] = df
        df = self._aggregate_components_concat_data(d, is_one_component)
        if not df.empty:
            df = self._apply_option_kwargs(
                df,
                drop_zero=drop_zero,
                round=round,
                nice_names=nice_names,  # TODO: nice_names does not have effect here
            )

        if aggregate_across_components:
            df = self._aggregate_across_components(df, agg)

        if isinstance(df, pd.Series):
            df.name = None

        return df

    def _aggregate_components_skip_iteration(self, vals: Any) -> bool:
        return False

    def _filter_active_assets(
        self, n: Network | NetworkCollection, c: str, obj: Any
    ) -> Any:
        """For static values iterate over periods and concat values."""
        if isinstance(obj, pd.DataFrame) or "snapshot" in getattr(obj, "dims", []):
            return obj
        idx = self._get_component_index(obj, c)

        if not self.is_multi_indexed:
            mask = n.c[c].get_active_assets()
            return obj.loc[mask.index[mask].intersection(idx)]

        per_period = {}
        for p in n.investment_periods:
            mask = n.c[c].get_active_assets(p)
            per_period[p] = obj.loc[mask.index[mask].intersection(idx)]
        return self._concat_periods(per_period, c)

    def _filter_bus_carrier(
        self,
        n: Network | NetworkCollection,
        c: str,
        port: str,
        bus_carrier: str | Sequence[str] | None,
        obj: Any,
    ) -> Any:
        """Filter for components which are connected to bus with `bus_carrier`."""
        if bus_carrier is None:
            return obj

        idx = self._get_component_index(obj, c)
        ports = n.c[c].static.loc[idx, f"bus{port}"]

        # Handle MultiIndex (Collection and Stochastic Networks)
        bus_carriers = n.c.buses.static.carrier
        if isinstance(bus_carriers.index, pd.MultiIndex):
            bus_carriers = bus_carriers.groupby(level="name").first()

        port_carriers = ports.map(bus_carriers)
        if isinstance(bus_carrier, str):
            if bus_carrier in bus_carriers.unique():
                mask = port_carriers == bus_carrier
            else:
                mask = port_carriers.str.contains(bus_carrier, regex=True)
        elif isinstance(bus_carrier, list):
            mask = port_carriers.isin(bus_carrier)
        else:
            msg = f"Argument `bus_carrier` must be a string or list, got {type(bus_carrier)}"
            raise TypeError(msg)
        # links may have empty ports which results in NaNs
        mask = mask.where(mask.notnull(), False)
        return obj.loc[ports.index[mask]]

    def _filter_carrier(
        self,
        n: Network | NetworkCollection,
        c: str,
        carrier: str | Sequence[str] | None,
        obj: Any,
    ) -> Any:
        """Filter the DataFrame for components which have the specified carrier."""
        if carrier is None or "carrier" not in n.c[c].static:
            return obj

        idx = self._get_component_index(obj, c)
        carriers = n.c[c].static.loc[idx, "carrier"]

        if isinstance(carrier, str):
            if carrier in carriers.unique():
                mask = carriers == carrier
            else:
                mask = carriers.str.contains(carrier)
        elif isinstance(carrier, Sequence):
            mask = carriers.isin(carrier)
        else:
            msg = f"Argument `carrier` must be a string or list, got {type(carrier)}"
            raise TypeError(msg)

        return obj.loc[carriers.index[mask]]
