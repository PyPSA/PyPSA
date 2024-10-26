"""
Statistics Expression Accessor.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

import linopy as ln
import pandas as pd
from linopy.expressions import LinearExpression
from xarray import DataArray

from pypsa.descriptors import nominal_attrs
from pypsa.statistics import (
    AbstractStatisticsAccessor,
    get_carrier_and_bus_carrier,
    get_transmission_branches,
    get_weightings,
    port_efficiency,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypsa import Network


def get_operational_attr(c: str) -> str | None:
    from pypsa.optimization.optimize import lookup

    if c not in lookup.index:
        return None

    attr = lookup.query("not nominal and not handle_separately").loc[c].index.item()
    return attr


def pass_none_if_keyerror(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (KeyError, AttributeError):
            return None

    return wrapper


class StatisticExpressionsAccessor(AbstractStatisticsAccessor):
    """
    Accessor to calculate different statistical expressions.

    This class is used to calculate different statistical expressions like
    capital expenditure, capacity, energy balance, etc.
    The results are aggregated by the given groupby function.
    """

    @classmethod
    def _get_grouping(
        cls,
        n: Network,
        c: str,
        groupby: Callable | Sequence[str] | str | bool,
        port: str | None = None,
        nice_names: bool = False,
    ) -> pd.DataFrame:
        result = super()._get_grouping(n, c, groupby, port, nice_names)
        by = result["by"]

        if isinstance(by, list):
            grouper = pd.concat(by, axis=1)
        elif isinstance(by, pd.Series):
            grouper = by.to_frame()
        elif groupby is False:
            grouper = pd.DataFrame(index=n.df(c).index)
        else:
            grouper = by

        grouper.insert(0, "component", c)  # for tracking the component
        return grouper.rename_axis(c)

    def _get_component_index(self, obj: LinearExpression, c: str) -> pd.Index:
        return obj.indexes[c]

    def _concat_periods(self, exprs: dict[str, LinearExpression], c: str) -> Any:
        return ln.merge(list(exprs.values()), dim=c)

    @staticmethod
    def _aggregate_with_weights(
        expr: LinearExpression,
        weights: pd.Series,
        agg: str | Callable,
    ) -> LinearExpression:
        """
        Apply weights to a time series.
        """
        if agg == "sum":
            if isinstance(weights.index, pd.MultiIndex):
                return expr.multiply(weights, axis=0).groupby(level=0).sum().T
            return expr @ weights
        else:
            raise ValueError(f"Aggregation method {agg} not supported.")

    def _aggregate_components_groupby(
        self, vals: LinearExpression, grouping: pd.DataFrame, agg: Callable | str
    ) -> pd.DataFrame:
        return vals.groupby(grouping).sum()

    def _aggregate_components_concat_values(
        self, exprs: list[LinearExpression], agg: Callable | str
    ) -> LinearExpression:
        res = ln.merge(exprs)
        if not (index := res.indexes[res.dims[0]]).is_unique:
            if agg != "sum":
                raise ValueError(f"Aggregation method {agg} not supported.")
            non_unique_groups = pd.DataFrame(list(index), columns=index.names)
            res = res.groupby(non_unique_groups).sum()
        return res

    def _aggregate_components_concat_data(
        self, res: dict[str, LinearExpression], is_one_component: bool
    ) -> LinearExpression:
        if res == {}:
            return LinearExpression(None, self.n.model)
        if is_one_component:
            first_key = next(iter(res))
            return res[first_key].loc[first_key]
        return ln.merge(list(res.values()), dim="group")

    def capex(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_groups: str = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = False,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
        cost_attribute: str = "capital_cost",
        include_non_extendable: bool = True,
    ) -> LinearExpression:
        """
        Calculate the capital expenditure of the network in given currency.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series | None:
            m = n.model
            capacity = m.variables[f"{c}-{nominal_attrs[c]}"]
            capacity = capacity.rename({f"{c}-ext": c})
            if include_non_extendable:
                query = f"~{nominal_attrs[c]}_extendable"
                capacity = capacity + n.df(c).query(query)["p_nom"]
            costs = n.df(c)[cost_attribute][capacity.indexes[c]]
            return capacity * costs

        return self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )

    def capacity(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_groups: str = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        storage: bool = False,
        nice_names: bool | None = None,
        include_non_extendable: bool = True,
    ) -> LinearExpression:
        """
        Calculate the optimal capacity of the network components in MW.

        If `bus_carrier` is given, the capacity is weighed by the output efficiency
        of components at buses with carrier `bus_carrier`.

        If storage is set to True, only storage capacities of the component
        `Store` and `StorageUnit` are taken into account.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """

        if storage:
            comps = ("Store", "StorageUnit")
        if bus_carrier and at_port is None:
            at_port = True

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series | None:
            m = n.model
            capacity = m.variables[f"{c}-{nominal_attrs[c]}"]
            capacity = capacity.rename({f"{c}-ext": c})
            if include_non_extendable:
                query = f"~{nominal_attrs[c]}_extendable"
                capacity = capacity + n.df(c).query(query)["p_nom"]
            efficiency = port_efficiency(n, c, port=port)[capacity.indexes[c]]
            res = capacity * efficiency
            if storage and (c == "StorageUnit"):
                res = res * n.df(c).max_hours
            return res

        return self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )

    def opex(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: str = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = False,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """
        Calculate the operational expenditure in the network in given currency.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated
            using snapshot weightings. With False the time series is given in currency/hour. Defaults to 'sum'.
        """
        from pypsa.optimization.optimize import lookup

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series | None:
            attr = lookup.query("not nominal and marginal_cost").loc[c].index.item()
            if attr is None:
                return None
            var = n.model.variables[f"{c}-{attr}"]
            opex = var * n.get_switchable_as_dense(c, "marginal_cost")
            weights = get_weightings(n, c)
            return self._aggregate_timeseries(opex, weights, agg=aggregate_time)

        return self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )

    def transmission(
        self,
        comps: Collection[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: str = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = False,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """
        Calculate the transmission of branch components in the network. Units
        depend on the regarded bus carrier.

        If `bus_carrier` is given, only the flow between buses with
        carrier `bus_carrier` is calculated.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to MWh
            using snapshot weightings. With False the time series is given in MW. Defaults to 'sum'.
        """

        if comps is None:
            comps = self.n.branch_components

        transmission_branches = get_transmission_branches(self.n, bus_carrier)

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            attr = get_operational_attr(c)
            idx = transmission_branches.get_loc_level(c)[1].rename(c)
            var = n.model.variables[f"{c}-{attr}"]
            efficiency = port_efficiency(n, c, port=port, dynamic=True)
            p = var.loc[:, idx] * efficiency[idx]
            weights = get_weightings(n, c)
            return self._aggregate_timeseries(p, weights, agg=aggregate_time)

        return self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )

    def energy_balance(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: str = "sum",
        groupby: Callable | None = get_carrier_and_bus_carrier,
        at_port: Sequence[str] | str | bool = True,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """
        Calculate the energy balance of components in the network. Positive
        values represent a supply and negative a withdrawal. Units depend on
        the regarded bus carrier.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Additional parameter
        --------------------
        aggregate_bus: bool, optional
            Whether to obtain the nodal or carrier-wise energy balance. Default is True, corresponding to the carrier-wise balance.
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to MWh
            using snapshot weightings. With False the time series is given in MW. Defaults to 'sum'.
        """

        if (
            self.n.buses.carrier.unique().size > 1
            and groupby is None
            and bus_carrier is None
        ):
            logger.warning(
                "Network has multiple bus carriers which are aggregated together. To separate bus carriers set `bus_carrier` or use groupers like `get_carrier_and_bus_carrier`."
            )

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            attr = get_operational_attr(c)
            var = n.model.variables[f"{c}-{attr}"]
            efficiency = port_efficiency(n, c, port=port, dynamic=True)
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            weights = get_weightings(n, c)
            p = var * efficiency * sign
            return self._aggregate_timeseries(p, weights, agg=aggregate_time)

        return self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )

    def curtailment(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: str = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = False,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """
        Calculate the curtailment of components in the network in MWh.

        The calculation only considers assets with a `p_max_pu` time
        series, which is used to quantify the available power potential.

        If `bus_carrier` is given, only the assets are considered which are
        connected to buses with carrier `bus_carrier`.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to MWh
            using snapshot weightings. With False the time series is given in MW. Defaults to 'sum'.
        """

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            attr = nominal_attrs[c]
            capacity = (
                n.model.variables[f"{c}-{attr}"].rename({f"{c}-ext": c})
                + n.df(c).query(f"~{attr}_extendable")["p_nom"]
            )
            idx = capacity.indexes[c]
            p_max_pu = DataArray(n.get_switchable_as_dense(c, "p_max_pu")[idx])
            operation = n.model.variables[f"{c}-{get_operational_attr(c)}"].loc[:, idx]
            # the following needs to be fixed in linopy, right now constants cannot be used for broadcasting
            # curtailment = capacity * p_max_pu - operation
            curtailment = (capacity - operation / p_max_pu) * p_max_pu
            weights = get_weightings(n, c)
            return self._aggregate_timeseries(curtailment, weights, agg=aggregate_time)

        return self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )

    def operation(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "mean",
        aggregate_groups: str = "sum",
        at_port: Sequence[str] | str | bool = False,
        groupby: Callable | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """
        Calculate the operation of components in the network.

        If `bus_carrier` is given, only the assets are considered which are
        connected to buses with carrier `bus_carrier`.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to
            using snapshot weightings. With False the time series is given. Defaults to 'mean'.
        """

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            attr = get_operational_attr(c)
            operation = 1 * n.model.variables[f"{c}-{attr}"]
            weights = get_weightings(n, c)
            return self._aggregate_timeseries(operation, weights, agg=aggregate_time)

        return self._aggregate_components(
            func,
            agg=aggregate_groups,
            comps=comps,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
