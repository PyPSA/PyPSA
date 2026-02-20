# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Statistics Expression Accessor."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import linopy as ln
import numpy as np
import pandas as pd
from linopy import LinearExpression, Variable
from packaging import version
from xarray import DataArray

from pypsa.common import deprecated_kwargs, pass_none_if_keyerror
from pypsa.statistics import (
    get_transmission_branches,
    port_efficiency,
)
from pypsa.statistics.abstract import AbstractStatisticsAccessor, resolve_at_port

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from pypsa import Network, NetworkCollection
    from pypsa.components.components import PortsLike
logger = logging.getLogger(__name__)


USE_EMPTY_PROPERTY = version.parse(ln.__version__) >= version.parse("0.5.1")


def check_if_empty(expr: LinearExpression) -> bool:
    """Check if the expression is empty.

    This is a workaround for the issue that linopy does not support
    the empty property for older versions (`.empty` in >=0.5.1 vs `.empty()` in <0.5.1).
    """
    if USE_EMPTY_PROPERTY:
        return expr.empty
    return expr.empty()


class StatisticExpressionsAccessor(AbstractStatisticsAccessor):
    """Accessor to calculate different statistical expressions.

    This class is used to calculate different statistical expressions like
    capital expenditure, capacity, energy balance, etc.
    The results are aggregated by the given groupby function.
    """

    def _get_grouping(
        self,
        n: Network | NetworkCollection,
        c: str,
        groupby: Callable | Sequence[str] | str | bool,
        port: str,
        nice_names: bool = False,
    ) -> pd.DataFrame:
        result = super()._get_grouping(n, c, groupby, port, nice_names)
        by = result["by"]

        if isinstance(by, list):
            grouper = pd.concat(by, axis=1)
        elif isinstance(by, pd.Series):
            grouper = by.to_frame()
        elif groupby is False:
            grouper = pd.DataFrame(index=n.c[c].static.index)
        else:
            grouper = by

        grouper.insert(0, "component", c)  # for tracking the component
        return grouper

    def _get_component_index(self, obj: LinearExpression, c: str) -> pd.Index:
        return obj.indexes["name"]

    def _concat_periods(self, exprs: dict[str, LinearExpression], c: str) -> Any:
        return ln.merge(list(exprs.values()), dim=c)

    @staticmethod
    def _aggregate_with_weights(
        expr: LinearExpression,
        weights: pd.Series,
        agg: str | Callable,
    ) -> LinearExpression:
        """Apply weights to a time series."""
        if agg == "sum":
            if isinstance(weights.index, pd.MultiIndex):
                return expr.multiply(weights, axis=0).groupby(level=0).sum().T
            return expr @ weights
        msg = f"Aggregation method {agg} not supported."
        raise ValueError(msg)

    def _aggregate_components_skip_iteration(self, vals: Any) -> bool:
        return vals is None or (not np.prod(vals.shape) and (vals.const == 0).all())

    def _aggregate_components_groupby(
        self,
        vals: LinearExpression,
        grouping: pd.DataFrame,
        agg: Callable | str,
        c: str,
    ) -> pd.DataFrame:
        return vals.groupby(grouping).sum()

    def _aggregate_components_concat_values(
        self, exprs: list[LinearExpression], agg: Callable | str
    ) -> LinearExpression:
        res = ln.merge(exprs)
        if not (index := res.indexes[res.dims[0]]).is_unique:
            if agg != "sum":
                msg = f"Aggregation method {agg} not supported."
                raise ValueError(msg)
            non_unique_groups = pd.DataFrame(list(index), columns=index.names)
            res = res.groupby(non_unique_groups).sum()
        return res

    def _aggregate_components_concat_data(
        self, res: dict[str, LinearExpression], is_one_component: bool
    ) -> LinearExpression:
        if res == {}:
            return LinearExpression(None, self._n.model)
        if is_one_component:
            first_key = next(iter(res))
            return res[first_key].loc[first_key]
        return ln.merge(list(res.values()), dim="group")

    def _apply_option_kwargs(
        self,
        expr: LinearExpression,
        nice_names: bool | None,
        drop_zero: bool | None,
        round: int | None,
    ) -> LinearExpression:
        # Expressions only support nice_names right now which applied elsewhere
        # TODO
        return expr

    def _aggregate_across_components(
        self, expr: LinearExpression, agg: Callable | str
    ) -> LinearExpression:
        if agg != "sum":
            msg = f"Aggregation method {agg} not supported."
            raise ValueError(msg)
        if check_if_empty(expr):
            return expr
        group = expr.indexes["group"].to_frame().drop(columns="component").squeeze()
        return expr.groupby(group).sum()

    def _get_operational_variable(self, c: str) -> Variable | LinearExpression:
        # TODO: move function to better place to avoid circular imports
        from pypsa.optimization.optimize import lookup  # noqa: PLC0415

        m = self._n.model

        if c == "Load":
            return LinearExpression(self._n.get_switchable_as_dense(c, "p_set"), m)
        attr = lookup.query("not nominal and not handle_separately").loc[c].index
        if c == "StorageUnit":
            return m.variables[f"{c}-p_dispatch"] - m.variables[f"{c}-p_store"]
        attr = attr.item()
        return m.variables[f"{c}-{attr}"]

    @deprecated_kwargs(
        deprecated_in="1.0",
        removed_in="2.0",
        comps="components",
        aggregate_groups="groupby_method",
        aggregate_time="groupby_time",
    )
    def capex(
        self,
        components: str | Sequence[str] | None = None,
        groupby_method: str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: PortsLike | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        cost_attribute: str = "capital_cost",
        include_non_extendable: bool = True,
    ) -> LinearExpression:
        """Calculate the capital expenditure of the network in given currency.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        at_port = resolve_at_port(at_port, bus_carrier)

        @pass_none_if_keyerror
        def func(n: Network, component: str, port: str) -> pd.Series | None:
            m = n.model
            c = n.c[component]
            nom_attr = c._operational_attrs["nom"]
            var_name = f"{component}-{nom_attr}"

            # Get non-extendable capacity using component's fixed property
            non_ext_capacity = (
                c.static.loc[c.fixed, nom_attr]
                if include_non_extendable
                else pd.Series(dtype=float)
            )

            # Build capacity expression handling both extendable and non-extendable
            if var_name in m.variables:
                capacity = m.variables[var_name] + non_ext_capacity
            elif not non_ext_capacity.empty:
                capacity = LinearExpression(non_ext_capacity, m)
            else:
                return None

            costs = c.static[cost_attribute][capacity.indexes["name"]]
            return capacity * costs

        return self._aggregate_components(
            func,
            components=components,
            agg=groupby_method,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            carrier=carrier,
            nice_names=nice_names,
        )

    @deprecated_kwargs(
        deprecated_in="1.0",
        removed_in="2.0",
        comps="components",
        aggregate_groups="groupby_method",
        aggregate_time="groupby_time",
    )
    def capacity(
        self,
        components: str | Sequence[str] | None = None,
        groupby_method: str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: PortsLike | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        carrier: str | Sequence[str] | None = None,
        storage: bool = False,
        nice_names: bool | None = None,
        include_non_extendable: bool = True,
    ) -> LinearExpression:
        """Calculate the optimal capacity of the network components in MW.

        If `bus_carrier` is given, the capacity is weighed by the output efficiency
        of components at buses with carrier `bus_carrier`.

        If storage is set to True, only storage capacities of the component
        `Store` and `StorageUnit` are taken into account.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        if storage:
            components = ("Store", "StorageUnit")
        at_port = resolve_at_port(at_port, bus_carrier)

        @pass_none_if_keyerror
        def func(n: Network, component: str, port: str) -> pd.Series | None:
            m = n.model
            c = n.c[component]
            nom_attr = c._operational_attrs["nom"]
            var_name = f"{component}-{nom_attr}"

            # Get non-extendable capacity using component's fixed property
            non_ext_capacity = (
                c.static.loc[c.fixed, nom_attr]
                if include_non_extendable
                else pd.Series(dtype=float)
            )

            # Build capacity expression handling both extendable and non-extendable
            if var_name in m.variables:
                capacity = m.variables[var_name] + non_ext_capacity
            elif not non_ext_capacity.empty:
                capacity = LinearExpression(non_ext_capacity, m)
            else:
                return None

            efficiency = port_efficiency(n, component, port=port)[
                capacity.indexes["name"]
            ]
            if c._as_ports(at_port) == [0]:
                efficiency = abs(efficiency)
            res = capacity * efficiency
            if storage and (component == "StorageUnit"):
                res = res * c.static.max_hours
            return res

        return self._aggregate_components(
            func,
            components=components,
            agg=groupby_method,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            carrier=carrier,
            nice_names=nice_names,
        )

    @deprecated_kwargs(
        deprecated_in="1.0",
        removed_in="2.0",
        comps="components",
        aggregate_groups="groupby_method",
        aggregate_time="groupby_time",
    )
    def opex(  # noqa: D417
        self,
        components: str | Sequence[str] | None = None,
        groupby_time: str | bool = "sum",
        groupby_method: str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: PortsLike | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """Calculate the operational expenditure in the network in given currency.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        groupby_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated
            using snapshot weightings. With False the time series is given in currency/hour. Defaults to 'sum'.

        """
        from pypsa.optimization.optimize import lookup  # noqa: PLC0415

        at_port = resolve_at_port(at_port, bus_carrier)

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series | None:
            attr = lookup.query("not nominal and marginal_cost").loc[c].index.item()
            if attr is None:
                return None
            var = n.model.variables[f"{c}-{attr}"]
            sns = var.indexes["snapshot"]
            opex = var * n.get_switchable_as_dense(c, "marginal_cost").loc[sns]
            weights = n.snapshot_weightings.objective.loc[sns]
            return self._aggregate_timeseries(opex, weights, agg=groupby_time)

        return self._aggregate_components(
            func,
            components=components,
            agg=groupby_method,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            carrier=carrier,
            nice_names=nice_names,
        )

    @deprecated_kwargs(
        deprecated_in="1.0",
        removed_in="2.0",
        comps="components",
        aggregate_groups="groupby_method",
        aggregate_time="groupby_time",
    )
    def transmission(  # noqa: D417
        self,
        components: Collection[str] | str | None = None,
        groupby_time: str | bool = "sum",
        groupby_method: str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: PortsLike | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """Calculate the transmission of branch components in the network.

        Units depend on the regarded bus carrier.

        If `bus_carrier` is given, only the flow between buses with
        carrier `bus_carrier` is calculated.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        groupby_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to MWh
            using snapshot weightings. With False the time series is given in MW. Defaults to 'sum'.

        """
        at_port = resolve_at_port(at_port, bus_carrier)

        if components is None:
            components = self._n.branch_components

        transmission_branches = get_transmission_branches(self._n, bus_carrier)

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            var = self._get_operational_variable(c)
            sns = var.indexes["snapshot"]
            idx = transmission_branches.get_loc_level(c)[1].rename(c)
            efficiency = port_efficiency(n, c, port=port, dynamic=True)
            if isinstance(efficiency, pd.DataFrame):
                efficiency = efficiency.loc[sns]
            p = var.loc[:, idx] * efficiency[idx]
            weights = n.snapshot_weightings.generators.loc[sns]
            return self._aggregate_timeseries(p, weights, agg=groupby_time)

        return self._aggregate_components(
            func,
            components=components,
            agg=groupby_method,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            carrier=carrier,
            nice_names=nice_names,
        )

    @deprecated_kwargs(
        deprecated_in="1.0",
        removed_in="2.0",
        comps="components",
        aggregate_groups="groupby_method",
        aggregate_time="groupby_time",
    )
    @deprecated_kwargs(
        deprecated_in="1.1",
        removed_in="2.0",
        kind="direction",
    )
    def energy_balance(  # noqa: D417
        self,
        components: str | Sequence[str] | None = None,
        groupby_time: str | bool = "sum",
        groupby_method: str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable | None = None,
        at_port: PortsLike = "all",
        bus_carrier: str | Sequence[str] | None = None,
        carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        direction: str | None = "both",
    ) -> LinearExpression:
        """Calculate the energy balance of components in the network.

        Positive values represent a supply and negative a withdrawal. Units depend on
        the regarded bus carrier.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        groupby_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to MWh
            using snapshot weightings. With False the time series is given in MW. Defaults to 'sum'.
        direction : str, default="both"
            Type of energy balance to calculate:
            - 'supply': Only consider positive values (energy production)
            - 'withdrawal': Only consider negative values (energy consumption)
            - 'both': Consider both supply and withdrawal

        """
        if groupby is None:
            groupby = ["carrier", "bus_carrier"]
        if direction is None:
            warnings.warn(
                "Passing `direction=None` is deprecated. Use `direction='both'` instead. Deprecated in version 1.1. Will be removed in version 2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            direction = "both"
        if (
            self._n.c.buses.static.carrier.unique().size > 1
            and groupby is None
            and bus_carrier is None
        ):
            logger.warning(
                "Network has multiple bus carriers which are aggregated together. "
                "To separate bus carriers set `bus_carrier` or use `bus_carrier` in the groupby argument."
            )

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            var = self._get_operational_variable(c)
            sns = var.indexes["snapshot"]
            # negative branch contributions are considered by the efficiency
            efficiency = port_efficiency(n, c, port=port, dynamic=True)
            if isinstance(efficiency, pd.DataFrame):
                efficiency = efficiency.loc[sns]
            sign = n.c[c].static.get("sign", 1.0)
            weights = n.snapshot_weightings.generators.loc[sns]
            coeffs = DataArray(efficiency * sign)
            if direction == "supply":
                coeffs = coeffs.clip(min=0)
            elif direction == "withdrawal":
                logger.warning(
                    "The sign convention for withdrawal has changed: withdrawal values are now reported as positive numbers instead of negative numbers."
                )
                coeffs = -coeffs.clip(max=0)
            elif direction != "both":
                msg = f"Got unexpected argument direction={direction}. Must be 'supply', 'withdrawal' or 'both'."
                raise ValueError(msg)
            p = var.where(coeffs != 0) * coeffs
            return self._aggregate_timeseries(p, weights, agg=groupby_time)

        return self._aggregate_components(
            func,
            components=components,
            agg=groupby_method,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            carrier=carrier,
            nice_names=nice_names,
        )

    @deprecated_kwargs(
        deprecated_in="1.0",
        removed_in="2.0",
        comps="components",
        aggregate_groups="groupby_method",
        aggregate_time="groupby_time",
    )
    def supply(
        self,
        components: str | Sequence[str] | None = None,
        groupby_time: str | bool = "sum",
        groupby_method: str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable | None = None,
        at_port: PortsLike = "all",
        bus_carrier: str | Sequence[str] | None = None,
        carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """Calculate the supply of components in the network.

        Units depend on the regarded bus carrier.

        If `bus_carrier` is given, only the supply to buses with carrier
        `bus_carrier` is calculated.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        if groupby is None:
            groupby = ["carrier", "bus_carrier"]
        return self.energy_balance(
            components=components,
            groupby_time=groupby_time,
            groupby_method=groupby_method,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            carrier=carrier,
            nice_names=nice_names,
            direction="supply",
        )

    @deprecated_kwargs(
        deprecated_in="1.0",
        removed_in="2.0",
        comps="components",
        aggregate_groups="groupby_method",
        aggregate_time="groupby_time",
    )
    def withdrawal(
        self,
        components: str | Sequence[str] | None = None,
        groupby_time: str | bool = "sum",
        groupby_method: str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable | None = None,
        at_port: PortsLike = "all",
        bus_carrier: str | Sequence[str] | None = None,
        carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """Calculate the withdrawal of components in the network.

        Units depend on the regarded bus carrier.

        If `bus_carrier` is given, only the withdrawal from buses with
        carrier `bus_carrier` is calculated.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        if groupby is None:
            groupby = ["carrier", "bus_carrier"]
        return self.energy_balance(
            components=components,
            groupby_time=groupby_time,
            groupby_method=groupby_method,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            carrier=carrier,
            nice_names=nice_names,
            direction="withdrawal",
        )

    @deprecated_kwargs(
        deprecated_in="1.0",
        removed_in="2.0",
        comps="components",
        aggregate_groups="groupby_method",
        aggregate_time="groupby_time",
    )
    def curtailment(  # noqa: D417
        self,
        components: str | Sequence[str] | None = None,
        groupby_time: str | bool = "sum",
        groupby_method: str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: PortsLike | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """Calculate the curtailment of components in the network in MWh.

        The calculation only considers assets with a `p_max_pu` time
        series, which is used to quantify the available power potential.

        If `bus_carrier` is given, only the assets are considered which are
        connected to buses with carrier `bus_carrier`.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        groupby_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to MWh
            using snapshot weightings. With False the time series is given in MW. Defaults to 'sum'.

        """
        at_port = resolve_at_port(at_port, bus_carrier)

        @pass_none_if_keyerror
        def func(n: Network, component: str, port: str) -> pd.Series:
            m = n.model
            c = n.c[component]
            nom_attr = c._operational_attrs["nom"]
            var_name = f"{component}-{nom_attr}"

            # Get non-extendable capacity using component's fixed property
            non_ext_capacity = c.static.loc[c.fixed, nom_attr]

            # Build capacity expression handling both extendable and non-extendable
            if var_name in m.variables:
                capacity = m.variables[var_name] + non_ext_capacity
            elif not non_ext_capacity.empty:
                capacity = LinearExpression(non_ext_capacity, m)
            else:
                return None

            idx = capacity.indexes["name"]
            operation = self._get_operational_variable(component).loc[:, idx]
            sns = operation.indexes["snapshot"]
            p_max_pu = DataArray(
                n.get_switchable_as_dense(component, "p_max_pu")[idx]
            ).loc[sns]
            # the following needs to be fixed in linopy, right now constants cannot be used for broadcasting
            # TODO curtailment = capacity * p_max_pu - operation
            curtailment = (capacity - operation / p_max_pu) * p_max_pu
            weights = n.snapshot_weightings.generators.loc[sns]
            return self._aggregate_timeseries(curtailment, weights, agg=groupby_time)

        return self._aggregate_components(
            func,
            components=components,
            agg=groupby_method,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            carrier=carrier,
            nice_names=nice_names,
        )

    @deprecated_kwargs(
        deprecated_in="1.0",
        removed_in="2.0",
        comps="components",
        aggregate_groups="groupby_method",
        aggregate_time="groupby_time",
    )
    def operation(  # noqa: D417
        self,
        components: str | Sequence[str] | None = None,
        groupby_time: str | bool = "mean",
        groupby_method: str = "sum",
        aggregate_across_components: bool = False,
        at_port: PortsLike | None = None,
        groupby: str | Sequence[str] | Callable = "carrier",
        bus_carrier: str | Sequence[str] | None = None,
        carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
    ) -> LinearExpression:
        """Calculate the operation of components in the network.

        If `bus_carrier` is given, only the assets are considered which are
        connected to buses with carrier `bus_carrier`.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        groupby_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to
            using snapshot weightings. With False the time series is given. Defaults to 'mean'.

        """
        at_port = resolve_at_port(at_port, bus_carrier)

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            operation = self._get_operational_variable(c)
            sns = operation.indexes["snapshot"]
            weights = n.snapshot_weightings.generators.loc[sns]
            return self._aggregate_timeseries(operation, weights, agg=groupby_time)

        return self._aggregate_components(
            func,
            agg=groupby_method,
            components=components,
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            at_port=at_port,
            bus_carrier=bus_carrier,
            carrier=carrier,
            nice_names=nice_names,
        )
