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

from pypsa._linopy_compat import suppress_semantics_warnings
from pypsa.common import deprecated_kwargs, pass_none_if_keyerror
from pypsa.components._types.mixin.multiports import _Multiport
from pypsa.optimization.window import apply_period_weighting
from pypsa.statistics import (
    get_transmission_branches,
    port_efficiency,
)
from pypsa.statistics.abstract import AbstractStatisticsAccessor, resolve_at_port

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from xarray import DataArray

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


def _port_coefficients(n: Network, c: str, port: str, sns: pd.Index) -> DataArray:
    """Port efficiencies of `c` as an array, restricted to `sns` if time-varying."""
    efficiency = port_efficiency(n, c, port=port, dynamic=True, as_xarray=True)
    if "snapshot" in efficiency.dims:
        return efficiency.sel(snapshot=sns)
    return efficiency


def _normalized_weights(weights: DataArray) -> DataArray:
    """Scale `weights` to sum to one, per investment period if the snapshots carry one."""
    if "period" not in weights.coords:
        return weights / weights.sum()
    totals = weights.groupby("period").sum().to_series()
    return apply_period_weighting(weights, 1 / totals)


def _require_sum_agg(agg: Callable | str) -> None:
    """Reject any aggregation method other than plain summation."""
    if agg != "sum":
        msg = f"Aggregation method {agg} not supported."
        raise ValueError(msg)


def _group_key_coords(expr: LinearExpression) -> list[str]:
    """Grouping-key coordinates carried along a flat `group` dimension."""
    return [
        k for k in expr.coords if k != "group" and expr.coords[k].dims == ("group",)
    ]


def _flatten_group_index(expr: LinearExpression) -> LinearExpression:
    """Turn a stacked `group` MultiIndex into a flat dim with the keys as aux coords.

    Legacy linopy's multi-key `groupby` returns a stacked `group` `MultiIndex`;
    v1 returns the flat form directly. Normalising here gives the rest of the
    pipeline a single representation. Inverse of `_restack_flat_groups`.
    """
    if isinstance(expr.indexes.get("group"), pd.MultiIndex):
        return expr.reset_index("group")
    return expr


def _restack_flat_groups(expr: Any) -> Any:
    """Give a flat-`group` expression the public stacked group index.

    The grouping keys ride as auxiliary coordinates on a flat `group` dim
    throughout the pipeline; restack them on exit so the public return shape is the
    same under both semantics. Inverse of `_flatten_group_index`.
    """
    if not isinstance(expr, LinearExpression) or "group" not in expr.dims:
        return expr
    keys = _group_key_coords(expr)
    if not keys:
        return expr
    indexed = expr.set_index(group=keys)
    return indexed.rename(group=keys[0]) if len(keys) == 1 else indexed


def _concat_flat_groups(exprs: list[LinearExpression]) -> LinearExpression:
    """Concatenate grouped expressions along a flat, globally unique `group` dim.

    `linopy.merge` cannot concatenate flat `group` dims whose labels overlap, so
    the keys are stripped, the groups renumbered to be globally unique, merged, and
    the keys reattached as `group`-indexed coordinates.
    """
    parts: list[LinearExpression] = []
    frames: list[pd.DataFrame] = []
    offset = 0
    for expr in exprs:
        keys = _group_key_coords(expr)
        frames.append(
            pd.DataFrame({k: np.asarray(expr.coords[k].values) for k in keys})
        )
        size = expr.sizes["group"]
        renumbered = expr.drop_vars(keys).assign_coords(
            group=np.arange(offset, offset + size)
        )
        offset += size
        parts.append(renumbered)
    merged = ln.merge(parts, dim="group") if len(parts) > 1 else parts[0]
    frame = pd.concat(frames, ignore_index=True)
    return merged.assign_coords(
        {col: ("group", frame[col].to_numpy()) for col in frame.columns}
    )


def _regroup_flat(expr: LinearExpression, by: list[str]) -> LinearExpression:
    """Regroup a flat-`group` expression by a subset of its key coordinates.

    Keeps the flat `group` dimension and reattaches the selected keys, summing
    entries that share the same key tuple.
    """
    keys = pd.MultiIndex.from_arrays(
        [np.asarray(expr.coords[k].values) for k in by], names=by
    )
    codes, uniques = pd.factorize(keys, sort=True)
    # Group by a single code coordinate (rather than a Series grouper, which
    # would misalign against other dimensions such as `snapshot`).
    grouped = (
        expr.assign_coords(_group_code=("group", codes))
        .groupby("_group_code")
        .sum()
        .rename(_group_code="group")
    )
    return grouped.assign_coords(
        {
            name: ("group", uniques.get_level_values(i).to_numpy())
            for i, name in enumerate(by)
        }
    )


def _capacity_expression(
    n: Network, component: str, include_non_extendable: bool = True
) -> LinearExpression | None:
    """Nominal capacity of `component` as an expression, or `None` if it has none.

    Combines the extendable capacity variables with the fixed capacities of the
    non-extendable assets.
    """
    m = n.model
    c = n.c[component]
    nom_attr = c._operational_attrs["nom"]
    var_name = f"{component}-{nom_attr}"
    fixed_capacity = (
        c.static.loc[c.fixed, nom_attr]
        if include_non_extendable
        else pd.Series(dtype=float)
    )
    if var_name in m.variables:
        return m.variables[var_name].to_linexpr().add(fixed_capacity, join="outer")
    if fixed_capacity.empty:
        return None
    return LinearExpression.from_constant(m, fixed_capacity)


def _split_piecewise(
    expr: Variable | LinearExpression, m: ln.Model, c: Any, attr: str
) -> tuple[Any, Variable | None]:
    """Split the names handled by the `attr` piecewise curve out of `expr`."""
    if not c.has_piecewise(attr):
        return expr, None
    aux = m.variables[c._piecewise_aux_var(attr)]
    return expr.drop_sel(name=aux.coords["name"]), aux


def _add_optional(
    expr: LinearExpression, other: Variable | LinearExpression | None
) -> LinearExpression:
    """Outer-add `other` to `expr`, passing `expr` through if there is nothing to add."""
    return expr if other is None else expr.add(other, join="outer")


def _direct_piecewise(
    c: Any, y_attr: str, sign: Any, pw: Variable, names: Any, direction: str
) -> Variable | LinearExpression:
    """Restrict a piecewise contribution to the requested supply/withdrawal direction.

    A port's direction follows the sign of its breakpoints (``sign * y_attr``),
    mirroring the clipping applied to linear coefficients. Mixed-sign curves are
    rejected at ``add`` time, so each port is unambiguously supply or withdrawal.
    """
    if direction == "both":
        return pw
    y = c.piecewise[y_attr].xs(y_attr, level="attribute", axis=1)[names]
    withdraws = (sign * y < 0).any()
    if direction == "withdrawal":
        return -1 * pw.sel(name=names[withdraws])
    else:
        return pw.sel(name=names[~withdraws])


class StatisticExpressionsAccessor(AbstractStatisticsAccessor):
    """Accessor to calculate different statistical expressions.

    This class is used to calculate different statistical expressions like
    capital expenditure, capacity, energy balance, etc.
    The results are aggregated by the given groupby function.
    """

    _n: Network

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
        periods = self._n.investment_periods
        res = ln.merge(list(exprs.values()), dim=periods.name)
        return res.assign_coords({periods.name: periods})

    def _aggregate_timeseries(
        self,
        expr: LinearExpression,
        weights: DataArray,
        agg: str | Callable | bool = "sum",
    ) -> LinearExpression:
        """Weight `expr` over the snapshots it spans and aggregate them.

        Overrides the pandas-side implementation: the weights arrive as an array on
        the model's snapshot labels, so both the restriction to `expr`'s snapshots
        and the per-period mean normalization run on the `period` coordinate,
        which the snapshots carry under either representation.
        """
        if not agg:
            return expr
        if agg is True:
            agg = "sum"
        weights = weights.sel(snapshot=expr.indexes["snapshot"])
        if agg == "mean":
            weights = _normalized_weights(weights)
            agg = "sum"
        return self._aggregate_with_weights(expr, weights, agg)

    def _aggregate_with_weights(
        self,
        expr: LinearExpression,
        weights: DataArray,
        agg: str | Callable,
    ) -> LinearExpression:
        """Apply weights to a time series."""
        _require_sum_agg(agg)
        if "period" not in expr.coords:
            return expr @ weights
        return expr.mul(weights, join="left").groupby("period").sum()

    def _aggregate_components(self, *args: Any, **kwargs: Any) -> Any:
        # Expressions built from masked model variables leave absent slots, the
        # legacy groupby yields a stacked `group` MultiIndex (flattened right
        # after), and cross-component merges drop a conflicting aux coord; all are
        # handled identically here under legacy and v1, so silence the notices.
        with suppress_semantics_warnings():
            res = super()._aggregate_components(*args, **kwargs)
        return _restack_flat_groups(res)

    def _aggregate_components_skip_iteration(self, vals: Any) -> bool:
        return vals is None or (not np.prod(vals.shape) and (vals.const == 0).all())

    def _aggregate_components_groupby(
        self,
        vals: LinearExpression,
        grouping: pd.DataFrame,
        agg: Callable | str,
        c: str,
    ) -> pd.DataFrame:
        grouping = grouping.reindex(vals.indexes["name"])
        return _flatten_group_index(vals.groupby(grouping).sum())

    def _aggregate_components_concat_values(
        self, exprs: list[LinearExpression], agg: Callable | str
    ) -> LinearExpression:
        if "group" in exprs[0].dims and len(exprs) > 1:
            # concatenate the ports and sum entries sharing a key tuple
            _require_sum_agg(agg)
            merged = _concat_flat_groups(exprs)
            return _regroup_flat(merged, _group_key_coords(merged))
        return ln.merge(exprs, join="outer")

    def _aggregate_components_concat_data(
        self, res: dict[str, LinearExpression], is_one_component: bool
    ) -> LinearExpression:
        if res == {}:
            return LinearExpression.from_constant(self._n.model, 0)
        first = next(iter(res.values()))
        if "group" in first.dims:
            if is_one_component:
                keys = [k for k in _group_key_coords(first) if k != "component"]
                return _regroup_flat(first, keys) if keys else first
            return _concat_flat_groups(list(res.values()))
        # groupby=False: no `group` dim, concatenate disjoint asset names
        if is_one_component:
            return first
        return ln.merge(list(res.values()), dim="name")

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
        _require_sum_agg(agg)
        if check_if_empty(expr):
            return expr
        keys = [k for k in _group_key_coords(expr) if k != "component"]
        return _regroup_flat(expr, keys) if keys else expr

    def _get_operational_variable(self, c: str) -> Variable | LinearExpression:
        # TODO: move function to better place to avoid circular imports
        from pypsa.optimization.optimize import lookup  # noqa: PLC0415

        m = self._n.model

        if c == "Load":
            window = self._n.optimize._window
            p_set = self._n.c[c].da.p_set.sel(snapshot=window.model_index)
            return LinearExpression.from_constant(m, p_set)
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
            c = n.c[component]
            capacity = _capacity_expression(n, component, include_non_extendable)
            if capacity is None:
                return None

            capacity, add_capex = _split_piecewise(capacity, n.model, c, cost_attribute)

            if cost_attribute == "capital_cost":
                costs = c.capital_cost[capacity.indexes["name"]]
            else:
                costs = c.static[cost_attribute][capacity.indexes["name"]]
            return _add_optional(capacity * costs, add_capex)

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
            c = n.c[component]
            capacity = _capacity_expression(n, component, include_non_extendable)
            if capacity is None:
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
        weights = self._n.optimize._window.snapshot_weightings("objective")

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series | None:
            attr = "marginal_cost"
            var = lookup.query(f"not nominal and {attr}").loc[c].index.item()
            if var is None:
                return None
            var = n.model.variables[f"{c}-{var}"]
            sns = var.indexes["snapshot"]

            var, add_opex = _split_piecewise(var, n.model, n.c[c], attr)

            cost = n.c[c].da[attr].sel(snapshot=sns, name=var.indexes["name"])
            opex = _add_optional(var * cost, add_opex)
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
        weights = self._n.optimize._window.snapshot_weightings("generators")

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            var = self._get_operational_variable(c)
            sns = var.indexes["snapshot"]
            idx = transmission_branches.get_loc_level(c)[1].rename("name")
            efficiency = _port_coefficients(n, c, port, sns)
            p = var.loc[:, idx] * efficiency.sel(name=idx)
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

        weights = self._n.optimize._window.snapshot_weightings("generators")

        @pass_none_if_keyerror
        def func(n: Network, component: str, port: str) -> pd.Series:
            c = n.c[component]
            var = self._get_operational_variable(component)
            sns = var.indexes["snapshot"]
            # negative branch contributions are considered by the efficiency
            sign = n.c[component].static.get("sign", 1.0)
            coeffs = _port_coefficients(n, component, port, sns) * sign

            pw_var = None
            if isinstance(c, _Multiport) and c.has_piecewise(
                y_attr := c._port_coefficient_attr(port)
            ):
                pw = sign * n.model.variables[c._piecewise_aux_var(y_attr)]
                names = pw.coords["name"].values
                coeffs.loc[{"name": names}] = 0
                pw_var = _direct_piecewise(c, y_attr, sign, pw, names, direction)

            expr = coeffs * var.where(coeffs != 0)

            if direction in ("supply", "withdrawal"):
                if direction == "withdrawal":
                    logger.warning(
                        "The sign convention for withdrawal has changed: withdrawal values are now reported as positive numbers instead of negative numbers."
                    )
                s = 1 if direction == "supply" else -1
                expr = expr.assign(
                    coeffs=(s * expr.coeffs).clip(min=0),
                    const=(s * expr.const).clip(min=0),
                )
            elif direction != "both":
                msg = f"Got unexpected argument direction={direction}. Must be 'supply', 'withdrawal' or 'both'."
                raise ValueError(msg)

            if pw_var is not None:
                expr = expr.fillna(0)
            return self._aggregate_timeseries(
                _add_optional(expr, pw_var), weights, agg=groupby_time
            )

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
        weights = self._n.optimize._window.snapshot_weightings("generators")

        @pass_none_if_keyerror
        def func(n: Network, component: str, port: str) -> pd.Series:
            c = n.c[component]
            if "p_max_pu" not in c.static.columns:
                return None
            capacity = _capacity_expression(n, component)
            if capacity is None:
                return None

            idx = capacity.indexes["name"]
            operation = self._get_operational_variable(component).loc[:, idx]
            sns = operation.indexes["snapshot"]
            p_max_pu = c.da.p_max_pu.sel(snapshot=sns, name=idx)
            # the following needs to be fixed in linopy, right now constants cannot be used for broadcasting
            # TODO curtailment = capacity * p_max_pu - operation
            curtailment = (capacity - operation / p_max_pu) * p_max_pu
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

        weights = self._n.optimize._window.snapshot_weightings("generators")

        @pass_none_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            operation = self._get_operational_variable(c)
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
