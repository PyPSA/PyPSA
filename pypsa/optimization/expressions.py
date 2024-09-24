"""
Statistics Accessor.
"""

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-4 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import logging
from collections.abc import Collection, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Callable, Union

import linopy as ln
import pandas as pd
from deprecation import deprecated

from pypsa.descriptors import nominal_attrs
from pypsa.statistics import (
    Groupers,
    aggregate_timeseries,
    get_bus_and_carrier_and_bus_carrier,
    get_carrier,
    get_carrier_and_bus_carrier,
    get_operation,
    get_transmission_branches,
    get_weightings,
    port_efficiency,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypsa import Network


def get_grouping(
    n: "Network",
    c: str,
    groupby: Callable | Sequence[str] | str | bool,
    port: str | None = None,
    nice_names: bool = False,
) -> pd.DataFrame:
    from pypsa.statistics import get_grouping

    result = get_grouping(n, c, groupby, port, nice_names)
    by = result["by"]

    if isinstance(by, list):
        grouper = pd.concat(by, axis=1)
    elif isinstance(by, pd.Series):
        grouper = by.to_frame()
    else:
        grouper = by

    return grouper.rename_axis("name")


def filter_active_assets(n, c, expr: Union[ln.Variable, ln.LinearExpression]):
    """
    For static values iterate over periods and concat values.
    """
    if not isinstance(n.snapshots, pd.MultiIndex) or "snapshot" in expr.dims:
        return expr
    per_period = {}
    for p in n.investment_periods:
        idx = n.get_active_assets(c, p)[lambda x: x].index.intersection(expr.indexes[c])
        per_period[p] = expr.loc[idx]
    return ln.merge(per_period.values(), keys=per_period.keys(), dim=c)


def filter_bus_carrier(
    n,
    c: str,
    port: str,
    bus_carrier: str | list[str],
    expr: Union[ln.Variable, ln.LinearExpression],
):
    """
    Filter the DataFrame for components which are connected to a bus with
    carrier `bus_carrier`.
    """
    if bus_carrier is None:
        return expr

    ports = n.df(c).loc[expr.indexes[c], f"bus{port}"]
    port_carriers = ports.map(n.buses.carrier)
    if isinstance(bus_carrier, str):
        if bus_carrier in n.buses.carrier.unique():
            idx = (port_carriers == bus_carrier)[lambda x: x].index
        else:
            idx = (bus_carrier in port_carriers)[lambda x: x].index
    elif isinstance(bus_carrier, list):
        idx = port_carriers.isin(bus_carrier)[lambda x: x].index
    else:
        raise ValueError(
            f"Argument `bus_carrier` must be a string or list, got {type(bus_carrier)}"
        )
    return expr.loc[idx]


def pass_none_if_keyerror(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyError, AttributeError):
            return None

    return wrapper


class StatisticExpressionsAccessor:
    """
    Accessor to calculate different statistical values.
    """

    def __init__(self, network):
        self._parent = network
        self.groupers = Groupers()  # Create an instance of the Groupers class

    def _aggregate_components(
        self,
        func: Callable,
        agg: Callable | str | bool = "sum",
        comps: Collection[str] | str | None = None,
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = True,
    ):
        """
        Apply a function and group the result for a collection of components.
        """
        n = self._parent

        res = []
        if not getattr(n, "model"):
            raise ValueError(
                "Model not created. Run `n.optimize.create_model()` first."
            )

        if agg != "sum":
            raise ValueError("Only 'sum' aggregation of components is supported.")

        if is_one_component := isinstance(comps, str):
            comps = [comps]
        if comps is None:
            comps = n.branch_components | n.one_port_components
        if groupby is None:
            groupby = get_carrier
        for c in comps:
            if n.df(c).empty:
                continue

            ports = [col[3:] for col in n.df(c) if col.startswith("bus")]
            if not at_port:
                ports = [ports[0]]

            exprs = []
            for port in ports:
                vals = func(n, c, port)
                if vals is None or vals.empty():
                    continue
                vals = filter_active_assets(n, c, vals)  # for multiinvest
                vals = filter_bus_carrier(n, c, port, bus_carrier, vals)
                vals = vals.rename({c: "name"})

                if groupby is not False:
                    grouping = get_grouping(
                        n, c, groupby, port=port, nice_names=nice_names
                    )
                    grouping.insert(0, "component", c)  # for tracking the component
                    vals = vals.groupby(grouping).sum()
                elif not is_one_component:
                    vals = vals.expand_dims({"component": [c]}).stack(
                        group=["component", "name"]
                    )
                exprs.append(vals)

            if not len(exprs):
                continue

            expr = ln.merge(exprs)
            res.append(expr)

        if res == {}:
            return ln.LinearExpression(None, n.model)
        if is_one_component:
            return res[0]
        return ln.merge(res, dim="group")

    def capex(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        at_port=False,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the capital expenditure of the network in given currency.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """

        @pass_none_if_keyerror
        def func(n, c, port):
            m = n.model
            costs = n.df(c)["capital_cost"]
            capacity = m.variables[f"{c}-{nominal_attrs[c]}"]
            capacity = capacity.rename({f"{c}-ext": c})
            return capacity * costs[capacity.indexes[c]]

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return df

    def installed_capex(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        at_port=False,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the capital expenditure of already built components of the
        network in given currency.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """

        @pass_none_if_keyerror
        def func(n, c, port):
            col = n.df(c).eval(f"{nominal_attrs[c]} * capital_cost")
            return col

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df.attrs["name"] = "Capital Expenditure Fixed"
        df.attrs["unit"] = "currency"
        return df

    def expanded_capex(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        at_port=False,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the capex of expanded capacities of the network components in
        currency.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        df = self.capex(
            comps=comps,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        ).sub(
            self.installed_capex(
                comps=comps,
                aggregate_groups=aggregate_groups,
                groupby=groupby,
                at_port=at_port,
                bus_carrier=bus_carrier,
                nice_names=nice_names,
            ),
            fill_value=0,
        )
        df.attrs["name"] = "Capital Expenditure Expanded"
        df.attrs["unit"] = "currency"
        return df

    def optimal_capacity(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        at_port=False,
        bus_carrier=None,
        storage=False,
        nice_names=True,
    ):
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

        @pass_none_if_keyerror
        def func(n, c, port):
            efficiency = abs(port_efficiency(n, c, port=port))
            col = n.df(c)[f"{nominal_attrs[c]}_opt"] * efficiency
            if storage and (c == "StorageUnit"):
                col = col * n.df(c).max_hours
            return col

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df.attrs["name"] = "Optimal Capacity"
        df.attrs["unit"] = "MW"
        return df

    def installed_capacity(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        at_port=False,
        bus_carrier=None,
        storage=False,
        nice_names=True,
    ):
        """
        Calculate the installed capacity of the network components in MW.

        If `at_port` is True, the capacity is weighed by the output efficiency
        of components at buses with carrier `bus_carrier`.

        If storage is set to True, only storage capacities of the component
        `Store` and `StorageUnit` are taken into account.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """

        if storage:
            comps = ("Store", "StorageUnit")

        @pass_none_if_keyerror
        def func(n, c, port):
            efficiency = abs(port_efficiency(n, c, port=port))
            col = n.df(c)[f"{nominal_attrs[c]}"] * efficiency
            if storage and (c == "StorageUnit"):
                col = col * n.df(c).max_hours
            return col

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df.attrs["name"] = "Installed Capacity"
        df.attrs["unit"] = "MW"
        return df

    def expanded_capacity(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        at_port=False,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the expanded capacity of the network components in MW.

        If `bus_carrier` is given, the capacity is weighed by the output efficiency
        of components at buses with carrier `bus_carrier`.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        df = self.optimal_capacity(
            comps=comps,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        ).sub(
            self.installed_capacity(
                comps=comps,
                aggregate_groups=aggregate_groups,
                groupby=groupby,
                at_port=at_port,
                bus_carrier=bus_carrier,
                nice_names=nice_names,
            ),
            fill_value=0,
        )
        df.attrs["name"] = "Expanded Capacity"
        df.attrs["unit"] = "MW"
        return df

    def opex(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
        at_port=False,
        bus_carrier=None,
        nice_names=True,
    ):
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
            using snapshot weightings. With False the time series is given. Defaults to 'sum'.
        """

        @pass_none_if_keyerror
        def func(n, c, port):
            if c in n.branch_components:
                p = n.pnl(c).p0
            elif c == "StorageUnit":
                p = n.pnl(c).p_dispatch
            else:
                p = n.pnl(c).p

            opex = p * n.get_switchable_as_dense(c, "marginal_cost")
            weights = get_weightings(n, c)
            return aggregate_timeseries(opex, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df.attrs["name"] = "Operational Expenditure"
        df.attrs["unit"] = "currency"
        return df

    def supply(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
        at_port=True,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the supply of components in the network. Units depend on the
        regarded bus carrier.

        If `bus_carrier` is given, only the supply to buses with carrier
        `bus_carrier` is calculated.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """

        df = self.energy_balance(
            comps=comps,
            aggregate_time=aggregate_time,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            kind="supply",
        )
        df.attrs["name"] = "Supply"
        df.attrs["unit"] = "carrier dependent"
        return df

    def withdrawal(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
        at_port=True,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the withdrawal of components in the network. Units depend on
        the regarded bus carrier.

        If `bus_carrier` is given, only the withdrawal from buses with
        carrier `bus_carrier` is calculated.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """

        df = self.energy_balance(
            comps=comps,
            aggregate_time=aggregate_time,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            kind="withdrawal",
        )
        df.attrs["name"] = "Withdrawal"
        df.attrs["unit"] = "carrier dependent"
        return df

    # TODO: remove in 0.29
    @deprecated(
        deprecated_in="0.28", removed_in="0.29", details="Use 'energy_balance' instead."
    )
    def dispatch(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
        at_port=True,
        bus_carrier=None,
        nice_names=True,
        kind=None,
    ):
        """
        Calculate the dispatch of components in the network. Units depend on
        the regarded bus carrier.

        If `bus_carrier` is given, only the dispatch to and from buses with
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

        df = self.energy_balance(
            comps=comps,
            aggregate_time=aggregate_time,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            kind=kind,
        )
        df.attrs["name"] = "Dispatch"
        df.attrs["unit"] = "carrier dependent"
        return df

    def transmission(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
        at_port=False,
        bus_carrier=None,
        nice_names=True,
    ):
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
        n = self._parent
        if comps is None:
            comps = n.branch_components

        transmission_branches = get_transmission_branches(n, bus_carrier)

        @pass_none_if_keyerror
        def func(n, c, port):
            p = n.pnl(c)[f"p{port}"][transmission_branches.get_loc_level(c)[1]]
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df.attrs["name"] = "Transmission"
        df.attrs["unit"] = "carrier dependent"
        return df

    def energy_balance(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        aggregate_bus=True,
        groupby=get_carrier_and_bus_carrier,
        at_port=True,
        bus_carrier=None,
        nice_names=True,
        kind=None,
    ):
        """
        Calculate the energy balance of components in the network. Positive
        values represent a supply and negative a withdrawal. Units depend on
        the regarded bus carrier.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Additional parameter
        ----------
        aggregate_bus: bool, optional
            Whether to obtain the nodal or carrier-wise energy balance. Default is True, corresponding to the carrier-wise balance.
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to MWh
            using snapshot weightings. With False the time series is given in MW. Defaults to 'sum'.
        """

        n = self._parent

        if (
            n.buses.carrier.unique().size > 1
            and groupby is None
            and bus_carrier is None
        ):
            logger.warning(
                "Network has multiple bus carriers which are aggregated together. To separate bus carriers set `bus_carrier` or use groupers like `get_carrier_and_bus_carrier`."
            )

        # TODO remove aggregate_bus in 0.29
        if not aggregate_bus:
            if groupby != get_carrier_and_bus_carrier:
                logger.warning(
                    "Argument 'groupby' is ignored when 'aggregate_bus' is set to True. Falling back to default."
                )
            logger.warning(
                "Argument 'aggregate_bus' is deprecated in 0.28 and will be removed in 0.29. Use grouper `get_bus_and_carrier_and_bus_carrier` instead."
            )
            switch = True
            groupby = get_bus_and_carrier_and_bus_carrier
        else:
            switch = False

        @pass_none_if_keyerror
        def func(n, c, port):
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            weights = get_weightings(n, c)
            p = sign * n.pnl(c)[f"p{port}"]
            if kind == "supply":
                p = p.clip(lower=0)
            elif kind == "withdrawal":
                p = -p.clip(upper=0)
            elif kind is not None:
                logger.warning(
                    "Argument 'kind' is not recognized. Falling back to energy balance."
                )
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )

        if switch:
            df = df.reorder_levels(["component", "carrier", "bus_carrier", "bus"])
        df.attrs["name"] = "Energy Balance"
        df.attrs["unit"] = "carrier dependent"
        return df

    def curtailment(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
        at_port=False,
        bus_carrier=None,
        nice_names=True,
    ):
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
        def func(n, c, port):
            p = (n.pnl(c).p_max_pu * n.df(c).p_nom_opt - n.pnl(c).p).clip(lower=0)
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df.attrs["name"] = "Curtailment"
        df.attrs["unit"] = "MWh"
        return df

    def capacity_factor(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        at_port=False,
        groupby=None,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the capacity factor of components in the network.

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
        def func(n, c, port):
            p = get_operation(n, c).abs()
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        kwargs = dict(
            comps=comps,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df = self._aggregate_components(func, agg=aggregate_groups, **kwargs)
        capacity = self.optimal_capacity(aggregate_groups=aggregate_groups, **kwargs)
        df = df.div(capacity.reindex(df.index), axis=0)
        df.attrs["name"] = "Capacity Factor"
        df.attrs["unit"] = "p.u."
        return df

    def revenue(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
        at_port=True,
        bus_carrier=None,
        nice_names=True,
        kind=None,
    ):
        """
        Calculate the revenue of components in the network in given currency.
        The revenue is defined as the net revenue of an asset, i.e cost of input - revenue of output.
        If kind is set to "input" or "output" only the revenue of the input or output is considered.

        If `bus_carrier` is given, only the revenue resulting from buses with carrier
        `bus_carrier` is considered.


        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to
            using snapshot weightings. With False the time series is given. Defaults to 'sum'.
        kind : str, optional
            Type of revenue to consider. If 'input' only the revenue of the input is considered.
            If 'output' only the revenue of the output is considered. Defaults to None.
        """

        @pass_none_if_keyerror
        def func(n, c, port):
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            df = sign * n.pnl(c)[f"p{port}"]
            buses = n.df(c)[f"bus{port}"][df.columns]
            prices = n.buses_t.marginal_price.reindex(
                columns=buses, fill_value=0
            ).values
            if kind is not None:
                if kind == "input":
                    df = df.clip(upper=0)
                elif kind == "output":
                    df = df.clip(lower=0)
                else:
                    raise ValueError(
                        f"Argument 'kind' must be 'input', 'output' or None, got {kind}"
                    )
            revenue = df * prices
            weights = get_weightings(n, c)
            return aggregate_timeseries(revenue, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df.attrs["name"] = "Revenue"
        df.attrs["unit"] = "currency"
        return df

    def market_value(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=None,
        at_port=True,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the market value of components in the network in given
        currency/MWh or currency/unit_{bus_carrier} where unit_{bus_carrier} is
        the unit of the bus carrier.

        If `bus_carrier` is given, only the market value resulting from buses with
        carrier `bus_carrier` are calculated.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to
            using snapshot weightings. With False the time series is given. Defaults to 'mean'.
        """
        kwargs = dict(
            comps=comps,
            aggregate_time=aggregate_time,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df = self.revenue(**kwargs) / self.supply(**kwargs)
        df.attrs["name"] = "Market Value"
        df.attrs["unit"] = "currency / MWh"
        return df
