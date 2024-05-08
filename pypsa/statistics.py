# -*- coding: utf-8 -*-
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
from functools import reduce, wraps
from inspect import signature

import numpy as np
import pandas as pd
from deprecation import deprecated

from pypsa.descriptors import nominal_attrs

logger = logging.getLogger(__name__)


def get_carrier(n, c, nice_names=True):
    """
    Get the nice carrier names for a component.
    """
    df = n.df(c)
    fall_back = pd.Series("", index=df.index)
    carrier_series = df.get("carrier", fall_back).rename("carrier")
    if nice_names:
        carrier_series = carrier_series.replace(
            n.carriers.nice_name[lambda ds: ds != ""]
        ).replace("", "-")
    return carrier_series


def get_bus_and_carrier(n, c, port="", nice_names=True):
    """
    Get the buses and nice carrier names for a component.
    """
    bus = f"bus{port}"
    return [n.df(c)[bus].rename("bus"), get_carrier(n, c, nice_names=nice_names)]


def get_bus_unit_and_carrier(n, c, port="", nice_names=True):
    """
    Get the buses and nice carrier names for a component.
    """
    bus = f"bus{port}"
    return [
        n.df(c)[bus].rename("bus"),
        n.df(c)[bus].map(n.buses.unit).rename("unit"),
        get_carrier(n, c, nice_names=nice_names),
    ]


def get_name_bus_and_carrier(n, c, port="", nice_names=True):
    """
    Get the name, buses and nice carrier names for a component.
    """
    return [
        n.df(c).index.to_series().rename("name"),
        *get_bus_and_carrier(n, c, port, nice_names=nice_names),
    ]


def get_country_and_carrier(n, c, port="", nice_names=True):
    """
    Get component country and carrier.
    """
    bus = f"bus{port}"
    bus, carrier = get_bus_and_carrier(n, c, port, nice_names=nice_names)
    country = bus.map(n.buses.country).rename("country")
    return [country, carrier]


def get_bus_and_carrier_and_bus_carrier(n, c, port="", nice_names=True):
    """
    Get component's carrier, bus and bus carrier in one combined list.

    Used for MultiIndex in energy balance.
    """
    bus_and_carrier = get_bus_and_carrier(n, c, port, nice_names=nice_names)
    bus_carrier = bus_and_carrier[0].map(n.buses.carrier).rename("bus_carrier")
    return [*bus_and_carrier, bus_carrier]


def get_carrier_and_bus_carrier(n, c, port="", nice_names=True):
    """
    Get component carrier and bus carrier in one combined list.
    """
    bus, carrier = get_bus_and_carrier(n, c, port, nice_names=nice_names)
    bus_carrier = bus.map(n.buses.carrier).rename("bus_carrier")
    return [carrier, bus_carrier]


def get_operation(n, c):
    """
    Get the operation time series of a component.
    """
    if c in n.branch_components:
        return n.pnl(c).p0
    elif c == "Store":
        return n.pnl(c).e
    else:
        return n.pnl(c).p


def get_weightings(n, c):
    """
    Get the relevant snapshot weighting for a component.
    """
    if c == "Generator":
        return n.snapshot_weightings["generators"]
    elif c in ["StorageUnit", "Store"]:
        return n.snapshot_weightings["stores"]
    else:
        return n.snapshot_weightings["objective"]


# TODO: remove in 0.29
@deprecated(deprecated_in="0.28", removed_in="0.29")
def get_ports(n, c):
    """
    Get a list of existent ports of a component.
    """
    return [col[3:] for col in n.df(c) if col.startswith("bus")]


def port_efficiency(n, c, port=""):
    if (port == "") or "efficiency" not in n.df(c):
        efficiency = 1
    elif port == "0":
        efficiency = -1
    elif port == "1":
        efficiency = n.df(c)["efficiency"]
    else:
        efficiency = n.df(c)[f"efficiency{port}"]
    return efficiency


def get_transmission_branches(n, bus_carrier=None):
    """
    Get the list of assets which transport between buses of the carrier
    `bus_carrier`.
    """

    index = {}
    for c in n.branch_components:
        bus_map = n.df(c).filter(like="bus").apply(lambda ds: ds.map(n.buses.carrier))
        if isinstance(bus_carrier, str):
            bus_carrier = [bus_carrier]
        elif bus_carrier is None:
            bus_carrier = n.buses.carrier.unique()
        res = set()
        for carrier in bus_carrier:
            res |= set(
                bus_map.eq(carrier).astype(int).sum(axis=1)[lambda ds: ds > 1].index
            )
        index[c] = pd.Index(res)
    return pd.MultiIndex.from_tuples(
        [(c, i) for c, idx in index.items() for i in idx], names=["component", "name"]
    )


def get_transmission_carriers(n, bus_carrier=None):
    """
    Get the carriers which transport between buses of the carrier
    `bus_carrier`.
    """
    branches = get_transmission_branches(n, bus_carrier)
    carriers = {}
    for c in branches.unique(0):
        idx = branches[branches.get_loc(c)].get_level_values(1)
        carriers[c] = n.df(c).carrier[idx].unique()
    return pd.MultiIndex.from_tuples(
        [(c, i) for c, idx in carriers.items() for i in idx],
        names=["component", "name"],
    )


def get_grouping(n, c, groupby, port=None, nice_names=False) -> [pd.Series, list]:
    by = None
    level = None
    if callable(groupby):
        if "port" in signature(groupby).parameters:
            by = groupby(n, c, port=port, nice_names=nice_names)
        else:
            by = groupby(n, c, nice_names=nice_names)
    elif isinstance(groupby, list):
        by = [n.df(c)[key] for key in groupby]
    elif isinstance(groupby, str):
        by = n.df(c)[groupby]
    elif groupby is not False:
        ValueError(
            f"Argument `groupby` must be a function, list, string, False or dict, got {type(groupby)}"
        )
    return dict(by=by, level=level)


def aggregate_timeseries(df, weights, agg="sum"):
    "Calculate the weighted sum or average of a DataFrame or Series."
    if isinstance(df.index, pd.MultiIndex):
        if agg == "mean":
            weights = weights.groupby(level=0).transform(lambda w: w / w.sum())
            return df.multiply(weights, axis=0).groupby(level=0).sum().T
        elif agg == "sum":
            return df.multiply(weights, axis=0).groupby(level=0).sum().T
        elif not agg:
            return df.T
    else:
        if agg == "mean":
            return (weights / weights.sum()) @ df
        elif agg == "sum":
            return weights @ df
        elif not agg:
            return df.T
    return df.agg(agg)


def filter_active_assets(n, c, df: [pd.Series, pd.DataFrame]):
    """
    For static values iterate over periods and concat values.
    """
    if not isinstance(n.snapshots, pd.MultiIndex) or isinstance(df, pd.DataFrame):
        return df
    per_period = {}
    for p in n.snapshots.get_level_values(0):
        per_period[p] = df[n.get_active_assets(c, p).loc[df.index]]
    return pd.concat(per_period, axis=1)


def filter_bus_carrier(n, c, port, bus_carrier, df):
    """
    Filter the DataFrame for components which are connected to a bus with
    carrier `bus_carrier`.
    """
    if bus_carrier is None:
        return df

    ports = n.df(c).loc[df.index, f"bus{port}"]
    port_carriers = ports.map(n.buses.carrier)
    if isinstance(bus_carrier, str):
        if bus_carrier in n.buses.carrier.unique():
            return df[port_carriers == bus_carrier]
        else:
            return df[bus_carrier in port_carriers]
    elif isinstance(bus_carrier, list):
        return df[port_carriers.isin(bus_carrier)]
    else:
        raise ValueError(
            f"Argument `bus_carrier` must be a string or list, got {type(bus_carrier)}"
        )


def aggregate_components(
    n,
    func,
    agg="sum",
    comps=None,
    groupby=None,
    at_port=None,
    bus_carrier=None,
    nice_names=True,
):
    """
    Apply a function and group the result for a collection of components.
    """
    d = {}

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

        df = []
        for port in ports:
            vals = func(n, c, port)
            vals = filter_active_assets(n, c, vals)  # for multiinvest
            vals = filter_bus_carrier(n, c, port, bus_carrier, vals)

            # unit tracker
            if groupby is not False:
                grouping = get_grouping(n, c, groupby, port=port, nice_names=nice_names)
                vals = vals.groupby(**grouping).agg(agg)
            df.append(vals)

        df = pd.concat(df, copy=False) if len(df) > 1 else df[0]
        if not df.index.is_unique:
            df = df.groupby(level=df.index.names).agg(agg)
        d[c] = df

    if d == {}:
        return pd.Series([])
    if is_one_component:
        return d[c]
    index_names = ["component"] + df.index.names
    return pd.concat(d, names=index_names)[lambda ds: ds != 0]


def pass_empty_series_if_keyerror(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyError, AttributeError):
            return pd.Series([], dtype=float)

    return wrapper


class Groupers:
    """
    Container for all the 'get_' methods.
    """

    get_carrier = staticmethod(get_carrier)
    get_bus_and_carrier = staticmethod(get_bus_and_carrier)
    get_name_bus_and_carrier = staticmethod(get_name_bus_and_carrier)
    get_country_and_carrier = staticmethod(get_country_and_carrier)
    get_carrier_and_bus_carrier = staticmethod(get_carrier_and_bus_carrier)
    get_bus_and_carrier_and_bus_carrier = staticmethod(
        get_bus_and_carrier_and_bus_carrier
    )
    get_bus_unit_and_carrier = staticmethod(get_bus_unit_and_carrier)


class StatisticsAccessor:
    """
    Accessor to calculate different statistical values.
    """

    def __init__(self, network):
        self._parent = network
        self.groupers = Groupers()  # Create an instance of the Groupers class

    def __call__(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=get_carrier,
        nice_names=True,
        **kwargs,
    ):
        """
        Calculate statistical values for a network.

        This function calls multiple function in the background in order to
        derive a full table of relevant network information. It groups the
        values to components according to the groupby argument.

        Parameters
        ----------
        comps: list-like
            Set of components to consider. Defaults to one-port and branch
            components.
        aggregate_groups : str, optional
            Type of aggregation when component groups. The default is 'sum'.
        groupby : callable, list, str, optional
            Specification how to group assets within one component class.
            If a function is passed, it should have the arguments network and
            component name. If a list is passed it should contain
            column names of the static DataFrame, same for a single string.
            Defaults to `get_carrier`.
        nice_names : bool, optional
            Whether to use the nice names of the carrier. Defaults to True.

        Returns
        -------
        df :
            pandas.DataFrame with columns given the different quantities.
        """
        if "aggregate_time" in kwargs:
            logger.warning(
                "Argument 'aggregate_time' ignored in overview table. Falling back to individual function defaults."
            )
            kwargs.pop("aggregate_time")

        # TODO replace dispatch by energy_balance

        funcs = [
            self.optimal_capacity,
            self.installed_capacity,
            self.supply,
            self.withdrawal,
            self.dispatch,
            self.transmission,
            self.capacity_factor,
            self.curtailment,
            self.capex,
            self.opex,
            self.revenue,
            self.market_value,
        ]

        res = {}
        for func in funcs:
            df = func(
                comps=comps,
                aggregate_groups=aggregate_groups,
                groupby=groupby,
                nice_names=nice_names,
                **kwargs,
            )
            res[df.attrs["name"]] = df
        index = pd.Index(set.union(*[set(df.index) for df in res.values()]))
        res = {k: v.reindex(index, fill_value=0.0) for k, v in res.items()}
        return pd.concat(res, axis=1).sort_index(axis=0)

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
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c, port):
            col = n.df(c).eval(f"{nominal_attrs[c]}_opt * capital_cost")
            return col

        df = aggregate_components(
            n,
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df.attrs["name"] = "Capital Expenditure"
        df.attrs["unit"] = "currency"
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
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c, port):
            col = n.df(c).eval(f"{nominal_attrs[c]} * capital_cost")
            return col

        df = aggregate_components(
            n,
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
        n = self._parent

        if storage:
            comps = ("Store", "StorageUnit")

        @pass_empty_series_if_keyerror
        def func(n, c, port):
            efficiency = abs(port_efficiency(n, c, port=port))
            col = n.df(c)[f"{nominal_attrs[c]}_opt"] * efficiency
            if storage and (c == "StorageUnit"):
                col = col * n.df(c).max_hours
            return col

        df = aggregate_components(
            n,
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
        n = self._parent

        if storage:
            comps = ("Store", "StorageUnit")

        @pass_empty_series_if_keyerror
        def func(n, c, port):
            efficiency = abs(port_efficiency(n, c, port=port))
            col = n.df(c)[f"{nominal_attrs[c]}"] * efficiency
            if storage and (c == "StorageUnit"):
                col = col * n.df(c).max_hours
            return col

        df = aggregate_components(
            n,
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
        n = self._parent

        @pass_empty_series_if_keyerror
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

        df = aggregate_components(
            n,
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
        n = self._parent

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
        n = self._parent

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

        n = self._parent

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

        @pass_empty_series_if_keyerror
        def func(n, c, port):
            p = n.pnl(c)[f"p{port}"][transmission_branches.get_loc_level(c)[1]]
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
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

        @pass_empty_series_if_keyerror
        def func(n, c, port):
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            weights = get_weightings(n, c)
            p = sign * n.pnl(c)[f"p{port}"]
            if kind == "supply":
                p = p.clip(lower=0)
            elif kind == "withdrawal":
                p = -p.clip(upper=0)
            elif kind != None:
                logger.warning(
                    f"Argument 'kind' is not recognized. Falling back to energy balance."
                )
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )

        if switch:
            res = df.reorder_levels(["component", "carrier", "bus_carrier", "bus"])
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
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c, port):
            p = (n.pnl(c).p_max_pu * n.df(c).p_nom_opt - n.pnl(c).p).clip(lower=0)
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
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
        n = self._parent

        @pass_empty_series_if_keyerror
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
        df = aggregate_components(n, func, agg=aggregate_groups, **kwargs)
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
        n = self._parent

        @pass_empty_series_if_keyerror
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

        df = aggregate_components(
            n,
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
