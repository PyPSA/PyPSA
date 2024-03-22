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

import inspect
import logging
from functools import reduce, wraps

import numpy as np
import pandas as pd

from pypsa.descriptors import nominal_attrs

logger = logging.getLogger(__name__)


def get_grouper(n, c, port="", groupby=None, nice_names=True):
    if groupby == None:
        grouper = get_carrier(n, c, port=port, nice_names=nice_names)
    elif callable(groupby):
        parameters = inspect.signature(groupby).parameters
        if "nice_names" in parameters and "port" in parameters:
            grouper = groupby(n, c, port=port, nice_names=nice_names)
        else:
            grouper = groupby(n, c)
    elif isinstance(groupby, list):
        grouper = [n.df(c)[key] for key in groupby]
    elif isinstance(groupby, str):
        grouper = n.df(c)[groupby]
    else:
        grouper = []
    if not isinstance(grouper, list):
        grouper = [grouper]
    index = n.df(c).index
    grouper = [pd.Series(index, index=index, name="name"), *grouper]
    return grouper


def get_carrier(n, c, port="", nice_names=True):
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
    if port == "":
        bus = "bus0" if "bus" not in n.df(c) else "bus"
    else:
        bus = f"bus{port}"
    return [n.df(c)[bus].rename("bus"), get_carrier(n, c, port, nice_names=nice_names)]


def get_name_bus_and_carrier(n, c, port="", nice_names=True):
    """
    Get the name, buses and nice carrier names for a component.
    """
    return [
        n.df(c).index.to_series().rename("component_name"),
        *get_bus_and_carrier(n, c, port, nice_names=nice_names),
    ]


def get_name_and_bus_and_carrier_and_bus_carrier(n, c, port="", nice_names=True):
    """
    Get component's name, carrier, bus and bus carrier in one combined list.

    Used for MultiIndex in energy balance.
    """
    bus_and_carrier = get_bus_and_carrier(n, c, port, nice_names=nice_names)
    bus_carrier = bus_and_carrier[0].map(n.buses.carrier).rename("bus_carrier")
    return [*get_name_bus_and_carrier(n, c, port, nice_names=nice_names), bus_carrier]


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


def get_ports(n, c):
    """
    Get a list of existent ports of a component.
    """
    return [col[3:] for col in n.df(c) if col.startswith("bus")]


def port_mask(n, c, port="", bus_carrier=None):
    """
    Get a mask of components which are optionally connected to a bus with a
    given carrier.
    """
    if bus_carrier is None:
        mask = n.df(c)[f"bus{port}"] != ""
    elif isinstance(bus_carrier, str):
        if bus_carrier in n.buses.carrier.unique():
            mask = n.df(c)[f"bus{port}"].map(n.buses.carrier) == bus_carrier
        else:
            mask = (
                n.df(c)[f"bus{port}"]
                .map(n.buses.carrier)
                .str.contains(bus_carrier)
                .fillna(False)
            )
    elif isinstance(bus_carrier, list):
        mask = n.df(c)[f"bus{port}"].map(n.buses.carrier).isin(bus_carrier)
    else:
        raise ValueError(
            f"Argument `bus_carrier` must be a string or list, got {type(bus_carrier)}"
        )
    return mask


def port_efficiency(n, c, port="", bus_carrier=None):
    mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
    if port == "":
        efficiency = n.df(c).get("efficiency", 1)
    elif port == "0":
        efficiency = -1
    elif port == "1":
        efficiency = n.df(c).get("efficiency", 1)
    else:
        efficiency = n.df(c)[f"efficiency{port}"]
    return mask * efficiency


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


def aggregate_components(
    n,
    func,
    comps=None,
    aggregate_groups="sum",
):
    """
    Apply a function and group the result for a collection of components.
    """
    d = {}
    kwargs = {}

    if is_one_component := isinstance(comps, str):
        comps = [comps]
    if comps is None:
        comps = n.branch_components | n.one_port_components
    for c in comps:
        if n.df(c).empty:
            continue

        df = func(n, c)
        if df.index.empty:
            continue
        if isinstance(n.snapshots, pd.MultiIndex) and not isinstance(df, pd.DataFrame):
            # for static values we have to iterate over periods and concat
            per_period = {}
            for p in n.investment_periods:
                per_period[p] = df[
                    n.get_active_assets(c, p).reindex(df.index, level="name")
                ]
            df = pd.concat(per_period, axis=1)
        if df.index.nlevels > 1:
            df = df.droplevel("name")
        df = df.groupby(level=df.index.names).agg(aggregate_groups)
        d[c] = df

    if d == {}:
        return pd.Series([])
    if is_one_component:
        return d[c]
    index_level_names = set.union(*[set(df.index.names) for df in d.values()])
    if not all(df.index.nlevels == len(index_level_names) for df in d.values()):
        index_level_names = []
    return pd.concat(d, names=["component", *index_level_names])


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

        funcs = [
            self.optimal_capacity,
            self.installed_capacity,
            self.capacity_factor,
            self.dispatch,
            self.transmission,
            self.withdrawal,
            self.supply,
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
            if df.empty:
                continue
            res[df.attrs["name"]] = df
        index = pd.Index(set.union(*[set(df.index) for df in res.values()]))
        res = {k: v.reindex(index, fill_value=0.0) for k, v in res.items()}
        return pd.concat(res, axis=1).sort_index(axis=0)

    def capex(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
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
        def func(n, c):
            col = n.df(c).eval(f"{nominal_attrs[c]}_opt * capital_cost")
            if bus_carrier is None:
                grouper = get_grouper(n, c, groupby=groupby, nice_names=nice_names)
                ds = col.groupby(grouper).agg(aggregate_groups)
                return ds
            else:
                sign = n.df(c).get("sign", 1)
                ports = get_ports(n, c)
                df = list()
                for port in ports:
                    mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                    ds = mask * col
                    col.mask(mask, 0, inplace=True)
                    ds.clip(lower=0, inplace=True)
                    grouper = get_grouper(
                        n, c, port=port, nice_names=nice_names, groupby=groupby
                    )
                    ds = ds.groupby(grouper).agg(aggregate_groups)
                    df.append(ds)
                df = pd.concat(df)
                df = df.groupby(df.index.names).agg(aggregate_groups)
                return df[df != 0]

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Total Capital Expenditure"
        df.attrs["unit"] = "currency"
        return df

    def installed_capex(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the capital expenditure of already built components of the
        network in given currency.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            col = n.df(c).eval(f"{nominal_attrs[c]} * capital_cost")
            if bus_carrier is None:
                grouper = get_grouper(n, c, groupby=groupby, nice_names=nice_names)
                ds = col.groupby(grouper).agg(aggregate_groups)
                return ds
            else:
                sign = n.df(c).get("sign", 1)
                ports = get_ports(n, c)
                df = list()
                for port in ports:
                    mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                    ds = mask * col
                    col.mask(mask, 0, inplace=True)
                    ds.clip(lower=0, inplace=True)
                    grouper = get_grouper(
                        n, c, port=port, nice_names=nice_names, groupby=groupby
                    )
                    ds = ds.groupby(grouper).agg(aggregate_groups)
                    df.append(ds)
                df = pd.concat(df)
                df = df.groupby(df.index.names).agg(aggregate_groups)
                return df[df != 0]

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Installed Capital Expenditure"
        df.attrs["unit"] = "currency"
        return df

    def expanded_capex(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the capital expenditure of the expanded components of the
        network in given currency. I.e. the difference between the capex and
        installed capex.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """

        df = self.capex(
            comps=comps,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
            nice_names=nice_names,
            bus_carrier=bus_carrier,
        ).sub(
            self.installed_capex(
                comps=comps,
                aggregate_groups=aggregate_groups,
                groupby=groupby,
                nice_names=nice_names,
                bus_carrier=bus_carrier,
            ),
            fill_value=0,
        )
        df.attrs["name"] = "Expanded Capital Expenditure"
        df.attrs["unit"] = "currency"
        return df

    def optimal_capacity(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
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
        def func(n, c):
            col = n.df(c)[f"{nominal_attrs[c]}_opt"]
            if storage and (c == "StorageUnit"):
                col = col * n.df(c).max_hours
            if bus_carrier is None:
                grouper = get_grouper(n, c, groupby=groupby, nice_names=nice_names)
                ds = col.groupby(grouper).agg(aggregate_groups)
                return ds
            else:
                sign = n.df(c).get("sign", 1)
                ports = get_ports(n, c)
                df = list()
                for port in ports:
                    efficiency = port_efficiency(
                        n, c, port=port, bus_carrier=bus_carrier
                    )
                    ds = sign * efficiency * col
                    ds.clip(lower=0, inplace=True)
                    grouper = get_grouper(
                        n, c, port=port, nice_names=nice_names, groupby=groupby
                    )
                    ds = ds.groupby(grouper).agg(aggregate_groups)
                    df.append(ds)
                df = pd.concat(df)
                df = df.groupby(df.index.names).agg(aggregate_groups)
                return df[df != 0]

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Optimal Capacity"
        df.attrs["unit"] = "MW"
        return df

    def installed_capacity(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        bus_carrier=None,
        storage=False,
        nice_names=True,
    ):
        """
        Calculate the installed capacity of the network components in MW.

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
        def func(n, c):
            col = n.df(c)[f"{nominal_attrs[c]}"]
            if storage and (c == "StorageUnit"):
                col = col * n.df(c).max_hours
            if bus_carrier is None:
                grouper = get_grouper(n, c, groupby=groupby, nice_names=nice_names)
                ds = col.groupby(grouper).agg(aggregate_groups)
                return ds
            else:
                sign = n.df(c).get("sign", 1)
                ports = get_ports(n, c)
                df = list()
                for port in ports:
                    efficiency = port_efficiency(
                        n, c, port=port, bus_carrier=bus_carrier
                    )
                    ds = sign * efficiency * col
                    ds.clip(lower=0, inplace=True)
                    grouper = get_grouper(
                        n, c, port=port, nice_names=nice_names, groupby=groupby
                    )
                    ds = ds.groupby(grouper).agg(aggregate_groups)
                    df.append(ds)
                df = pd.concat(df)
                df = df.groupby(df.index.names).agg(aggregate_groups)
                return df[df != 0]

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Installed Capacity"
        df.attrs["unit"] = "MW"
        return df

    def expanded_capacity(
        self,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        bus_carrier=None,
        storage=False,
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
            nice_names=nice_names,
            storage=storage,
            bus_carrier=bus_carrier,
        ).sub(
            self.installed_capacity(
                comps=comps,
                aggregate_groups=aggregate_groups,
                groupby=groupby,
                nice_names=nice_names,
                storage=storage,
                bus_carrier=bus_carrier,
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
        def func(n, c):
            marginal_cost = n.get_switchable_as_dense(c, "marginal_cost")
            if c in n.branch_components:
                p = n.pnl(c).p0
            elif c == "StorageUnit":
                p = n.pnl(c).p_dispatch
            else:
                p = n.pnl(c).p
            if bus_carrier is None:
                opex = p * marginal_cost
                grouper = get_grouper(n, c, groupby=groupby, nice_names=nice_names)
                opex = opex.T.groupby(grouper).agg(aggregate_groups).T
            else:
                ports = get_ports(n, c)
                df = list()
                for port in ports:
                    mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                    opex = mask * p * marginal_cost
                    grouper = get_grouper(
                        n, c, port=port, nice_names=nice_names, groupby=groupby
                    )
                    opex = opex.T.groupby(grouper).agg(aggregate_groups).T
                    df.append(opex)
                opex = pd.concat(df, axis=1)

            opex = opex.T.groupby(opex.columns.names).agg(aggregate_groups).T
            opex = opex.loc[:, (opex != 0).any()]
            weights = get_weightings(n, c)
            return aggregate_timeseries(opex, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Operational Expenditure"
        df.attrs["unit"] = "currency"
        return df

    def supply(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=get_carrier_and_bus_carrier,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the supply of components in the network. It is the positive
        part of the energy balance. Units depend on the regarded bus carrier.

        If `bus_carrier` is given, only the supply to buses with carrier
        `bus_carrier` is calculated.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            ports = get_ports(n, c)
            p = []
            for port in ports:
                mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                df = n.pnl(c)[f"p{port}"]
                df = sign * mask * df.reindex(columns=mask.index, fill_value=0)
                df.clip(lower=0, inplace=True)
                grouper = get_grouper(
                    n, c, port=port, nice_names=nice_names, groupby=groupby
                )
                df = df.T.groupby(grouper).agg(aggregate_groups).T
                p.append(df)
            p = pd.concat(p, axis=1)
            p = p.T.groupby(p.columns.names).agg(aggregate_groups).T
            p = p.loc[:, (p != 0).any()]

            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
            aggregate_groups=aggregate_groups,
        )
        df.attrs["name"] = "Supply"
        df.attrs["unit"] = "carrier dependent"
        return df

    def withdrawal(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=get_carrier_and_bus_carrier,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the withdrawal of components in the network. It is the
        negative part of the energy balance. Units depend on the regarded bus
        carrier.

        If `bus_carrier` is given, only the withdrawal from buses with
        carrier `bus_carrier` is calculated.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            ports = get_ports(n, c)
            p = list()
            for port in ports:
                mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                df = n.pnl(c)[f"p{port}"]
                df = sign * mask * df.reindex(columns=mask.index, fill_value=0)
                df = -df.clip(upper=0)
                grouper = get_grouper(
                    n, c, port=port, nice_names=nice_names, groupby=groupby
                )
                df = df.T.groupby(grouper).agg(aggregate_groups).T
                p.append(df)
            p = pd.concat(p, axis=1)
            p = p.T.groupby(p.columns.names).agg(aggregate_groups).T
            p = p.loc[:, (p != 0).any()]

            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Withdrawal"
        df.attrs["unit"] = "carrier dependent"
        return df

    def dispatch(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=get_carrier_and_bus_carrier,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the dispatch of components in the network, i.e. the active
        power 'p' which is dispatched by a component. Units depend on the
        regarded bus carrier.

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

        @pass_empty_series_if_keyerror
        def func(n, c):
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            port = ""
            mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
            p = n.pnl(c)[f"p{port}"]
            p = sign * p.reindex(columns=mask.index, fill_value=0)
            grouper = get_grouper(
                n, c, port=port, nice_names=nice_names, groupby=groupby
            )
            p = p.T.groupby(grouper).agg(aggregate_groups).T
            p = p.loc[:, (p != 0).any()]

            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
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
        def func(n, c):
            p = n.pnl(c).p0[transmission_branches.get_loc_level(c)[1]]
            weights = get_weightings(n, c)
            grouper = get_grouper(n, c, groupby=groupby, nice_names=nice_names)
            p = p.T.groupby(grouper).agg(aggregate_groups).T
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Transmission"
        df.attrs["unit"] = "carrier dependent"
        return df

    def energy_balance(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        aggregate_bus=False,
        groupby=get_carrier_and_bus_carrier,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the energy balance of components in the network. Positive
        values represent a supply and negative a withdrawal. Units depend on
        the regarded bus carrier.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Additional parameter
        ----------
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated to MWh
            using snapshot weightings. With False the time series is given in MW. Defaults to 'sum'.
        """
        n = self._parent

        if aggregate_bus:
            logger.warning(
                "Argument 'aggregate_bus' is deprecated. Use groupby instead with grouper function 'get_bus_and_carrier_and_bus_carrier' from statistics.groupers. Falling back to grouper function 'get_bus_and_carrier_and_bus_carrier'."
            )
            grouper = get_bus_and_carrier_and_bus_carrier

        @pass_empty_series_if_keyerror
        def func(n, c):
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            ports = get_ports(n, c)
            p = list()
            for port in ports:
                mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                df = n.pnl(c)[f"p{port}"]
                df = sign * mask * df.reindex(columns=mask.index, fill_value=0)
                grouper = get_grouper(
                    n, c, port=port, nice_names=nice_names, groupby=groupby
                )
                df = df.T.groupby(grouper).agg(aggregate_groups).T
                p.append(df)
            p = pd.concat(p, axis=1)
            p = p.T.groupby(p.columns.names).agg(aggregate_groups).T
            p = p.loc[:, (p != 0).any()]

            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Energy Balance"
        df.attrs["unit"] = "carrier dependent"
        return df

    def curtailment(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
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
        def func(n, c):
            p = (n.pnl(c).p_max_pu * n.df(c).p_nom_opt - n.pnl(c).p).clip(lower=0)
            if bus_carrier is None:
                grouper = get_grouper(n, c, groupby=groupby, nice_names=nice_names)
                p = p.T.groupby(grouper).agg(aggregate_groups).T
            else:
                ports = get_ports(n, c)
                df = list()
                for port in ports:
                    mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                    p_i = mask * p.reindex(columns=mask.index)
                    grouper = get_grouper(
                        n, c, port=port, nice_names=nice_names, groupby=groupby
                    )
                    p_i = p_i.T.groupby(grouper).agg(aggregate_groups).T
                    df.append(p_i)
                p = pd.concat(df, axis=1)
            p = p.T.groupby(p.columns.names).agg(aggregate_groups).T
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Curtailment"
        df.attrs["unit"] = "MWh"
        return df

    def capacity_factor(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
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
        def func(n, c):
            p = get_operation(n, c).abs()

            if bus_carrier is None:
                grouper = get_grouper(n, c, groupby=groupby, nice_names=nice_names)
                p = p.T.groupby(grouper).agg(aggregate_groups).T
            else:
                ports = get_ports(n, c)
                df = list()
                for port in ports:
                    mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                    p_i = mask * p.reindex(columns=mask.index)
                    grouper = get_grouper(
                        n, c, port=port, nice_names=nice_names, groupby=groupby
                    )
                    p_i = p_i.T.groupby(grouper).agg(aggregate_groups).T
                    df.append(p_i)
                p = pd.concat(df, axis=1)
            p = p.T.groupby(p.columns.names).agg(aggregate_groups).T

            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        capacity = self.optimal_capacity(
            comps=comps, aggregate_groups=aggregate_groups, groupby=groupby
        )
        if not df.empty:
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
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the revenue of components in the network in given currency.
        The revenue is calculated as the difference between the output revenues
        and the input cost. Units depend on the regarded bus carrier.

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
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            ports = get_ports(n, c)
            revenue = list()
            for port in ports:
                mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                p = n.pnl(c)[f"p{port}"]
                p = sign * mask * p.reindex(columns=mask.index, fill_value=0)
                buses = n.df(c)[f"bus{port}"].reindex(p.columns)
                prices = n.buses_t.marginal_price.reindex(columns=buses, fill_value=0)
                revenue_i = p * prices.values
                grouper = get_grouper(
                    n, c, port=port, nice_names=nice_names, groupby=groupby
                )
                revenue_i = revenue_i.T.groupby(grouper).agg(aggregate_groups).T
                revenue.append(revenue_i)
            revenue = pd.concat(revenue, axis=1)
            revenue = revenue.T.groupby(revenue.columns.names).agg(aggregate_groups).T

            revenue = revenue.loc[:, (revenue != 0).any()]
            weights = get_weightings(n, c)
            return aggregate_timeseries(revenue, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Revenue"
        df.attrs["unit"] = "currency"
        return df

    def input_cost(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the cost for the input of components in the network in given
        currency. The cost is calculated as the product of the input power and
        the marginal price of the buses. Units depend on the regarded bus
        carrier.

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
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            ports = get_ports(n, c)
            revenue = list()
            for port in ports:
                mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                p = n.pnl(c)[f"p{port}"]
                p = sign * mask * p.reindex(columns=mask.index, fill_value=0)
                buses = n.df(c)[f"bus{port}"].reindex(p.columns)
                prices = n.buses_t.marginal_price.reindex(columns=buses, fill_value=0)
                revenue_i = p.clip(lower=0) * prices.values
                grouper = get_grouper(
                    n, c, port=port, nice_names=nice_names, groupby=groupby
                )
                revenue_i = revenue_i.T.groupby(grouper).agg(aggregate_groups).T
                revenue.append(revenue_i)
            revenue = pd.concat(revenue, axis=1)
            revenue = revenue.T.groupby(revenue.columns.names).agg(aggregate_groups).T

            revenue = revenue.loc[:, (revenue != 0).any()]
            weights = get_weightings(n, c)
            return aggregate_timeseries(revenue, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Input Cost"
        df.attrs["unit"] = "currency"
        return df

    def output_revenue(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the revenue the output of components in the network in given
        currency. The revenue is calculated as the product of the output power
        and the marginal price of the buses. Units depend on the regarded bus
        carrier.

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
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            sign = -1.0 if c in n.branch_components else n.df(c).get("sign", 1.0)
            ports = get_ports(n, c)
            revenue = list()
            for port in ports:
                mask = port_mask(n, c, port=port, bus_carrier=bus_carrier)
                p = n.pnl(c)[f"p{port}"]
                p = sign * mask * p.reindex(columns=mask.index, fill_value=0)
                buses = n.df(c)[f"bus{port}"].reindex(p.columns)
                prices = n.buses_t.marginal_price.reindex(columns=buses, fill_value=0)
                revenue_i = -p.clip(upper=0) * prices.values
                grouper = get_grouper(
                    n, c, port=port, nice_names=nice_names, groupby=groupby
                )
                revenue_i = revenue_i.T.groupby(grouper).agg(aggregate_groups).T
                revenue.append(revenue_i)
            revenue = pd.concat(revenue, axis=1)
            revenue = revenue.T.groupby(revenue.columns.names).agg(aggregate_groups).T

            revenue = revenue.loc[:, (revenue != 0).any()]
            weights = get_weightings(n, c)
            return aggregate_timeseries(revenue, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
        )
        df.attrs["name"] = "Output Revenue"
        df.attrs["unit"] = "currency"
        return df

    def market_value(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=None,
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
            nice_names=nice_names,
            bus_carrier=bus_carrier,
        )
        df = self.revenue(**kwargs) / self.supply(**kwargs)
        df.attrs["name"] = "Market Value"
        df.attrs["unit"] = "currency / MWh"
        return df
