# -*- coding: utf-8 -*-

"""
Statistics Accessor.
"""


__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2022 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

from functools import wraps

import numpy as np
import pandas as pd

from pypsa.descriptors import nominal_attrs


def get_carrier(n, c):
    """
    Get the nice carrier names for a component.
    """
    df = n.df(c)
    fall_back = pd.Series("", index=df.index)
    return (
        df.get("carrier", fall_back)
        .replace(n.carriers.nice_name[lambda ds: ds != ""])
        .replace("", "-")
        .rename("carrier")
    )


def get_bus_and_carrier(n, c):
    """
    Get the buses and nice carrier names for a component.
    """
    if "bus" not in n.df(c):
        bus = "bus0"
    else:
        bus = "bus"
    return [n.df(c)[bus].rename("bus"), get_carrier(n, c)]


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


def aggregate_timeseries(df, weights, agg="sum"):
    "Calculate the weighed sum or average of a DataFrame or Series."
    if agg == "mean":
        return df.multiply(weights, axis=0).sum().div(weights.sum())
    elif agg == "sum":
        return weights @ df
    else:
        return df.agg(agg)


def aggregate_components(n, func, agg="sum", comps=None, groupby=None):
    """
    Apply a function and group the result for a collection of components.
    """
    d = {}
    if comps is None:
        comps = n.branch_components | n.one_port_components
    if groupby is None:
        groupby = get_carrier
    for c in comps:
        if callable(groupby):
            grouping = groupby(n, c)
        elif isinstance(groupby, list):
            grouping = [n.df(c)[key] for key in groupby]
        elif isinstance(groupby, str):
            grouping = n.df(c)[groupby]
        else:
            ValueError(
                f"Argument `groupby` must be a function, list or string, got {type(groupby)}"
            )
        d[c] = func(n, c).groupby(grouping).agg(agg)
    return pd.concat(d)


def pass_empty_series_if_keyerror(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyError, AttributeError):
            return pd.Series([], dtype=float)

    return wrapper


class StatisticsAccessor:
    """
    Accessor to calculate different statistical values.
    """

    def __init__(self, network):
        self._parent = network

    def __call__(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=get_carrier,
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
        aggregate_time : str, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time-series are aggregated
            using snapshot weightings. The default is 'mean'.
        aggregate_groups : str, optional
            Type of aggregation when component groups. The default is 'sum'.
        groupby : callable, list, str, optional
            Specification how to group assets within one component class.
            If a function is passed, it should have the arguments network and
            component name. If a list is passed it should contain
            column names of the static dataframe, same for a single string.
            Defaults to `get_carrier`.

        Returns
        -------
        df :
            pandas.DataFrame with columns given the different quantities.
        """
        static_funcs = [
            self.capex,
            self.optimal_capacity,
            self.installed_capacity,
        ]
        dynamic_funcs = [
            self.opex,
            self.supply,
            self.withdrawal,
            self.curtailment,
            self.capacity_factor,
            self.revenue,
        ]
        kwargs = dict(comps=comps, aggregate_groups=aggregate_groups, groupby=groupby)
        res = []
        for func in static_funcs:
            res.append(func(**kwargs))
        for func in dynamic_funcs:
            res.append(func(aggregate_time=aggregate_time, **kwargs))
        return pd.concat(res, axis=1).sort_index(axis=0).sort_index(axis=1)

    def get_carrier(self, c):
        """
        Get the buses and nice carrier names for a component.
        """
        return get_carrier(self._parent, c)

    def get_bus_and_carrier(self, c):
        """
        Get the buses and nice carrier names for a component.
        """
        return get_bus_and_carrier(self._parent, c)

    def capex(self, comps=None, aggregate_groups="sum", groupby=None):
        """
        Calculate the capital expenditure of the network.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            return n.df(c).eval(f"{nominal_attrs[c]}_opt * capital_cost")

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["unit"] = "€"
        return df.rename("Capital Expenditure")

    def optimal_capacity(self, comps=None, aggregate_groups="sum", groupby=None):
        """
        Calculate the optimal capacity of the network components.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            return n.df(c)[f"{nominal_attrs[c]}_opt"]

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["unit"] = "MW"
        return df.rename("Optimal Capacity")

    def installed_capacity(self, comps=None, aggregate_groups="sum", groupby=None):
        """
        Calculate the installed capacity of the network components.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            return n.df(c)[f"{nominal_attrs[c]}"]

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["unit"] = "MW"
        return df.rename("Installed Capacity")

    def expanded_capacity(self, comps=None, aggregate_groups="sum", groupby=None):
        """
        Calculate the expanded capacity of the network components.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        df = self.optimal_capacity(
            comps=comps, aggregate_groups=aggregate_groups, groupby=groupby
        ) - self.installed_capacity(
            comps=comps, aggregate_groups=aggregate_groups, groupby=groupby
        )
        df.attrs["unit"] = "MW"
        return df.rename("Expanded Capacity")

    def opex(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the operational expenditure in the network.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            if c in n.branch_components:
                p = n.pnl(c).p0
            else:
                p = n.pnl(c).p
            opex = p * n.get_switchable_as_dense(c, "marginal_cost")
            weights = get_weightings(n, c)
            return aggregate_timeseries(opex, weights, agg=aggregate_time)

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["unit"] = "€"
        return df.rename("Operational Expenditure")

    def supply(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the supply of components in the network.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            if c in n.branch_components:
                p = -n.pnl(c).p0.clip(upper=0)
                p -= n.pnl(c).p1.clip(upper=0)
            else:
                p = (n.pnl(c).p * n.df(c).sign).clip(lower=0)
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["unit"] = "MWh"
        return df.rename("Supply")

    def withdrawal(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the withdrawal of components in the network.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            if c in n.branch_components:
                p = -(n.pnl(c).p0).clip(lower=0)
                p -= n.pnl(c).p1.clip(lower=0)
            else:
                p = (n.pnl(c).p * n.df(c).sign).clip(upper=0)
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["unit"] = "MWh"
        return df.rename("Withdrawal")

    def curtailment(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the curtailment of components in the network.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            p = (n.pnl(c).p_max_pu * n.df(c).p_nom_opt - n.pnl(c).p).clip(upper=0)
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["unit"] = "MWh"
        return df.rename("Curtailment")

    def capacity_factor(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the capacity factor of components in the network.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            p = get_operation(n, c).abs()
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        capacity = self.optimal_capacity(
            comps=comps, aggregate_groups=aggregate_groups, groupby=groupby
        )
        df = df.div(capacity, fill_value=np.nan)
        df.attrs["unit"] = "p.u."
        return df.rename("Capacity Factor")

    def revenue(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the revenue of components in the network.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            if c in n.one_port_components:
                prices = n.buses_t.marginal_price.reindex(columns=n.df(c).bus)
                prices.columns = n.df(c).index
                revenue = n.pnl(c).p * prices
            else:
                prices0 = n.buses_t.marginal_price.reindex(columns=n.df(c).bus0)
                prices0.columns = n.df(c).index
                prices1 = n.buses_t.marginal_price.reindex(columns=n.df(c).bus1)
                prices1.columns = n.df(c).index
                revenue = -(n.pnl(c).p0 * prices0 + n.pnl(c).p1 * prices1)
            weights = get_weightings(n, c)
            return aggregate_timeseries(revenue, weights, agg=aggregate_time)

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["unit"] = "€"
        return df.rename("Revenue")

    def market_value(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the market value of components in the network.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statitics.StatisticsAccessor`.
        """
        kwargs = dict(
            comps=comps,
            aggregate_time=aggregate_time,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
        )
        df = self.revenue(**kwargs) / self.supply(**kwargs)
        df.attrs["unit"] = "€ / MWh"
        return df.rename("Market Value")
