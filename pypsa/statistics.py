# -*- coding: utf-8 -*-
"""
Statistics Accessor.
"""


__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2023 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import logging
from functools import wraps

import numpy as np
import pandas as pd

from pypsa.descriptors import nominal_attrs

logger = logging.getLogger(__name__)


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


def get_bus_and_carrier(n, c, port=""):
    """
    Get the buses and nice carrier names for a component.
    """
    if port == "":
        if "bus" not in n.df(c):
            bus = "bus0"
        else:
            bus = "bus"
    else:
        bus = f"bus{port}"
    return [n.df(c)[bus].rename("bus"), get_carrier(n, c)]


def get_country_and_carrier(n, c, port=""):
    """
    Get component country and carrier.
    """
    bus = f"bus{port}"
    bus, carrier = get_bus_and_carrier(n, c, port)
    country = bus.map(n.buses.country).rename("country")
    return [country, carrier]


def get_carrier_and_bus_carrier(n, c, port=""):
    """
    Get component carrier and bus carrier in one combined DataFrame.

    Used for MultiIndex in energy balance.
    """
    bus = f"bus{port}"
    bus_and_carrier = pd.concat(get_bus_and_carrier(n, c, port), axis=1)
    bus_carrier = n.df(c)[bus].map(n.buses.carrier).rename("bus_carrier")
    return pd.concat([bus_and_carrier, bus_carrier], axis=1)


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
    elif not agg:
        return df.T
    else:
        return df.agg(agg)


def aggregate_components(n, func, agg="sum", comps=None, groupby=None):
    """
    Apply a function and group the result for a collection of components.
    """
    d = {}
    kwargs = {}
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
        elif isinstance(groupby, dict):
            grouping = None
            kwargs = groupby
        else:
            ValueError(
                f"Argument `groupby` must be a function, list, string or dict, got {type(groupby)}"
            )
        d[c] = func(n, c).groupby(grouping, **kwargs).agg(agg)
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
        aggregate_groups="sum",
        groupby=get_carrier,
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

        Returns
        -------
        df :
            pandas.DataFrame with columns given the different quantities.
        """
        if "aggregate_time" in kwargs:
            logger.warning(
                "Argument 'aggregate_time' ignored in overview table. Falling back to individual function defaults."
            )

        funcs = [
            self.capex,
            self.optimal_capacity,
            self.installed_capacity,
            self.opex,
            self.supply,
            self.withdrawal,
            self.dispatch,
            self.curtailment,
            self.capacity_factor,
            self.revenue,
            self.market_value,
        ]
        kwargs = dict(comps=comps, aggregate_groups=aggregate_groups, groupby=groupby)
        res = []
        for func in funcs:
            df = func(**kwargs)
            res.append(df.rename(df.attrs["name"]))
        return pd.concat(res, axis=1).sort_index(axis=0).sort_index(axis=1)

    def get_carrier(self, n, c):
        """
        Get the buses and nice carrier names for a component.
        """
        return get_carrier(n, c)

    def get_bus_and_carrier(self, n, c):
        """
        Get the buses and nice carrier names for a component.
        """
        return get_bus_and_carrier(n, c)

    def get_country_and_carrier(self, n, c):
        """
        Get the country and nice carrier names for a component.
        """
        return get_country_and_carrier(n, c)

    def capex(self, comps=None, aggregate_groups="sum", groupby=None):
        """
        Calculate the capital expenditure of the network in given currency.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            return n.df(c).eval(f"{nominal_attrs[c]}_opt * capital_cost")

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["name"] = "Capital Expenditure"
        df.attrs["unit"] = "currency"
        return df

    def installed_capex(self, comps=None, aggregate_groups="sum", groupby=None):
        """
        Calculate the capital expenditure of already built components of the
        network in given currency.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            return n.df(c).eval(f"{nominal_attrs[c]} * capital_cost")

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["name"] = "Capital Expenditure Fixed"
        df.attrs["unit"] = "currency"
        return df

    def optimal_capacity(self, comps=None, aggregate_groups="sum", groupby=None):
        """
        Calculate the optimal capacity of the network components in MW.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            return n.df(c)[f"{nominal_attrs[c]}_opt"]

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["name"] = "Optimal Capacity"
        df.attrs["unit"] = "MW"
        return df

    def installed_capacity(self, comps=None, aggregate_groups="sum", groupby=None):
        """
        Calculate the installed capacity of the network components in MW.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        n = self._parent

        @pass_empty_series_if_keyerror
        def func(n, c):
            return n.df(c)[f"{nominal_attrs[c]}"]

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["name"] = "Installed Capacity"
        df.attrs["unit"] = "MW"
        return df

    def expanded_capacity(self, comps=None, aggregate_groups="sum", groupby=None):
        """
        Calculate the expanded capacity of the network components in MW.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        df = self.optimal_capacity(
            comps=comps, aggregate_groups=aggregate_groups, groupby=groupby
        ) - self.installed_capacity(
            comps=comps, aggregate_groups=aggregate_groups, groupby=groupby
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
    ):
        """
        Calculate the operational expenditure in the network in given currency.

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
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
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
    ):
        """
        Calculate the supply of components in the network. Units depend on the
        regarded bus carrier.

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
        df.attrs["name"] = "Supply"
        df.attrs["unit"] = "carrier dependent"
        return df

    def withdrawal(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the withdrawal of components in the network. Units depend on
        the regarded bus carrier.

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
        df.attrs["name"] = "Withdrawal"
        df.attrs["unit"] = "carrier dependent"
        return df

    def dispatch(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the dispatch of components in the network. Units depend on
        the regarded bus carrier.

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
            if c in n.branch_components:
                p = -n.pnl(c).p0
            else:
                p = n.pnl(c).p * n.df(c).sign
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        df.attrs["name"] = "Dispatch"
        df.attrs["unit"] = "carrier dependent"
        return df

    def energy_balance(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        aggregate_bus=True,
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

        @pass_empty_series_if_keyerror
        def func(n, c):
            sign = -1 if c in n.branch_components else n.df(c).get("sign", 1)
            ports = [col[3:] for col in n.df(c).columns if col[:3] == "bus"]
            p = list()
            for port in ports:
                mask = n.df(c)[f"bus{port}"] != ""
                df = sign * n.pnl(c)[f"p{port}"].loc[:, mask]
                index = get_carrier_and_bus_carrier(n, c, port=port)[mask]
                df.columns = pd.MultiIndex.from_frame(index.reindex(df.columns))
                p.append(df)
            p = pd.concat(p, axis=1)
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        groupby = ["carrier", "bus_carrier"]
        if not aggregate_bus:
            groupby.append("bus")

        df = aggregate_components(
            n,
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby={"level": groupby},
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
    ):
        """
        Calculate the curtailment of components in the network in MWh.

        The calculation only considers assets with a `p_max_pu` time
        series, which is used to quantify the available power potential.

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
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
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
    ):
        """
        Calculate the capacity factor of components in the network.

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
            weights = get_weightings(n, c)
            return aggregate_timeseries(p, weights, agg=aggregate_time)

        df = aggregate_components(
            n, func, comps=comps, agg=aggregate_groups, groupby=groupby
        )
        capacity = self.optimal_capacity(
            comps=comps, aggregate_groups=aggregate_groups, groupby=groupby
        )
        df = df.div(capacity, fill_value=np.nan)
        df.attrs["name"] = "Capacity Factor"
        df.attrs["unit"] = "p.u."
        return df

    def revenue(
        self,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the revenue of components in the network in given currency.

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
        df.attrs["name"] = "Revenue"
        df.attrs["unit"] = "currency"
        return df

    def market_value(
        self,
        comps=None,
        aggregate_time="mean",
        aggregate_groups="sum",
        groupby=None,
    ):
        """
        Calculate the market value of components in the network in given
        currency/MWh.

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
        )
        df = self.revenue(**kwargs) / self.dispatch(**kwargs)
        df.attrs["name"] = "Market Value"
        df.attrs["unit"] = "currency / MWh"
        return df
