"""
Statistics Accessor.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Sequence
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypsa import Network

import pandas as pd

from pypsa.descriptors import nominal_attrs

logger = logging.getLogger(__name__)


def get_carrier(n: Network, c: str, nice_names: bool = True) -> pd.Series:
    """
    Get the nice carrier names for a component.
    """
    static = n.static(c)
    fall_back = pd.Series("", index=static.index)
    carrier_series = static.get("carrier", fall_back).rename("carrier")
    if nice_names:
        carrier_series = carrier_series.replace(
            n.carriers.nice_name[lambda ds: ds != ""]
        ).replace("", "-")
    return carrier_series


def get_bus_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> pd.Series:
    """
    Get the bus carrier for a component.
    """
    bus = f"bus{port}"
    buses_carrier = get_carrier(n, "Bus", nice_names=nice_names)
    return n.static(c)[bus].map(buses_carrier).rename("bus_carrier")


def get_bus_and_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get the buses and nice carrier names for a component.
    """
    bus = f"bus{port}"
    return [n.static(c)[bus].rename("bus"), get_carrier(n, c, nice_names=nice_names)]


def get_bus_unit_and_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get the buses and nice carrier names for a component.
    """
    bus = f"bus{port}"
    return [
        n.static(c)[bus].rename("bus"),
        n.static(c)[bus].map(n.buses.unit).rename("unit"),
        get_carrier(n, c, nice_names=nice_names),
    ]


def get_name_bus_and_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get the name, buses and nice carrier names for a component.
    """
    return [
        n.static(c).index.to_series().rename("name"),
        *get_bus_and_carrier(n, c, port, nice_names=nice_names),
    ]


def get_country_and_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get component country and carrier.
    """
    # bus = f"bus{port}"
    bus, carrier = get_bus_and_carrier(n, c, port, nice_names=nice_names)
    country = bus.map(n.buses.country).rename("country")
    return [country, carrier]


def get_bus_and_carrier_and_bus_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get component's carrier, bus and bus carrier in one combined list.

    Used for MultiIndex in energy balance.
    """
    bus_and_carrier = get_bus_and_carrier(n, c, port, nice_names=nice_names)
    bus_carrier = get_bus_carrier(n, c, port, nice_names=nice_names)
    return [*bus_and_carrier, bus_carrier]


def get_carrier_and_bus_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get component carrier and bus carrier in one combined list.
    """
    carrier = get_carrier(n, c, nice_names=nice_names)
    bus_carrier = get_bus_carrier(n, c, port, nice_names=nice_names)
    return [carrier, bus_carrier]


def get_operation(n: Network, c: str) -> pd.DataFrame:
    """
    Get the operation time series of a component.
    """
    if c in n.branch_components:
        return n.dynamic(c).p0
    elif c == "Store":
        return n.dynamic(c).e
    else:
        return n.dynamic(c).p


def get_weightings(n: Network, c: str) -> pd.Series:
    """
    Get the relevant snapshot weighting for a component.
    """
    if c == "Generator":
        return n.snapshot_weightings["generators"]
    elif c in ["StorageUnit", "Store"]:
        return n.snapshot_weightings["stores"]
    else:
        return n.snapshot_weightings["objective"]


def port_efficiency(n: Network, c: str, port: str = "") -> float:
    if port == "":
        efficiency = 1
    elif port == "0":
        efficiency = -1
    elif port == "1":
        efficiency = n.static(c).get("efficiency", 1)
    else:
        efficiency = n.static(c).get(f"efficiency{port}", 1)
    return efficiency


def get_transmission_branches(
    n: Network, bus_carrier: Sequence[str] | str | None = None
) -> pd.MultiIndex:
    """
    Get the list of assets which transport between buses of the carrier
    `bus_carrier`.
    """
    index = {}
    for c in n.branch_components:
        bus_map = (
            n.static(c).filter(like="bus").apply(lambda ds: ds.map(n.buses.carrier))
        )
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


def get_transmission_carriers(
    n: Network, bus_carrier: Sequence[str] | str | None = None
) -> pd.MultiIndex:
    """
    Get the carriers which transport between buses of the carrier
    `bus_carrier`.
    """
    branches = get_transmission_branches(n, bus_carrier)
    carriers = {}
    for c in branches.unique(0):
        idx = branches[branches.get_loc(c)].get_level_values(1)
        carriers[c] = n.static(c).carrier[idx].unique()
    return pd.MultiIndex.from_tuples(
        [(c, i) for c, idx in carriers.items() for i in idx],
        names=["component", "carrier"],
    )


def get_grouping(
    n: Network,
    c: str,
    groupby: Callable | Sequence[str] | str | bool,
    port: str | None = None,
    nice_names: bool = False,
) -> pd.Series | list:
    by = None
    level = None
    if callable(groupby):
        if "port" in signature(groupby).parameters:
            by = groupby(n, c, port=port, nice_names=nice_names)
        else:
            by = groupby(n, c, nice_names=nice_names)
    elif isinstance(groupby, list):
        by = [n.static(c)[key] for key in groupby]
    elif isinstance(groupby, str):
        by = n.static(c)[groupby]
    elif groupby is not False:
        raise ValueError(
            f"Argument `groupby` must be a function, list, string, False or dict, got {type(groupby)}"
        )
    return dict(by=by, level=level)


def aggregate_timeseries(
    df: pd.DataFrame, weights: pd.Series, agg: str | bool = "sum"
) -> pd.Series:
    """
    Calculate the weighted sum or average of a DataFrame or Series.
    """
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


def filter_active_assets(
    n: Network, c: str, df: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    For static values iterate over periods and concat values.
    """

    if isinstance(df, pd.DataFrame):
        return df
    if not isinstance(n.snapshots, pd.MultiIndex):
        return df[n.get_active_assets(c).loc[df.index]]
    per_period = {}
    for p in n.snapshots.unique(0):
        per_period[p] = df[n.get_active_assets(c, p).loc[df.index]]
    return pd.concat(per_period, axis=1)


def filter_bus_carrier(
    n: Network,
    c: str,
    port: str,
    bus_carrier: Sequence[str] | str | None,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter the DataFrame for components which are connected to a bus with
    carrier `bus_carrier`.
    """
    if bus_carrier is None:
        return df

    ports = n.static(c).loc[df.index, f"bus{port}"]
    port_carriers = ports.map(n.buses.carrier)
    if isinstance(bus_carrier, str):
        if bus_carrier in n.buses.carrier.unique():
            return df[port_carriers == bus_carrier]
        else:
            return df[port_carriers.str.contains(bus_carrier).fillna(False)]
    elif isinstance(bus_carrier, list):
        return df[port_carriers.isin(bus_carrier)]
    else:
        raise ValueError(
            f"Argument `bus_carrier` must be a string or list, got {type(bus_carrier)}"
        )


def pass_empty_series_if_keyerror(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> pd.Series:
        try:
            return func(*args, **kwargs)
        except (KeyError, AttributeError):
            return pd.Series([], dtype=float)

    return wrapper


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


class Groupers:
    """
    Container for all the get_ methods.
    """

    get_carrier = staticmethod(get_carrier)
    get_bus_carrier = staticmethod(get_bus_carrier)
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

    def __init__(self, n: Network) -> None:
        self.n = n
        self.groupers = Groupers()  # Create an instance of the Groupers class
        self.parameters = Parameters()  # Create an instance of the Parameters class

    def set_parameters(self, **kwargs: Any) -> None:
        """
        Setting the parameters for the statistics accessor.

        To see the list of parameters, one can simply call `n.statistics.parameters`.
        """
        self.parameters.set_parameters(**kwargs)

    def _aggregate_components(
        self,
        func: Callable,
        agg: Callable | str | bool = "sum",
        comps: Collection[str] | str | None = None,
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = True,
    ) -> pd.DataFrame:
        """
        Apply a function and group the result for a collection of components.
        """
        df: pd.DataFrame
        d = {}
        n = self.n

        if is_one_component := isinstance(comps, str):
            comps = [comps]
        if comps is None:
            comps = n.branch_components | n.one_port_components
        if groupby is None:
            groupby = get_carrier
        if nice_names is None:
            nice_names = self.parameters.nice_names
        for c in comps:
            if n.static(c).empty:
                continue

            ports = [col[3:] for col in n.static(c) if col.startswith("bus")]
            if not at_port:
                ports = [ports[0]]

            df = []
            for port in ports:
                vals = func(n, c, port)
                vals = filter_active_assets(n, c, vals)  # for multiinvest
                vals = filter_bus_carrier(n, c, port, bus_carrier, vals)

                # unit tracker
                if groupby is not False:
                    grouping = get_grouping(
                        n, c, groupby, port=port, nice_names=nice_names
                    )
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
        df = pd.concat(d, names=index_names)
        if self.parameters.round:
            df = df.round(self.parameters.round)
        if self.parameters.drop_zero:
            df = df[df != 0]
        return df

    def __call__(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable = get_carrier,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
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

        funcs: list[Callable] = [
            self.optimal_capacity,
            self.installed_capacity,
            self.supply,
            self.withdrawal,
            self.energy_balance,
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
        comps: Sequence[str] | str | None = None,
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = False,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
        cost_attribute: str = "capital_cost",
    ) -> pd.DataFrame:
        """
        Calculate the capital expenditure of the network in given currency.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        cost_attribute : str
            Network attribute that should be used to calculate Capital Expenditure. Defaults to `capital_cost`.
        """

        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            col = n.static(c).eval(f"{nominal_attrs[c]}_opt * {cost_attribute}")
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
        df.attrs["name"] = "Capital Expenditure"
        df.attrs["unit"] = "currency"
        return df

    def installed_capex(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = False,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
        cost_attribute: str = "capital_cost",
    ) -> pd.DataFrame:
        """
        Calculate the capital expenditure of already built components of the
        network in given currency.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        cost_attribute : str
            Network attribute that should be used to calculate Capital Expenditure. Defaults to `capital_cost`.
        """

        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            col = n.static(c).eval(f"{nominal_attrs[c]} * {cost_attribute}")
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
        comps: Sequence[str] | str | None = None,
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = False,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
        cost_attribute: str = "capital_cost",
    ) -> pd.DataFrame:
        """
        Calculate the capex of expanded capacities of the network components in
        currency.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        cost_attribute : str
            Network attribute that should be used to calculate Capital Expenditure. Defaults to `capital_cost`.
        """
        df = self.capex(
            comps=comps,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            cost_attribute=cost_attribute,
        ).sub(
            self.installed_capex(
                comps=comps,
                aggregate_groups=aggregate_groups,
                groupby=groupby,
                at_port=at_port,
                bus_carrier=bus_carrier,
                nice_names=nice_names,
                cost_attribute=cost_attribute,
            ),
            fill_value=0,
        )
        df.attrs["name"] = "Capital Expenditure Expanded"
        df.attrs["unit"] = "currency"
        return df

    def optimal_capacity(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        storage: bool = False,
        nice_names: bool | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the optimal capacity of the network components in MW.
        Positive capacity values correspond to production capacities and
        negative values to consumption capacities.

        If `bus_carrier` is given, the capacity is weighted by the output efficiency of `bus_carrier`.

        If storage is set to True, only storage capacities of the component
        `Store` and `StorageUnit` are taken into account.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """

        if storage:
            comps = ("Store", "StorageUnit")
        if bus_carrier and at_port is None:
            at_port = True

        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            efficiency = port_efficiency(n, c, port=port)
            if not at_port:
                efficiency = abs(efficiency)
            col = n.static(c)[f"{nominal_attrs[c]}_opt"] * efficiency
            if storage and (c == "StorageUnit"):
                col = col * n.static(c).max_hours
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
        comps: Sequence[str] | str | None = None,
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        storage: bool = False,
        nice_names: bool | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the installed capacity of the network components in MW.
        Positive capacity values correspond to production capacities and
        negative values to consumption capacities.

        If `bus_carrier` is given, the capacity is weighted by the output efficiency of `bus_carrier`.

        If storage is set to True, only storage capacities of the component
        `Store` and `StorageUnit` are taken into account.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """

        if storage:
            comps = ("Store", "StorageUnit")
        if bus_carrier and at_port is None:
            at_port = True

        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            efficiency = port_efficiency(n, c, port=port)
            if not at_port:
                efficiency = abs(efficiency)
            col = n.static(c)[f"{nominal_attrs[c]}"] * efficiency
            if storage and (c == "StorageUnit"):
                col = col * n.static(c).max_hours
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
        comps: Sequence[str] | str | None = None,
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the expanded capacity of the network components in MW.
        Positive capacity values correspond to production capacities and
        negative values to consumption capacities.

        If `bus_carrier` is given, the capacity is weighted by the output efficiency of `bus_carrier`.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """
        optimal = self.optimal_capacity(
            comps=comps,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        installed = self.installed_capacity(
            comps=comps,
            aggregate_groups=aggregate_groups,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        installed = installed.reindex(optimal.index, fill_value=0)
        df = optimal.sub(installed).where(optimal.abs() > installed.abs(), 0)
        df.attrs["name"] = "Expanded Capacity"
        df.attrs["unit"] = "MW"
        return df

    def opex(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = False,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> pd.DataFrame:
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

        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            if c in n.branch_components:
                p = n.dynamic(c).p0
            elif c == "StorageUnit":
                p = n.dynamic(c).p_dispatch
            else:
                p = n.dynamic(c).p

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
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = True,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> pd.DataFrame:
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
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = True,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> pd.DataFrame:
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

    def transmission(
        self,
        comps: Collection[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = False,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> pd.DataFrame:
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
        n = self.n

        if comps is None:
            comps = n.branch_components

        transmission_branches = get_transmission_branches(n, bus_carrier)

        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            p = n.dynamic(c)[f"p{port}"][transmission_branches.get_loc_level(c)[1]]
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
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = get_carrier_and_bus_carrier,
        at_port: Sequence[str] | str | bool = True,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
        kind: str | None = None,
    ) -> pd.DataFrame:
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
        n = self.n

        if (
            n.buses.carrier.unique().size > 1
            and groupby is None
            and bus_carrier is None
        ):
            logger.warning(
                "Network has multiple bus carriers which are aggregated together. To separate bus carriers set `bus_carrier` or use groupers like `get_carrier_and_bus_carrier`."
            )

        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            sign = -1.0 if c in n.branch_components else n.static(c).get("sign", 1.0)
            weights = get_weightings(n, c)
            p = sign * n.dynamic(c)[f"p{port}"]
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

        df.attrs["name"] = "Energy Balance"
        df.attrs["unit"] = "carrier dependent"
        return df

    def curtailment(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = False,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> pd.DataFrame:
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

        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            p = (
                n.get_switchable_as_dense(c, "p_max_pu") * n.static(c).p_nom_opt
                - n.dynamic(c).p
            ).clip(lower=0)
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
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "mean",
        aggregate_groups: Callable | str | bool = "sum",
        at_port: Sequence[str] | str | bool = False,
        groupby: Callable | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> pd.DataFrame:
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

        # TODO: Why not just take p_max_pu, s_max_pu, etc. directly from the network?
        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
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
        df = self._aggregate_components(func, agg=aggregate_groups, **kwargs)  # type: ignore
        capacity = self.optimal_capacity(aggregate_groups=aggregate_groups, **kwargs)  # type: ignore
        df = df.div(capacity.reindex(df.index), axis=0)
        df.attrs["name"] = "Capacity Factor"
        df.attrs["unit"] = "p.u."
        return df

    def revenue(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = True,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
        kind: str | None = None,
    ) -> pd.DataFrame:
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
            using snapshot weightings. With False the time series is given in currency/hour. Defaults to 'sum'.
        kind : str, optional
            Type of revenue to consider. If 'input' only the revenue of the input is considered.
            If 'output' only the revenue of the output is considered. Defaults to None.

        """

        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            sign = -1.0 if c in n.branch_components else n.static(c).get("sign", 1.0)
            df = sign * n.dynamic(c)[f"p{port}"]
            buses = n.static(c)[f"bus{port}"][df.columns]
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
        comps: Sequence[str] | str | None = None,
        aggregate_time: str | bool = "mean",
        aggregate_groups: Callable | str | bool = "sum",
        groupby: Callable | None = None,
        at_port: Sequence[str] | str | bool = True,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = None,
    ) -> pd.DataFrame:
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
        df = self.revenue(**kwargs) / self.supply(**kwargs)  # type: ignore
        df.attrs["name"] = "Market Value"
        df.attrs["unit"] = "currency / MWh"
        return df
