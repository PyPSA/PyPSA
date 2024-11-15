"""
Statistics Accessor.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Any

from numpy import prod

if TYPE_CHECKING:
    from pypsa import Network

import pandas as pd
from deprecation import deprecated

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


def get_bus(n: Network, c: str, port: str = "") -> pd.Series:
    """
    Get the buses for a component.
    """
    bus = f"bus{port}"
    return n.static(c)[bus].rename("bus")


def get_country(n: Network, c: str, port: str = "") -> pd.Series:
    """
    Get the country for a component.
    """
    bus = f"bus{port}"
    return n.static(c)[bus].map(n.buses.country).rename("country")


def get_unit(n: Network, c: str, port: str = "") -> pd.Series:
    """
    Get the unit for a component.
    """
    bus = f"bus{port}"
    return n.static(c)[bus].map(n.buses.unit).rename("unit")


def get_name(n: Network, c: str) -> pd.Series:
    """
    Get the name for a component.
    """
    return n.static(c).index.to_series().rename("name")


def get_bus_and_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get the buses and nice carrier names for a component.
    """
    return LOCAL_GROUPERS.create_grouper(["bus", "carrier"])(
        n, c, port, nice_names=nice_names
    )


def get_bus_unit_and_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get the buses and nice carrier names for a component.
    """
    return LOCAL_GROUPERS.create_grouper(["bus", "unit", "carrier"])(
        n, c, port, nice_names=nice_names
    )


def get_name_bus_and_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get the name, buses and nice carrier names for a component.
    """
    return LOCAL_GROUPERS.create_grouper(["name", "bus", "carrier"])(
        n, c, port, nice_names=nice_names
    )


def get_country_and_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get component country and carrier.
    """
    return LOCAL_GROUPERS.create_grouper(["country", "carrier"])(
        n, c, port, nice_names=nice_names
    )


def get_bus_and_carrier_and_bus_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get component's carrier, bus and bus carrier in one combined list.

    Used for MultiIndex in energy balance.
    """
    return LOCAL_GROUPERS.create_grouper(["bus", "carrier", "bus_carrier"])(
        n, c, port, nice_names=nice_names
    )


def get_carrier_and_bus_carrier(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """
    Get component carrier and bus carrier in one combined list.
    """
    return LOCAL_GROUPERS.create_grouper(["carrier", "bus_carrier"])(
        n, c, port, nice_names=nice_names
    )


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


def port_efficiency(
    n: Network, c_name: str, port: str = "", dynamic: bool = False
) -> pd.Series | pd.DataFrame:
    ones = pd.Series(1, index=n.static(c_name).index)
    if port == "":
        efficiency = ones
    elif port == "0":
        efficiency = -ones
    else:
        key = "efficiency" if port == "1" else f"efficiency{port}"
        if dynamic and key in n.static(c_name):
            efficiency = n.get_switchable_as_dense(c_name, key)
        else:
            efficiency = n.static(c_name).get(key, ones)
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


@deprecated("Use n.statistics._get_grouping instead.")
def get_grouping(
    n: Network,
    c: str,
    groupby: Callable | Sequence[str] | str | bool,
    port: str | None = None,
    nice_names: bool = False,
) -> dict:
    return n.statistics._get_grouping(n, c, groupby, port, nice_names)


@deprecated("Use n.statistics._aggregate_timeseries instead.")
def aggregate_timeseries(
    df: pd.DataFrame, weights: pd.Series, agg: str = "sum"
) -> pd.Series:
    return AbstractStatisticsAccessor._aggregate_timeseries(df, weights, agg)


@deprecated("Use n.statistics._filter_active_assets instead.")
def filter_active_assets(
    n: Network, c: str, df: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    return n.statistics._filter_active_assets(n, c, df)


@deprecated("Use n.statistics._filter_bus_carrier instead.")
def filter_bus_carrier(
    n: Network,
    c: str,
    port: str,
    bus_carrier: Sequence[str] | str | None,
    df: pd.DataFrame,
) -> pd.DataFrame:
    return n.statistics._filter_bus_carrier(n, c, port, bus_carrier, df)


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

    # Class-level dictionary for registered groupers
    _registered_groupers: dict[str, Callable] = {
        "carrier": get_carrier,
        "bus_carrier": get_bus_carrier,
        "name": get_name,
        "bus": get_bus,
        "country": get_country,
        "unit": get_unit,
    }

    def __init__(self) -> None:
        pass

    @classmethod
    def __repr__(cls) -> str:
        return f"Groupers with registered groupers: {list(cls._registered_groupers.keys())}"

    @classmethod
    def register_grouper(cls, name: str, func: Callable) -> None:
        """Register a new grouper function"""
        cls._registered_groupers[name] = func

    @classmethod
    def create_grouper(cls, keys: str | tuple[str] | list[str]) -> Callable:
        if scalar_passed := isinstance(keys, str):
            keys = (keys,)

        def group_by_keys(
            n: Network, c: str, port: str, nice_names: bool = False
        ) -> list:
            grouped_data = []
            for key in keys:
                if key not in cls._registered_groupers:
                    grouped_data.append(n.static(c)[key].rename(key))
                    continue

                method = cls._registered_groupers[key]
                kwargs: dict[str, str | bool] = {}
                if "port" in signature(method).parameters:
                    kwargs["port"] = port
                if "nice_names" in signature(method).parameters:
                    kwargs["nice_names"] = nice_names
                grouped_data.append(method(n, c, **kwargs))

            return grouped_data[0] if scalar_passed else grouped_data

        return group_by_keys

    # Single groupers
    get_carrier = staticmethod(get_carrier)
    get_bus = staticmethod(get_bus)
    get_bus_carrier = staticmethod(get_bus_carrier)
    get_country = staticmethod(get_country)
    get_name = staticmethod(get_name)
    get_unit = staticmethod(get_unit)

    # Combined groupers
    get_bus_and_carrier = staticmethod(get_bus_and_carrier)
    get_bus_unit_and_carrier = staticmethod(get_bus_unit_and_carrier)
    get_name_bus_and_carrier = staticmethod(get_name_bus_and_carrier)
    get_country_and_carrier = staticmethod(get_country_and_carrier)
    get_bus_and_carrier_and_bus_carrier = staticmethod(
        get_bus_and_carrier_and_bus_carrier
    )
    get_carrier_and_bus_carrier = staticmethod(get_carrier_and_bus_carrier)


LOCAL_GROUPERS = Groupers()


class AbstractStatisticsAccessor(ABC):
    """
    Abstract accessor to calculate different statistical values.
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

    def _get_grouping(
        self,
        n: Network,
        c: str,
        groupby: Callable | Sequence[str] | str | bool,
        port: str | None = None,
        nice_names: bool = False,
    ) -> dict:
        by = None
        level = None
        if callable(groupby):
            if "port" in signature(groupby).parameters:
                by = groupby(n, c, port=port, nice_names=nice_names)
            else:
                by = groupby(n, c, nice_names=nice_names)
        elif isinstance(groupby, (str, list)):
            by = self.groupers.create_grouper(groupby)(
                n, c, port=port, nice_names=nice_names
            )
        elif groupby is not False:
            raise ValueError(
                f"Argument `groupby` must be a function, list, string, False or dict, got {type(groupby)}"
            )
        return dict(by=by, level=level)

    @property
    def is_multi_indexed(self) -> bool:
        return isinstance(self.n.snapshots, pd.MultiIndex)

    @classmethod
    def _aggregate_timeseries(
        cls, obj: Any, weights: pd.Series, agg: str | Callable | bool = "sum"
    ) -> Any:
        """
        Calculate the weighted sum or average of a DataFrame or Series.
        """
        if not agg:
            return obj.T if isinstance(obj, pd.DataFrame) else obj

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
        comps: Collection[str] | str | None = None,
        groupby: str | list[str] | Callable | None = None,
        aggregate_across_components: bool = False,
        at_port: Sequence[str] | str | bool | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        nice_names: bool | None = True,
    ) -> pd.Series | pd.DataFrame:
        """
        Apply a function and group the result for a collection of components.
        """
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

            ports = [str(col)[3:] for col in n.static(c) if str(col).startswith("bus")]
            if not at_port:
                ports = [ports[0]]

            values = []
            for port in ports:
                vals = func(n, c, port)
                if vals is None or not prod(vals.shape):
                    continue

                vals = self._filter_active_assets(n, c, vals)  # for multiinvest
                vals = self._filter_bus_carrier(n, c, port, bus_carrier, vals)

                if vals is None or not prod(vals.shape):
                    continue

                if groupby is not False:
                    grouping = self._get_grouping(
                        n, c, groupby, port=port, nice_names=nice_names
                    )
                    vals = self._aggregate_components_groupby(vals, grouping, agg)
                values.append(vals)

            if not values:
                continue

            df = self._aggregate_components_concat_values(values, agg)

            d[c] = df

        df = self._aggregate_components_concat_data(d, is_one_component)

        if aggregate_across_components:
            df = self._aggregate_across_components(df, agg)

        return df

    def _filter_active_assets(self, n: Network, c: str, obj: Any) -> Any:
        """
        For static values iterate over periods and concat values.
        """
        if isinstance(obj, pd.DataFrame) or "snapshot" in getattr(obj, "dims", []):
            return obj
        idx = self._get_component_index(obj, c)
        if not self.is_multi_indexed:
            mask = n.get_active_assets(c)
            idx = mask.index[mask].intersection(idx)
            return obj.loc[idx]

        per_period = {}
        for p in n.investment_periods:
            mask = n.get_active_assets(c, p)
            idx = mask.index[mask].intersection(idx)
            per_period[p] = obj.loc[idx]

        return self._concat_periods(per_period, c)

    def _filter_bus_carrier(
        self,
        n: Network,
        c: str,
        port: str,
        bus_carrier: Sequence[str] | str | None,
        obj: Any,
    ) -> Any:
        """
        Filter the DataFrame for components which are connected to a bus with
        carrier `bus_carrier`.
        """
        if bus_carrier is None:
            return obj

        idx = self._get_component_index(obj, c)
        ports = n.static(c).loc[idx, f"bus{port}"]
        port_carriers = ports.map(n.buses.carrier)
        if isinstance(bus_carrier, str):
            if bus_carrier in n.buses.carrier.unique():
                mask = port_carriers == bus_carrier
            else:
                mask = port_carriers.str.contains(bus_carrier)
        elif isinstance(bus_carrier, list):
            mask = port_carriers.isin(bus_carrier)
        else:
            raise ValueError(
                f"Argument `bus_carrier` must be a string or list, got {type(bus_carrier)}"
            )
        # links may have empty ports which results in NaNs
        mask = mask.where(mask.notnull(), False)
        return obj.loc[ports.index[mask]]


class StatisticsAccessor(AbstractStatisticsAccessor):
    """
    Accessor to calculate different statistical values.
    """

    def _get_component_index(self, df: pd.DataFrame | pd.Series, c: str) -> pd.Index:
        return df.index

    def _concat_periods(
        self, dfs: list[pd.DataFrame] | dict[str, pd.DataFrame], c: str
    ) -> pd.DataFrame:
        return pd.concat(dfs, axis=1)

    @staticmethod
    def _aggregate_with_weights(
        df: pd.DataFrame,
        weights: pd.Series,
        agg: str | Callable,
    ) -> pd.Series | pd.DataFrame:
        if agg == "sum":
            if isinstance(weights.index, pd.MultiIndex):
                return df.multiply(weights, axis=0).groupby(level=0).sum().T
            return weights @ df
        else:
            # Todo: here we leave out the weights, is that correct?
            return df.agg(agg)

    def _aggregate_components_groupby(
        self, vals: pd.DataFrame, grouping: dict, agg: Callable | str
    ) -> pd.DataFrame:
        return vals.groupby(**grouping).agg(agg)

    def _aggregate_components_concat_values(
        self, values: list[pd.DataFrame], agg: Callable | str
    ) -> pd.DataFrame:
        """
        Concatenate a list of DataFrames.
        """
        df = pd.concat(values, copy=False) if len(values) > 1 else values[0]
        if not df.index.is_unique:
            df = df.groupby(level=df.index.names).agg(agg)
        return df

    def _aggregate_components_concat_data(
        self, d: dict[str, pd.DataFrame], is_one_component: bool
    ) -> pd.DataFrame | pd.Series:
        if d == {}:
            idx = pd.MultiIndex.from_tuples([], names=["component", "name"])
            return pd.Series([], index=idx)
        first_key = next(iter(d))
        if is_one_component:
            return d[first_key]
        index_names = ["component"] + d[first_key].index.names
        df = pd.concat(d, names=index_names)
        if self.parameters.round:
            df = df.round(self.parameters.round)
        if self.parameters.drop_zero:
            df = df[df != 0]
        return df

    def _aggregate_across_components(
        self, df: pd.Series | pd.DataFrame, agg: Callable | str
    ) -> pd.Series | pd.DataFrame:
        index_wo_component = df.index.droplevel("component")
        return df.groupby(index_wo_component).agg(agg)

    def __call__(
        self,
        comps: Sequence[str] | str | None = None,
        aggregate_groups: Callable | str = "sum",
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        installed = self.installed_capacity(
            comps=comps,
            aggregate_groups=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            return self._aggregate_timeseries(opex, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            return self._aggregate_timeseries(p, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = ["carrier", "bus_carrier"],
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
                "Network has multiple bus carriers which are aggregated together. "
                "To separate bus carriers set `bus_carrier` or use `bus_carrier` in the groupby argument."
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
            return self._aggregate_timeseries(p, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            return self._aggregate_timeseries(p, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        at_port: Sequence[str] | str | bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            return self._aggregate_timeseries(p, weights, agg=aggregate_time)

        kwargs = dict(
            comps=comps,
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            return self._aggregate_timeseries(revenue, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
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
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | list[str] | Callable | None = None,
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
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        df = self.revenue(**kwargs) / self.supply(**kwargs)  # type: ignore
        df.attrs["name"] = "Market Value"
        df.attrs["unit"] = "currency / MWh"
        return df
