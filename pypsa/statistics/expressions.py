"""Statistics Accessor."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pypsa.plot.statistics.plotter import StatisticPlotter

if TYPE_CHECKING:
    from pypsa import Network

import pandas as pd

from pypsa._options import options
from pypsa.common import (
    MethodHandlerWrapper,
    deprecated_kwargs,
    pass_empty_series_if_keyerror,
)
from pypsa.descriptors import bus_carrier_unit, nominal_attrs
from pypsa.statistics.abstract import AbstractStatisticsAccessor

if TYPE_CHECKING:
    from pypsa import Network


logger = logging.getLogger(__name__)


def get_operation(n: Network, c: str) -> pd.DataFrame:
    """Get the operation time series of a component."""
    if c in n.branch_components:
        return n.dynamic(c).p0
    elif c == "Store":
        return n.dynamic(c).e
    else:
        return n.dynamic(c).p


def get_weightings(n: Network, c: str) -> pd.Series:
    """Get the relevant snapshot weighting for a component."""
    if c == "Generator":
        return n.snapshot_weightings["generators"]
    elif c in ["StorageUnit", "Store"]:
        return n.snapshot_weightings["stores"]
    else:
        return n.snapshot_weightings["objective"]


def port_efficiency(
    n: Network, c_name: str, port: str = "", dynamic: bool = False
) -> pd.Series | pd.DataFrame:
    """Get the efficiency of a component at a specific port."""
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
    n: Network, bus_carrier: str | Sequence[str] | None = None
) -> pd.MultiIndex:
    """Get list of assets which transport between buses of the carrier `bus_carrier`."""
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
    n: Network, bus_carrier: str | Sequence[str] | None = None
) -> pd.MultiIndex:
    """Get the carriers which transport between buses of the carrier `bus_carrier`."""
    branches = get_transmission_branches(n, bus_carrier)
    carriers = {}
    for c in branches.unique(0):
        idx = branches[branches.get_loc(c)].get_level_values(1)
        carriers[c] = n.static(c).carrier[idx].unique()
    return pd.MultiIndex.from_tuples(
        [(c, i) for c, idx in carriers.items() for i in idx],
        names=["component", "carrier"],
    )


class StatisticHandler:
    """
    Statistic method handler.

    This class wraps a statistic method and provides a callable instance. To the get
    the statistic output as a DataFrame, call the instance with the desired arguments.

    See Also
    --------
    pypsa.common.MethodHandlerWrapper

    """

    def __init__(self, bound_method: Callable, n: Network) -> None:
        """
        Initialize the statistic handler.

        Parameters
        ----------
        bound_method : Callable
            The bound method/ underlying statistic function to call.
        n : Network
            The network object to use for the statistic calculation.

        """
        self._bound_method = bound_method
        self._n = n
        self.plot = StatisticPlotter(n=n, bound_method=bound_method)

    def __call__(self, *args: Any, **kwargs: Any) -> pd.DataFrame:  # noqa: D102
        return self._bound_method(*args, **kwargs)

    def __repr__(self) -> str:
        """
        Return the string representation of the statistic handler.

        Returns
        -------
        str
            String representation of the statistic handler

        Examples
        --------
        >>> handler = StatisticHandler(lambda x: x,n=n)
        >>> handler
        StatisticHandler(<lambda>)

        """
        return f"StatisticHandler({self._bound_method.__name__})"


class StatisticsAccessor(AbstractStatisticsAccessor):
    """Accessor to calculate different statistical values."""

    _methods = [
        "capex",
        "installed_capex",
        "expanded_capex",
        "optimal_capacity",
        "installed_capacity",
        "expanded_capacity",
        "opex",
        "supply",
        "withdrawal",
        "transmission",
        "energy_balance",
        "curtailment",
        "capacity_factor",
        "revenue",
        "market_value",
    ]

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
        """Concatenate a list of DataFrames."""
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
        return df

    def _apply_option_kwargs(
        self,
        df: pd.DataFrame,
        nice_names: bool | None,
        drop_zero: bool | None,
        round: int | None,
    ) -> pd.DataFrame:
        # nice_names_ = (
        #     options.params.statistics.nice_names if nice_names is None else nice_names
        # )
        # TODO move nice names here and drop from groupers
        round_ = options.params.statistics.round if round is None else round
        drop_zero_ = (
            options.params.statistics.drop_zero if drop_zero is None else drop_zero
        )
        if round_ is not None:
            df = df.round(round_)
        if drop_zero_ is not None:
            df = df[df != 0]

        return df

    def _aggregate_across_components(
        self, df: pd.Series | pd.DataFrame, agg: Callable | str
    ) -> pd.Series | pd.DataFrame:
        levels = [l for l in df.index.names if l != "component"]
        return df.groupby(level=levels).agg(agg)

    def _aggregate_components_skip_iteration(
        self, vals: pd.Series | pd.DataFrame
    ) -> bool:
        return vals.empty

    def __call__(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: bool | str | Sequence[str] = False,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
        aggregate_time: None = None,
    ) -> pd.DataFrame:
        """
        Calculate statistical values for a network.

        This function calls multiple function in the background in order to
        derive a full table of relevant network information. It groups the
        values to components according to the groupby argument.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.
        aggregate_time : None
            Deprecated. Use dedicated functions for individual statistics instead.

        Returns
        -------
        df :
            pandas.DataFrame with columns given the different quantities.

        """
        if aggregate_time is not None:
            warnings.warn(
                "The parameter `aggregate_time` is deprecated for the summary function."
                "Please use it for individual statistics instead.",
                DeprecationWarning,
            )
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
                aggregate_across_components=aggregate_across_components,
                groupby=groupby,
                at_port=at_port,
                carrier=carrier,
                bus_carrier=bus_carrier,
                nice_names=nice_names,
                drop_zero=drop_zero,
                round=round,
            )
            res[df.attrs["name"]] = df
        index = pd.Index(set.union(*[set(df.index) for df in res.values()]))
        res = {k: v.reindex(index, fill_value=0.0) for k, v in res.items()}
        return pd.concat(res, axis=1).sort_index(axis=0)

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def capex(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: bool | str | Sequence[str] = False,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
        cost_attribute: str = "capital_cost",
    ) -> pd.DataFrame:
        """
        Calculate the capital expenditure.

        Includes newly installed and existing assets, measured in the specified
        currency.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        cost_attribute : str
            Network attribute that should be used to calculate Capital Expenditure.
            Defaults to `capital_cost`.

        Returns
        -------
        pd.DataFrame
            Capital expenditure with components as rows and a single column of
            aggregated values.

        Example
        -------
        >>> n.statistics.capex()
        Series([], dtype: float64)

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
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        df.attrs["name"] = "Capital Expenditure"
        df.attrs["unit"] = "currency"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def installed_capex(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: bool | str | Sequence[str] = False,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
        cost_attribute: str = "capital_cost",
    ) -> pd.DataFrame:
        """
        Calculate the capital expenditure of already built capacities.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        cost_attribute : str
            Network attribute that should be used to calculate Capital Expenditure.
            Defaults to `capital_cost`.

        Returns
        -------
        pd.DataFrame
            Capital expenditure of already built capacities with components as rows and
            a single column of aggregated values.

        Example
        -------


        >>> n.statistics.installed_capex().sort_index()
        component  carrier
        Generator  gas        2.120994e+07
                   wind       6.761698e+05
        Line       AC         1.653634e+04
        Link       DC         1.476534e+03
        dtype: float64

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
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        df.attrs["name"] = "Capital Expenditure Fixed"
        df.attrs["unit"] = "currency"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def expanded_capex(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: bool | str | Sequence[str] = False,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
        cost_attribute: str = "capital_cost",
    ) -> pd.DataFrame:
        """
        Calculate the capital expenditure of expanded capacities.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        cost_attribute : str
            Network attribute that should be used to calculate Capital Expenditure.
            Defaults to `capital_cost`.

        Returns
        -------
        pd.DataFrame
            Capital expenditure of expanded capacities with components as rows and
            a single column of aggregated values.

        Example
        -------
        >>> n.statistics.expanded_capex().sort_index() # doctest: +ELLIPSIS
        component  carrier
        Generator  gas       -2.120994e+07
                   wind      -6.761698e+05
        ...

        """
        df = self.capex(
            comps=comps,
            aggregate_groups=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
            cost_attribute=cost_attribute,
        ).sub(
            self.installed_capex(
                comps=comps,
                aggregate_groups=aggregate_groups,
                aggregate_across_components=aggregate_across_components,
                groupby=groupby,
                at_port=at_port,
                carrier=carrier,
                bus_carrier=bus_carrier,
                nice_names=nice_names,
                drop_zero=drop_zero,
                round=round,
                cost_attribute=cost_attribute,
            ),
            fill_value=0,
        )
        df.attrs["name"] = "Capital Expenditure Expanded"
        df.attrs["unit"] = "currency"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def optimal_capacity(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: str | Sequence[str] | bool | None = None,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
        storage: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate the optimal capacity of the network components in MW.

        Positive capacity values correspond to production capacities and
        negative values to consumption capacities.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        storage : bool, default=False
            Whether to consider only storage capacities of the components
            `Store` and `StorageUnit`.

        Returns
        -------
        pd.DataFrame
            Optimal capacity of the network components with components as rows and
            a single column of aggregated values.

        Example
        -------

        >>> n.statistics.optimal_capacity()
        Series([], dtype: float64)

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
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        df.attrs["name"] = "Optimal Capacity"
        df.attrs["unit"] = "MW"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def installed_capacity(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: str | Sequence[str] | bool | None = None,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
        storage: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate the installed capacity of the network components in MW.

        Positive capacity values correspond to production capacities and
        negative values to consumption capacities.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        storage : bool, default=False
            Whether to consider only storage capacities of the components
            `Store` and `StorageUnit`.

        Returns
        -------
            pd.DataFrame
                Installed capacity of the network components with components as rows and
                a single column of aggregated values.

        Example
        -------

        >>> n.statistics.installed_capacity().sort_index()
        component  carrier
        Generator  gas        150000.0
                   wind          290.0
        Line       AC         160000.0
        Link       DC           4000.0
        dtype: float64

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
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        df.attrs["name"] = "Installed Capacity"
        df.attrs["unit"] = "MW"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def expanded_capacity(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: str | Sequence[str] | bool | None = None,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the expanded capacity of the network components in MW.

        Positive capacity values correspond to production capacities and
        negative values to consumption capacities.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Returns
        -------
        pd.DataFrame
            Expanded capacity of the network components with components as rows and
            a single column of aggregated values.

        Example
        -------

        >>> n.statistics.expanded_capacity()
        Series([], dtype: float64)

        """
        optimal = self.optimal_capacity(
            comps=comps,
            aggregate_groups=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        installed = self.installed_capacity(
            comps=comps,
            aggregate_groups=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        installed = installed.reindex(optimal.index, fill_value=0)
        df = optimal.sub(installed).where(optimal.abs() > installed.abs(), 0)
        df.attrs["name"] = "Expanded Capacity"
        df.attrs["unit"] = "MW"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def opex(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: bool | str | Sequence[str] = False,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the ongoing operational costs.

        Takes into account various operational parameters and their costs, measured in
        the specified currency.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        aggregate_time : str | bool, default="sum"
            Type of aggregation when aggregating time series. Deactivate by setting to
            False. Any pandas aggregation function can be used. Note that when
            aggregating the time series are aggregated to MWh using snapshot weightings.
            With False the time series is given in MW.

        Returns
        -------
        pd.DataFrame
            Ongoing operational costs with components as rows and
            either time steps as columns (if aggregate_time=False) or a single column
            of aggregated values.

        Example
        -------

        >>> n.statistics.opex()
        Series([], dtype: float64)

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
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        df.attrs["name"] = "Operational Expenditure"
        df.attrs["unit"] = "currency"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def supply(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: bool | str | Sequence[str] = True,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the supply of components in the network.

        Units depend on the regarded bus carrier.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        aggregate_time : str | bool, default="sum"
            Type of aggregation when aggregating time series. Deactivate by setting to
            False. Any pandas aggregation function can be used. Note that when
            aggregating the time series are aggregated to MWh using snapshot weightings.
            With False the time series is given in MW.

        Returns
        -------
        pd.DataFrame
            Supply of components in the network with components as rows and
            either time steps as columns (if aggregate_time=False) or a single column
            of aggregated values.

        Example
        -------
        >>> n.statistics.supply()
        Series([], dtype: float64)

        """
        df = self.energy_balance(
            comps=comps,
            aggregate_time=aggregate_time,
            aggregate_groups=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
            direction="supply",
        )
        df.attrs["name"] = "Supply"
        df.attrs["unit"] = "carrier dependent"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def withdrawal(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: bool | str | Sequence[str] = True,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the withdrawal of components in the network.

        Units depend on the regarded bus carrier.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        aggregate_time : str | bool, default="sum"
            Type of aggregation when aggregating time series. Deactivate by setting to
            False. Any pandas aggregation function can be used. Note that when
            aggregating the time series are aggregated to MWh using snapshot weightings.
            With False the time series is given in MW.

        Returns
        -------
        pd.DataFrame
            Withdrawal of components in the network with components as rows and
            either time steps as columns (if aggregate_time=False) or a single column
            of aggregated values.

        Example
        -------
        >>> n.statistics.withdrawal()
        Series([], dtype: float64)

        """
        df = self.energy_balance(
            comps=comps,
            aggregate_time=aggregate_time,
            aggregate_groups=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
            direction="withdrawal",
        )
        df.attrs["name"] = "Withdrawal"
        df.attrs["unit"] = "carrier dependent"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def transmission(
        self,
        comps: Collection[str] | str | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable | Literal[False] = "carrier",
        at_port: bool | str | Sequence[str] = False,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the transmission of branch components in the network.

        Units depend on the regarded bus carrier.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        aggregate_time : str | bool, default="sum"
            Type of aggregation when aggregating time series. Deactivate by setting to
            False. Any pandas aggregation function can be used. Note that when
            aggregating the time series are aggregated to MWh using snapshot weightings.
            With False the time series is given in MW.

        Returns
        -------
        pd.DataFrame
            Transmission of branch components in the network with components as rows and
            either time steps as columns (if aggregate_time=False) or a single column
            of aggregated values.

        Example
        -------
        >>> n.statistics.transmission()
        Series([], dtype: object)

        """
        n = self._n

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
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        df.attrs["name"] = "Transmission"
        df.attrs["unit"] = "carrier dependent"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    @deprecated_kwargs(kind="direction")
    def energy_balance(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = ["carrier", "bus_carrier"],
        at_port: bool | str | Sequence[str] = True,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
        direction: str | None = None,
    ) -> pd.DataFrame:
        """
        Calculate energy balance of components in network.

        This method computes the energy balance across various network components, where
        positive values represent supply and negative values represent withdrawal. Units
        are inherited from the respective bus carriers.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        aggregate_time : str | bool, default="sum"
            Type of aggregation when aggregating time series. Deactivate by setting to
            False. Any pandas aggregation function can be used. Note that when
            aggregating the time series are aggregated to MWh using snapshot weightings.
            With False the time series is given in MW.
        direction : str | None, default=None
            Type of energy balance to calculate:
            - 'supply': Only consider positive values (energy production)
            - 'withdrawal': Only consider negative values (energy consumption)
            - None: Consider both supply and withdrawal

        Returns
        -------
        pd.DataFrame
            Energy balance with components as rows and either time steps as columns
            (if aggregate_time=False) or a single column of aggregated values.
            Units depend on the bus carrier and aggregation method.

        Example
        -------
        >>> n.statistics.energy_balance()
        Series([], dtype: float64)

        """
        n = self._n

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
            if direction == "supply":
                p = p.clip(lower=0)
            elif direction == "withdrawal":
                p = -p.clip(upper=0)
            elif direction is not None:
                logger.warning(
                    "Argument 'direction' is not recognized. Falling back to energy balance."
                )
            return self._aggregate_timeseries(p, weights, agg=aggregate_time)

        df = self._aggregate_components(
            func,
            comps=comps,
            agg=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )

        df.attrs["name"] = "Energy Balance"
        df.attrs["unit"] = bus_carrier_unit(n, bus_carrier)
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def curtailment(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: bool | str | Sequence[str] = False,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the curtailment of components in the network in MWh.

        The calculation only considers assets with a `p_max_pu` time
        series, which is used to quantify the available power potential.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        aggregate_time : str | bool, default="sum"
            Type of aggregation when aggregating time series. Deactivate by setting to
            False. Any pandas aggregation function can be used. Note that when
            aggregating the time series are aggregated to MWh using snapshot weightings.
            With False the time series is given in MW.

        Returns
        -------
        pd.DataFrame
            Curtailment of components in the network with components as rows and
            either time steps as columns (if aggregate_time=False) or a single column
            of aggregated values.

        Example
        -------
        >>> n.statistics.curtailment()
        Series([], Name: generators, dtype: float64)

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
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        df.attrs["name"] = "Curtailment"
        df.attrs["unit"] = "MWh"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def capacity_factor(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_time: str | bool = "mean",
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        at_port: bool | str | Sequence[str] = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the capacity factor of components in the network.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        aggregate_time : str | bool, default="sum"
            Type of aggregation when aggregating time series. Deactivate by setting to
            False. Any pandas aggregation function can be used. Note that when
            aggregating the time series are aggregated to MWh using snapshot weightings.
            With False the time series is given in MW.

        Returns
        -------
        pd.DataFrame
            Capacity factor of components in the network with components as rows and
            either time steps as columns (if aggregate_time=False) or a single column
            of aggregated values.

        Example
        -------
        >>> n.statistics.capacity_factor()
        Series([], dtype: float64)

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
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        df = self._aggregate_components(func, agg=aggregate_groups, **kwargs)  # type: ignore
        capacity = self.optimal_capacity(aggregate_groups=aggregate_groups, **kwargs)
        df = df.div(capacity.reindex(df.index), axis=0)
        df.attrs["name"] = "Capacity Factor"
        df.attrs["unit"] = "p.u."
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    @deprecated_kwargs(kind="direction")
    def revenue(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_time: str | bool = "sum",
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: bool | str | Sequence[str] = True,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
        direction: str | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the revenue of components in the network in given currency.

        The revenue is defined as the net revenue of an asset, i.e cost
        of input - revenue of output.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        aggregate_time : str | bool, default="sum"
            Type of aggregation when aggregating time series. Deactivate by setting to
            False. Any pandas aggregation function can be used. Note that when
            aggregating the time series are aggregated to MWh using snapshot weightings.
            With False the time series is given in MW.
        direction : str, optional, default=None
            Type of revenue to consider. If 'input' only the revenue of the input is considered.
            If 'output' only the revenue of the output is considered. Defaults to None.

        Returns
        -------
        pd.DataFrame
            Revenue of components in the network with components as rows and
            either time steps as columns (if aggregate_time=False) or a single column
            of aggregated values.

        Example
        -------
        >>> n.statistics.revenue()
        Series([], dtype: float64)

        """

        @pass_empty_series_if_keyerror
        def func(n: Network, c: str, port: str) -> pd.Series:
            sign = -1.0 if c in n.branch_components else n.static(c).get("sign", 1.0)
            df = sign * n.dynamic(c)[f"p{port}"]
            buses = n.static(c)[f"bus{port}"][df.columns]
            prices = n.buses_t.marginal_price.reindex(
                columns=buses, fill_value=0
            ).values
            if direction is not None:
                if direction == "input":
                    df = df.clip(upper=0)
                elif direction == "output":
                    df = df.clip(lower=0)
                else:
                    raise ValueError(
                        f"Argument 'direction' must be 'input', 'output' or None, got {direction}"
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
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        df.attrs["name"] = "Revenue"
        df.attrs["unit"] = "currency"
        return df

    @MethodHandlerWrapper(handler_class=StatisticHandler, inject_attrs={"n": "_n"})
    def market_value(
        self,
        comps: str | Sequence[str] | None = None,
        aggregate_time: str | bool = "mean",
        aggregate_groups: Callable | str = "sum",
        aggregate_across_components: bool = False,
        groupby: str | Sequence[str] | Callable = "carrier",
        at_port: bool | str | Sequence[str] = True,
        carrier: str | Sequence[str] | None = None,
        bus_carrier: str | Sequence[str] | None = None,
        nice_names: bool | None = None,
        drop_zero: bool | None = None,
        round: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the market value of components in the network.

        Curreny is currency/MWh or currency/unit_{bus_carrier} where unit_{bus_carrier}
        is the unit of the bus carrier.

        Parameters
        ----------
        comps : str | Sequence[str] | None, default=None
            Components to include in the calculation. If None, includes all one-port
            and branch components. Available components are 'Generator', 'StorageUnit',
            'Store', 'Load', 'Line', 'Transformer' and'Link'.
        aggregate_groups : Callable | str, default="sum"
            Function to aggregate groups when using the groupby parameter.
            Any pandas aggregation function can be used.
        aggregate_across_components : bool, default=False
            Whether to aggregate across components. If there are different components
            which would be grouped together due to the same index, this is avoided.
        groupby : str | Sequence[str] | Callable, default=["carrier", "bus_carrier"]
            How to group components:
            - str or list of str: Column names from component static DataFrames
            - callable: Function that takes network and component name as arguments
        at_port : bool | str | Sequence[str], default=True
            Which ports to consider:
            - True: All ports of components
            - False: Exclude first port ("bus"/"bus0")
            - str or list of str: Specific ports to include
        carrier : str | Sequence[str] | None, default=None
            Filter by carrier. If specified, only considers assets with given
            carrier(s).
        bus_carrier : str | Sequence[str] | None, default=None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s).
        nice_names : bool | None, default=None
            Whether to use carrier nice names defined in n.carriers.nice_name. Defaults
            to module wide option (default: True).
            See `pypsa.options.params.statistics.describe()` for more information.
        drop_zero : bool | None, default=None
            Whether to drop zero values from the result. Defaults to module wide option
            (default: True). See `pypsa.options.params.statistics.describe()` for more
            information.
        round : int | None, default=None
            Number of decimal places to round the result to. Defaults to module wide
            option (default: 2). See `pypsa.options.params.statistics.describe()` for
            more information.

        Additional Parameters
        ---------------------
        aggregate_time : str | bool, default="sum"
            Type of aggregation when aggregating time series. Deactivate by setting to
            False. Any pandas aggregation function can be used. Note that when
            aggregating the time series are aggregated to MWh using snapshot weightings.
            With False the time series is given in MW.

        Returns
        -------
        pd.DataFrame
            Market value of components in the network with components as rows and
            either time steps as columns (if aggregate_time=False) or a single column
            of aggregated values.

        Example
        -------
        >>> n.statistics.market_value()
        Series([], dtype: float64)

        """
        kwargs = dict(
            comps=comps,
            aggregate_time=aggregate_time,
            aggregate_groups=aggregate_groups,
            aggregate_across_components=aggregate_across_components,
            groupby=groupby,
            at_port=at_port,
            carrier=carrier,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            drop_zero=drop_zero,
            round=round,
        )
        df = self.revenue(**kwargs) / self.supply(**kwargs)
        df.attrs["name"] = "Market Value"
        df.attrs["unit"] = "currency / MWh"
        return df
