"""
Descriptors for component attributes.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Collection, Iterable, Sequence
from itertools import product, repeat
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd

from pypsa.utils import as_index, deprecated_common_kwargs

if TYPE_CHECKING:
    from pypsa.components import Network, SubNetwork

logger = logging.getLogger(__name__)


class OrderedGraph(nx.MultiGraph):
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


@deprecated_common_kwargs
def get_switchable_as_dense(
    n: Network,
    component: str,
    attr: str,
    snapshots: Sequence | None = None,
    inds: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Return a Dataframe for a time-varying component attribute with values for
    all non-time-varying components filled in with the default values for the
    attribute.

    Parameters
    ----------
    n : pypsa.Network
    component : string
        Component object name, e.g. 'Generator' or 'Link'
    attr : string
        Attribute name
    snapshots : pandas.Index
        Restrict to these snapshots rather than n.snapshots.
    inds : pandas.Index
        Restrict to these components rather than n.components.index

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> get_switchable_as_dense(n, 'Generator', 'p_max_pu')
    """
    static = n.static(component)
    dynamic = n.dynamic(component)

    index = static.index

    varying_i = dynamic[attr].columns
    fixed_i = static.index.difference(varying_i)

    if inds is not None:
        index = index.intersection(inds)
        varying_i = varying_i.intersection(inds)
        fixed_i = fixed_i.intersection(inds)
    if snapshots is None:
        snapshots = n.snapshots

    vals = np.repeat([static.loc[fixed_i, attr].values], len(snapshots), axis=0)
    static = pd.DataFrame(vals, index=snapshots, columns=fixed_i)
    varying = dynamic[attr].loc[snapshots, varying_i]

    res = pd.merge(static, varying, left_index=True, right_index=True, how="inner")
    del static
    del varying
    res = res.reindex(columns=index)
    res.index.name = "snapshot"  # reindex with multiindex does not preserve name

    return res


@deprecated_common_kwargs
def get_switchable_as_iter(
    n: Network,
    component: str,
    attr: str,
    snapshots: Sequence,
    inds: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Return an iterator over snapshots for a time-varying component attribute
    with values for all non-time-varying components filled in with the default
    values for the attribute.

    Parameters
    ----------
    n : pypsa.Network
    component : string
        Component object name, e.g. 'Generator' or 'Link'
    attr : string
        Attribute name
    snapshots : pandas.Index
        Restrict to these snapshots rather than n.snapshots.
    inds : pandas.Index
        Restrict to these items rather than all of n.{generators, ..}.index

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> get_switchable_as_iter(n, 'Generator', 'p_max_pu', snapshots)
    """
    static = n.static(component)
    dynamic = n.dynamic(component)

    index = static.index
    varying_i = dynamic[attr].columns
    fixed_i = static.index.difference(varying_i)

    if inds is not None:
        inds = pd.Index(inds)
        index = inds.intersection(index)
        varying_i = inds.intersection(varying_i)
        fixed_i = inds.intersection(fixed_i)

    # Short-circuit only fixed
    if len(varying_i) == 0:
        return repeat(static.loc[fixed_i, attr], len(snapshots))

    def is_same_indices(i1: pd.Index, i2: pd.Index) -> bool:
        return len(i1) == len(i2) and (i1 == i2).all()

    if is_same_indices(fixed_i.append(varying_i), index):

        def reindex_maybe(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
            return s

    else:

        def reindex_maybe(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
            return s.reindex(index)

    return (
        reindex_maybe(
            static.loc[fixed_i, attr].append(dynamic[attr].loc[sn, varying_i])
        )
        for sn in snapshots
    )


@deprecated_common_kwargs
def allocate_series_dataframes(n: Network, series: dict) -> None:
    """
    Populate time-varying outputs with default values.

    Parameters
    ----------
    n : pypsa.Network
    series : dict
        Dictionary of components and their attributes to populate (see example)

    Returns
    -------
    None

    Examples
    --------
    >>> allocate_series_dataframes(n, {'Generator': ['p'],
                                             'Load': ['p']})
    """
    for component, attributes in series.items():
        static = n.static(component)
        dynamic = n.dynamic(component)

        for attr in attributes:
            dynamic[attr] = dynamic[attr].reindex(
                columns=static.index,
                fill_value=n.components[component]["attrs"].at[attr, "default"],
            )


@deprecated_common_kwargs
def free_output_series_dataframes(
    n: Network, components: Collection[str] | None = None
) -> None:
    if components is None:
        components = n.all_components

    for component in components:
        attrs = n.components[component]["attrs"]
        dynamic = n.dynamic(component)

        for attr in attrs.index[attrs["varying"] & (attrs["status"] == "Output")]:
            dynamic[attr] = pd.DataFrame(index=n.snapshots, columns=[])


def zsum(s: pd.Series, *args: Any, **kwargs: Any) -> Any:
    """
    Pandas 0.21.0 changes sum() behavior so that the result of applying sum
    over an empty DataFrame is NaN.

    Meant to be set as pd.Series.zsum = zsum.
    """
    return 0 if s.empty else s.sum(*args, **kwargs)


# Perhaps this should rather go into components.py
nominal_attrs = {
    "Generator": "p_nom",
    "Line": "s_nom",
    "Transformer": "s_nom",
    "Link": "p_nom",
    "Store": "e_nom",
    "StorageUnit": "p_nom",
}


def expand_series(ser: pd.Series, columns: Sequence[str]) -> pd.DataFrame:
    """
    Helper function to quickly expand a series to a dataframe with according
    column axis and every single column being the equal to the given series.
    """
    return ser.to_frame(columns[0]).reindex(columns=columns).ffill(axis=1)


def get_extendable_i(n: Network, c: str) -> pd.Index:
    """
    Getter function.

    Get the index of extendable elements of a given component.
    """
    idx = n.static(c)[lambda ds: ds[nominal_attrs[c] + "_extendable"]].index
    return idx.rename(f"{c}-ext")


def get_non_extendable_i(n: Network, c: str) -> pd.Index:
    """
    Getter function.

    Get the index of non-extendable elements of a given component.
    """
    idx = n.static(c)[lambda ds: ~ds[nominal_attrs[c] + "_extendable"]].index
    return idx.rename(f"{c}-fix")


def get_committable_i(n: Network, c: str) -> pd.Index:
    """
    Getter function.

    Get the index of commitable elements of a given component.
    """
    if "committable" not in n.static(c):
        idx = pd.Index([])
    else:
        idx = n.static(c)[lambda ds: ds["committable"]].index
    return idx.rename(f"{c}-com")


def get_active_assets(
    n: Network | SubNetwork,
    c: str,
    investment_period: int | str | Sequence | None = None,
) -> pd.Series:
    """
    Get active components mask of component type in investment period(s).

    See the :py:meth:`pypsa.descriptors.components.Component.get_active_assets`.

    Parameters
    ----------
    n : pypsa.Network
        Network instance
    c : string
        Component name
    investment_period : int, str, Sequence
        Investment period(s) to check

    Returns
    -------
    pd.Series
        Boolean mask for active components
    """
    return n.component(c).get_active_assets(investment_period=investment_period)


@deprecated_common_kwargs
def get_activity_mask(
    n: Network,
    c: str,
    sns: Sequence | None = None,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Get active components mask indexed by snapshots.

    Wrapper around the
    `:py:meth:`pypsa.descriptors.components.Componenet.get_active_assets` method.
    Get's the boolean mask for active components, but indexed by snapshots and
    components instead of just components.

    Parameters
    ----------
    n : pypsa.Network
        Network instance
    c : string
        Component name
    sns : pandas.Index, default None
        Set of snapshots for the mask. If None (default) all snapshots are returned.
    index : pd.Index, default None
        Subset of the component elements. If None (default) all components are returned.
    """

    sns_ = as_index(n, sns, "snapshots", "snapshot")

    if getattr(n, "_multi_invest", False):
        active_assets_per_period = {
            period: get_active_assets(n, c, period) for period in n.investment_periods
        }
        mask = (
            pd.concat(active_assets_per_period, axis=1)
            .T.reindex(n.snapshots, level=0)
            .loc[sns_]
        )
    else:
        active_assets = get_active_assets(n, c)
        mask = pd.DataFrame(
            np.tile(active_assets, (len(sns_), 1)),
            index=sns_,
            columns=active_assets.index,
        )

    if index is not None:
        mask = mask.reindex(columns=index)

    return mask


@deprecated_common_kwargs
def get_bounds_pu(
    n: Network,
    c: str,
    sns: Sequence,
    index: pd.Index | None = None,
    attr: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Getter function to retrieve the per unit bounds of a given compoent for
    given snapshots and possible subset of elements (e.g. non-extendables).
    Depending on the attr you can further specify the bounds of the variable
    you are looking at, e.g. p_store for storage units.

    Parameters
    ----------
    n : pypsa.Network
    c : string
        Component name, e.g. "Generator", "Line".
    sns : pandas.Index/pandas.DateTimeIndex
        set of snapshots for the bounds
    index : pd.Index, default None
        Subset of the component elements. If None (default) bounds of all
        elements are returned.
    attr : string, default None
        attribute name for the bounds, e.g. "p", "s", "p_store"
    """
    min_pu_str = nominal_attrs[c].replace("nom", "min_pu")
    max_pu_str = nominal_attrs[c].replace("nom", "max_pu")

    max_pu = get_switchable_as_dense(n, c, max_pu_str, sns)
    if c in n.passive_branch_components:
        min_pu = -max_pu
    elif c == "StorageUnit":
        min_pu = pd.DataFrame(0, max_pu.index, max_pu.columns)
        if attr == "p_store":
            max_pu = -get_switchable_as_dense(n, c, min_pu_str, sns)
        if attr == "state_of_charge":
            max_pu = expand_series(n.static(c).max_hours, sns).T
            min_pu = pd.DataFrame(0, *max_pu.axes)
    else:
        min_pu = get_switchable_as_dense(n, c, min_pu_str, sns)

    if index is None:
        return min_pu, max_pu
    else:
        return min_pu.reindex(columns=index), max_pu.reindex(columns=index)


def update_linkports_doc_changes(s: Any, i: int, j: str) -> Any:
    """
    Update components documentation for link ports.

    Multi-linkports require the following changes:
    1. Replaces every occurrence of the substring `j` with `i`.
    2. Make attribute required

    Parameters
    ----------
    s : An
        String to update.
    i : int
        Integer to replace `j` with.
    j : string
        Substring to replace.

    Returns
    -------
    Any : Updated string or original value if not a string.

    """

    if not isinstance(s, str) or len(s) == 1:
        return s
    return s.replace(j, str(i)).replace("required", "optional")


@deprecated_common_kwargs
def update_linkports_component_attrs(
    n: Network, where: Iterable[str] | None = None
) -> None:
    """
    Update the Link components attributes to add the additional ports.

    Parameters
    ----------
    n : Network
        Network instance to which additional ports will be added.
    where : Iterable[str] or None, optional

        Filters for specific subsets of data by providing an iterable of tags
        or identifiers. If None, no filtering is applied and additional link
        ports are considered for all connectors.
    """

    ports = additional_linkports(n, where)
    ports.sort(reverse=True)
    c = "Link"

    for i, attr in product(ports, ["bus", "efficiency", "p"]):
        target = f"{attr}{i}"
        if target in n.components[c]["attrs"].index:
            continue
        j = "1" if attr != "efficiency" else ""
        base_attr = attr + j
        base_attr_index = n.components[c]["attrs"].index.get_loc(base_attr)
        n.components[c]["attrs"].index.insert(base_attr_index + 1, target)
        n.components[c]["attrs"].loc[target] = (
            n.components[c]["attrs"]
            .loc[attr + j]
            .apply(update_linkports_doc_changes, args=("1", i))
        )
        # Also update container for varying attributes
        if attr in ["efficiency", "p"] and target not in n.dynamic(c).keys():
            df = pd.DataFrame(index=n.snapshots, columns=[], dtype=float)
            df.index.name = "snapshot"
            df.columns.name = c
            n.dynamic(c)[target] = df
        elif attr == "bus" and target not in n.static(c).columns:
            n.static(c)[target] = n.components[c]["attrs"].loc[target, "default"]


def additional_linkports(n: Network, where: Iterable[str] | None = None) -> list[str]:
    """
    Identify additional link ports (bus connections) beyond predefined ones.

    Parameters
    ----------
    n : pypsa.Network
    where : iterable of strings, default None
        Subset of columns to consider. Takes link columns by default.

    Returns
    -------
    list of strings
        List of additional link ports. E.g. ["2", "3"] for bus2, bus3.
    """
    if where is None:
        where = n.links.columns
    return [i[3:] for i in where if i.startswith("bus") and i not in ["bus0", "bus1"]]
