"""Descriptors for component attributes."""

from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from itertools import product
from typing import TYPE_CHECKING, Any

import networkx as nx
import pandas as pd
from deprecation import deprecated

from pypsa.common import deprecated_common_kwargs, deprecated_in_next_major
from pypsa.constants import RE_PORTS_GE_2

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from pypsa import Network, SubNetwork
    from pypsa.type_utils import NetworkType

logger = logging.getLogger(__name__)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `pypsa.graph.network.OrderedGraph` instead.",
)
class OrderedGraph(nx.MultiGraph):
    """Ordered graph."""

    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


@deprecated_in_next_major(details="Use `n.get_switchable_as_dense` instead.")
@deprecated_common_kwargs
def get_switchable_as_dense(
    n: Network,
    component: str,
    attr: str,
    snapshots: Sequence | None = None,
    inds: pd.Index | None = None,
) -> pd.DataFrame:
    """Return a Dataframe for a time-varying component attribute .

    Deprecation
    ------------
    Use `n.get_switchable_as_dense` instead.

    """
    return n.get_switchable_as_dense(component, attr, snapshots, inds)


@deprecated_in_next_major(details="Use `n.get_switchable_as_iter` instead.")
@deprecated_common_kwargs
def get_switchable_as_iter(
    n: Network,
    component: str,
    attr: str,
    snapshots: Sequence,
    inds: pd.Index | None = None,
) -> pd.DataFrame:
    """Return an iterator over snapshots for a time-varying component attribute.

    Deprecation
    ------------
    Use `n.get_switchable_as_iter` instead.

    """
    return n.get_switchable_as_iter(component, attr, snapshots, inds)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `pypsa.pf.allocate_series_dataframes` instead.",
)
@deprecated_common_kwargs
def allocate_series_dataframes(n: Network, series: dict) -> None:
    """Populate time-varying outputs with default values."""
    from pypsa.pf import allocate_series_dataframes as allocate_series_dataframes_pf

    allocate_series_dataframes_pf(n, series)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Will be removed in the next major release.",
)
@deprecated_common_kwargs
def free_output_series_dataframes(
    n: Network, components: Collection[str] | None = None
) -> None:
    """Free output series dataframes.

    Parameters
    ----------
    n : Network
        Network instance.
    components : Collection[str] | None
        Components to free. If None, all components are freed.

    """
    if components is None:
        components = n.all_components

    for component in components:
        attrs = n.components[component]["attrs"]
        dynamic = n.dynamic(component)

        for attr in attrs.index[attrs["varying"] & (attrs["status"] == "Output")]:
            dynamic[attr] = pd.DataFrame(index=n.snapshots, columns=[])


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `pypsa.pf.zsum` instead.",
)
def zsum(s: pd.Series, *args: Any, **kwargs: Any) -> Any:
    """Sum values in a series, returning 0 for empty series.

    Pandas 0.21.0 changes sum() behavior so that the result of applying sum
    over an empty DataFrame is NaN.

    Meant to be set as pd.Series.zsum = zsum.
    """
    # TODO Remove
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


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `pypsa.common.expand_series` instead.",
)
def expand_series(ser: pd.Series, columns: Sequence[str]) -> pd.DataFrame:
    """Expand a series to a dataframe.

    Columns are the given series and every single column being the equal to
    the given series.
    """
    return ser.to_frame(columns[0]).reindex(columns=columns).ffill(axis=1)


@deprecated_in_next_major(details="Use `n.components[c].extendables` instead.")
def get_extendable_i(n: Network, c: str) -> pd.Index:
    """Get the index of extendable elements of a given component."""
    return n.components[c].extendables


@deprecated_in_next_major(details="Use `n.components[c].fixed` instead.")
def get_non_extendable_i(n: Network, c: str) -> pd.Index:
    """Getter function.

    Get the index of non-extendable elements of a given component.

    Deprecated: Use n.components[c].self.fixed instead.
    """
    return n.components[c].fixed


@deprecated_in_next_major(details="Use `n.components[c].committables` instead.")
def get_committable_i(n: Network, c: str) -> pd.Index:
    """Getter function.

    Get the index of commitable elements of a given component.

    Deprecated: Use n.components[c].get_committable_i() instead.
    """
    return n.components[c].committables


@deprecated_in_next_major(details="Use `n.components[c].get_active_assets` instead.")
def get_active_assets(
    n: Network | SubNetwork,
    c: str,
    investment_period: int | str | Sequence | None = None,
) -> pd.Series:
    """Get active assets. Use `c.get_active_assets`.

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


@deprecated_in_next_major(details="Use `n.components[c].get_activity_mask` instead.")
@deprecated_common_kwargs
def get_activity_mask(
    n: Network,
    c: str,
    sns: Sequence | None = None,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Get active components mask indexed by snapshots.

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
    return n.components[c].get_activity_mask(sns, index)


@deprecated_in_next_major(details="Deprecate with new-opt.")
@deprecated_common_kwargs
def get_bounds_pu(
    n: Network,
    c: str,
    sns: Sequence,
    index: pd.Index | None = None,
    attr: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve per unit bounds of a given component.

    Getter function to retrieve the per unit bounds of a given compoent for
    given snapshots and possible subset of elements (e.g. non-extendables).
    Depending on the attr you can further specify the bounds of the variable
    you are looking at, e.g. p_store for storage units.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
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
            from pypsa.common import expand_series

            max_pu = expand_series(n.static(c).max_hours, sns).T
            min_pu = pd.DataFrame(0, *max_pu.axes)
    else:
        min_pu = get_switchable_as_dense(n, c, min_pu_str, sns)

    if index is None:
        return min_pu, max_pu
    return min_pu.reindex(columns=index), max_pu.reindex(columns=index)


def _update_linkports_doc_changes(s: Any, i: int, j: str) -> Any:
    """Update components documentation for link ports.

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


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Will be removed in the next major release.",
)
def update_linkports_doc_changes(s: Any, i: int, j: str) -> Any:
    """Update components documentation for link ports.

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
    return _update_linkports_doc_changes(s, i, j)


def _update_linkports_component_attrs(
    n: NetworkType, where: Iterable[str] | None = None
) -> None:
    """Update the Link components attributes to add the additional ports.

    Parameters
    ----------
    n : Network
        Network instance to which additional ports will be added.
    where : Iterable[str] or None, optional
        Filters for specific subsets of data by providing an iterable of tags
        or identifiers. If None, no filtering is applied and additional link
        ports are considered for all connectors.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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
            .apply(_update_linkports_doc_changes, args=("1", i))
        )
        # Also update container for varying attributes
        if attr in ["efficiency", "p"] and target not in n.dynamic(c):
            df = pd.DataFrame(index=n.snapshots, columns=[], dtype=float)
            df.columns.name = c
            n.dynamic(c)[target] = df
        elif attr == "bus" and target not in n.static(c).columns:
            n.static(c)[target] = n.components[c]["attrs"].loc[target, "default"]


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Will be removed in the next major release.",
)
@deprecated_common_kwargs
def update_linkports_component_attrs(
    n: Network, where: Iterable[str] | None = None
) -> None:
    """Update the Link components attributes to add the additional ports.

    Parameters
    ----------
    n : Network
        Network instance to which additional ports will be added.
    where : Iterable[str] or None, optional
        Filters for specific subsets of data by providing an iterable of tags
        or identifiers. If None, no filtering is applied and additional link
        ports are considered for all connectors.

    """
    _update_linkports_component_attrs(n, where)


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.components.links.additional_ports` instead. Passing `where` will be deprecated.",
)
def additional_linkports(n: Network, where: Iterable[str] | None = None) -> list[str]:
    """Identify additional link ports (bus connections) beyond predefined ones.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    where : iterable of strings, default None
        Subset of columns to consider. Takes link columns by default.

    Returns
    -------
    list of strings
        List of additional link ports. E.g. ["2", "3"] for bus2, bus3.

    """
    if where is None:
        where = n.links.columns
    return [match.group(1) for col in where if (match := RE_PORTS_GE_2.search(col))]


@deprecated(
    deprecated_in="0.35",
    removed_in="1.0",
    details="Use `n.bus_carrier_unit` instead.",
)
def bus_carrier_unit(n: Network, bus_carrier: str | Sequence[str] | None) -> str:
    """Determine the unit associated with a specific bus carrier in the network."""
    return n.bus_carrier_unit(bus_carrier)
