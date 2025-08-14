"""Descriptors for component attributes."""

from __future__ import annotations

import logging
import warnings
from itertools import product
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.common import deprecated_in_next_major
from pypsa.constants import RE_PORTS_GE_2

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pypsa import Network, SubNetwork
    from pypsa.type_utils import NetworkType

logger = logging.getLogger(__name__)


@deprecated_in_next_major(details="Use `n.get_switchable_as_dense` instead.")
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


# Perhaps this should rather go into components.py
nominal_attrs = {
    "Generator": "p_nom",
    "Line": "s_nom",
    "Transformer": "s_nom",
    "Link": "p_nom",
    "Store": "e_nom",
    "StorageUnit": "p_nom",
}


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
    return n.components[c].get_active_assets(investment_period=investment_period)


@deprecated_in_next_major(details="Use `n.components[c].get_activity_mask` instead.")
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


@deprecated_in_next_major(details="Use `n.components[c].get_bounds_pu` instead.")
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
    attr : string, default None
        attribute name for the bounds, e.g. "p", "s", "p_store"
    sns : pandas.Index/pandas.DateTimeIndex
        Deprecated.
    index : pd.Index, default None
        Deprecated.

    """
    min_bounds, max_bounds = n.components[c].get_bounds_pu(attr)
    sel_kwargs = {}
    if sns is not None:
        sel_kwargs["snapshot"] = sns
    if index is not None:
        sel_kwargs["name"] = index
    return (
        min_bounds.sel(**sel_kwargs).to_dataframe().unstack(level=0),
        max_bounds.sel(**sel_kwargs).to_dataframe().unstack(level=0),
    )


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


def _additional_linkports(
    n: NetworkType, where: Iterable[str] | None = None
) -> list[str]:
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
        where = n.c.links.static.columns
    return [match.group(1) for col in where if (match := RE_PORTS_GE_2.search(col))]


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
        ports = _additional_linkports(n, where)
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
            df = pd.DataFrame(
                index=n.snapshots, columns=n.c.links.static.index[:0], dtype=float
            )
            n.dynamic(c)[target] = df
        elif attr == "bus" and target not in n.static(c).columns:
            n.static(c)[target] = n.components[c]["attrs"].loc[target, "default"]
