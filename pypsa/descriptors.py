# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Descriptors for component attributes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deprecation import deprecated

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from pypsa import Network, SubNetwork

logger = logging.getLogger(__name__)


@deprecated(
    deprecated_in="1.0.0",
    removed_in="2.0.0",
    details="Use `n.get_switchable_as_dense` instead.",
)
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
    return n.c[component]._as_dynamic(attr, snapshots, inds)


@deprecated(
    deprecated_in="1.0.0",
    removed_in="2.0.0",
    details="Use `n.get_switchable_as_iter` instead.",
)
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
    "generators": "p_nom",
    "lines": "s_nom",
    "transformers": "s_nom",
    "links": "p_nom",
    "processes": "p_nom",
    "stores": "e_nom",
    "storage_units": "p_nom",
}


@deprecated(
    deprecated_in="1.0.0",
    removed_in="2.0.0",
    details="Use `n.components[c].extendables` instead.",
)
def get_extendable_i(n: Network, c: str) -> pd.Index:
    """Get the index of extendable elements of a given component."""
    return n.components[c].extendables


@deprecated(
    deprecated_in="1.0.0",
    removed_in="2.0.0",
    details="Use `n.components[c].fixed` instead.",
)
def get_non_extendable_i(n: Network, c: str) -> pd.Index:
    """Getter function.

    Get the index of non-extendable elements of a given component.

    Deprecated: Use n.components[c].self.fixed instead.
    """
    return n.components[c].fixed


@deprecated(
    deprecated_in="1.0.0",
    removed_in="2.0.0",
    details="Use `n.components[c].committables` instead.",
)
def get_committable_i(n: Network, c: str) -> pd.Index:
    """Getter function.

    Get the index of commitable elements of a given component.

    Deprecated: Use n.components[c].get_committable_i() instead.
    """
    return n.components[c].committables


@deprecated(
    deprecated_in="1.0.0",
    removed_in="2.0.0",
    details="Use `n.components[c].get_active_assets` instead.",
)
def get_active_assets(
    n: Network | SubNetwork,
    c: str,
    investment_period: int | str | Sequence | None = None,
) -> pd.Series:
    """Get active assets. Use `c.get_active_assets`.

    See the [`pypsa.descriptors.components.Component.get_active_assets`][].

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


@deprecated(
    deprecated_in="1.0.0",
    removed_in="2.0.0",
    details="Use `n.components[c].get_activity_mask` instead.",
)
def get_activity_mask(
    n: Network,
    c: str,
    sns: Sequence | None = None,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Get active components mask indexed by snapshots.

    Wrapper around the
    [`pypsa.descriptors.components.Componenet.get_active_assets`][] method.
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


@deprecated(
    deprecated_in="1.0.0",
    removed_in="2.0.0",
    details="Use `n.components[c].get_bounds_pu` instead.",
)
def get_bounds_pu(
    n: Network,
    c: str,
    sns: Sequence,
    index: pd.Index | None = None,
    attr: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve per unit bounds of a given component.

    Getter function to retrieve the per unit bounds of a given component for
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
