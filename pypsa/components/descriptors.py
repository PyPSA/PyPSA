"""
Descriptors module of PyPSA components.

Contains all descriptor functions which can be used as methods of the
Components class. Descriptor functions only describe data and do not
modify it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pypsa.descriptors import expand_series

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pypsa import Components


def get_active_assets(
    c: Components,
    investment_period: int | str | Sequence | None = None,
) -> pd.Series:
    """
    Get active components mask of componen type in investment period(s).

    A component is considered active when:
    - it's active attribute is True
    - it's build year + lifetime is smaller than the investment period (if given)

    Parameters
    ----------
    c : pypsa.Components
        Components instance.
    investment_period : int, str, Sequence
        Investment period(s) to check for active within build year and lifetime. If
        none only the active attribute is considered and build year and lifetime are
        ignored. If multiple periods are given the mask is True if component is
        active in any of the given periods.

    Returns
    -------
    pd.Series
        Boolean mask for active components

    """
    if investment_period is None:
        return c.static.active
    if not {"build_year", "lifetime"}.issubset(c.static):
        return c.static.active

    # Logical OR of active assets in all investment periods and
    # logical AND with active attribute
    active = {}
    for period in np.atleast_1d(investment_period):
        if period not in c.n_save.investment_periods:
            raise ValueError("Investment period not in `n.investment_periods`")
        active[period] = c.static.eval("build_year <= @period < build_year + lifetime")
    return pd.DataFrame(active).any(axis=1) & c.static.active


def get_activity_mask(
    c: Components,
    sns: Sequence | None = None,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Get active components mask indexed by snapshots.

    Gets the boolean mask for active components, indexed by snapshots and
    components instead of just components.

    Parameters
    ----------
    c : pypsa.Components
        Components instance.
    sns : pandas.Index, default None
        Set of snapshots for the mask. If None (default) all snapshots are returned.
    index : pd.Index, default None
        Subset of the component elements. If None (default) all components are returned.

    Returns
    -------
    pd.DataFrame
        Boolean mask for active components indexed by snapshots.

    """
    sns_ = c.snapshots if sns is None else sns

    if c.has_investment_periods:
        active_assets_per_period = {
            period: c.get_active_assets(investment_period=period)
            for period in c.investment_periods
        }
        mask = (
            pd.concat(active_assets_per_period, axis=1)
            .T.reindex(c.snapshots, level=0)
            .loc[sns_]
        )
    else:
        active_assets = c.get_active_assets()
        mask = pd.DataFrame(
            np.tile(active_assets, (len(sns_), 1)),
            index=sns_,
            columns=active_assets.index,
        )

    if index is not None:
        mask = mask.reindex(columns=index)

    return mask


def get_bounds_pu(
    c: Components,
    sns: Sequence,
    index: pd.Index | None = None,
    attr: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get per unit bounds of the component for given snapshots.

    Retrieve the per unit bounds of this component for given snapshots and
    possibly a subset of elements (e.g. non-extendables).
    Depending on the attr you can further specify the bounds of the variable
    you are looking at, e.g. p_store for storage units.

    Parameters
    ----------
    c : pypsa.Components
        Components instance.
    sns : pandas.Index/pandas.DateTimeIndex
        Set of snapshots for the bounds
    index : pd.Index, default None
        Subset of the component elements. If None (default) bounds of all
        elements are returned.
    attr : string, default None
        Attribute name for the bounds, e.g. "p", "s", "p_store"

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (min_pu, max_pu) DataFrames.

    """
    min_pu_str = c.operational_attrs["min_pu"]
    max_pu_str = c.operational_attrs["max_pu"]

    max_pu = c.as_dynamic(max_pu_str, sns)

    if c.category in {"passive_branch"}:
        min_pu = -max_pu
    elif c.name == "StorageUnit":
        min_pu = pd.DataFrame(0, max_pu.index, max_pu.columns)
        if attr == "p_store":
            max_pu = -c.as_dynamic(min_pu_str, sns)
        if attr == "state_of_charge":
            max_pu = expand_series(c.static.max_hours, sns).T
            min_pu = pd.DataFrame(0, *max_pu.axes)
    else:
        min_pu = c.as_dynamic(min_pu_str, sns)

    if index is None:
        return min_pu, max_pu
    else:
        return min_pu.reindex(columns=index), max_pu.reindex(columns=index)


# TODO: remove as soon as deprecated renaming is removed
def get_extendable_i(c: Components, rename_index: bool = True) -> pd.Index:
    """Get the index of extendable elements of this component."""
    idx = c.extendables
    return idx.rename(idx.name + "-ext") if rename_index else idx


# TODO: remove as soon as deprecated renaming is removed
def get_non_extendable_i(c: Components, rename_index: bool = True) -> pd.Index:
    """Get the index of non-extendable elements of this component."""
    idx = c.fixed
    return idx.rename(idx.name + "-fix") if rename_index else idx


# TODO: remove as soon as deprecated renaming is removed
def get_committable_i(c: Components, rename_index: bool = True) -> pd.Index:
    """Get the index of committable elements of this component."""
    idx = c.committables
    return idx.rename(idx.name + "-com") if rename_index else idx
