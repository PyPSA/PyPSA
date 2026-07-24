# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Use common methods for optimization problem definition with Linopy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr
from deprecation import deprecated
from linopy import merge
from numpy import hstack, ravel, roll, zeros

from pypsa._linopy_compat import (
    SNAPSHOT_LEVELS,
    attach_snapshot_aux,
    drop_snapshot_aux,
)
from pypsa.constants import RE_PORTS

if TYPE_CHECKING:
    from linopy import LinearExpression, Variable

    from pypsa import Network


def build_window(n: Network, sns: pd.Index) -> pd.Index:
    """(Multi)Index snapshots of the current build window.

    While a multi-period model is built over a flat positional ``snapshot`` dim,
    the original MultiIndex window is stashed on the network; return it. Outside
    the flat path ``sns`` already *is* the window, so return it unchanged. Single
    source of truth for the flat position -> period mapping.
    """
    window = n._optimize_window_snapshots
    return sns if window is None else window


def snapshot_weightings(n: Network, sns: pd.Index, kind: str) -> pd.Series:
    """Snapshot weightings of ``kind`` aligned to ``sns``.

    During multi-period model building ``sns`` is a flat positional index, while
    the weightings are indexed by the window MultiIndex; select by the latter and
    relabel to the flat index. A no-op relabel outside the flat path.
    """
    return n.snapshot_weightings[kind].loc[build_window(n, sns)].set_axis(sns)


def iter_snapshot_periods(n: Network, sns: pd.Index) -> Any:
    """Yield ``(period, snapshots)`` pairs for per-period constraint building.

    Works with the flat positional snapshot dim used during model building: each
    snapshot's period comes from the build window (aligned by position), so
    ``sns`` need not be a MultiIndex.
    """
    if not n._multi_invest:
        yield None, sns
        return
    window = build_window(n, sns)
    period_of = window.get_level_values("period")
    for period in window.unique("period"):
        yield period, sns[period_of == period]


def _period_start_mask(n: Network, sns: pd.Index) -> xr.DataArray:
    """Mark the first snapshot of each investment period within the build window."""
    is_start = zeros(len(sns), dtype=bool)
    is_start[0] = True
    window = build_window(n, sns)
    if isinstance(window, pd.MultiIndex) and "period" in window.names:
        periods = window.get_level_values("period").to_numpy()
        is_start[1:] = periods[1:] != periods[:-1]
    return xr.DataArray(is_start, coords=[sns])


def _roll_within_periods(v: Variable) -> Variable:
    """Cyclically roll ``v`` by one snapshot within each investment period.

    Groups by the ``period`` auxiliary coordinate carried on the flat snapshot
    dim, then restores the original snapshot coordinates after the positional
    roll so the result stays aligned with the un-rolled variable.
    """
    sns = v.indexes["snapshot"]
    period = v.coords["period"].to_numpy()
    positions = pd.Series(range(len(sns)), index=sns)
    roll_index = positions.groupby(period).transform(lambda s: roll(s, 1))
    rolled = v.isel(snapshot=roll_index.to_numpy())
    keep = {c: v.coords[c] for c in ("snapshot", *SNAPSHOT_LEVELS) if c in v.coords}
    return rolled.assign_coords(keep)


def merge_over_snapshots(exprs: list, n: Network, sns: pd.Index) -> LinearExpression:
    """Merge expressions on ``snapshot`` while preserving the flat-snapshot aux coords.

    The aux coords must be dropped before the strict outer merge — differing periods
    on collided labels read as a conflict — and re-derived from the tuple labels after.
    """
    merged = merge([drop_snapshot_aux(e) for e in exprs], dim="snapshot", join="outer")
    return attach_snapshot_aux(merged, build_window(n, sns))


@deprecated(
    deprecated_in="1.0.0",
    removed_in="2.0.0",
    details="Use xarray functionality instead (e.g. `ds.sel({dim: index}).rename({dim: index.name})`).",
)
def reindex(ds: xr.DataArray, dim: str, index: pd.Index) -> xr.DataArray:
    """Index a xarray.DataArray by a pandas.Index while renaming according to the new index name.

    Parameters
    ----------
    ds : xr.DataArray
        The input DataArray to reindex.
    dim : str
        The dimension name to reindex.
    index : pd.Index
        The new index to use for reindexing.

    Returns
    -------
    ds
        Reindexed dataarray with renamed dimension.

    """
    return ds.sel({dim: index}).rename({dim: index.name})


def _set_dynamic_data(n: Network, component: str, attr: str, df: pd.DataFrame) -> None:
    """Update values in time-dependent attribute from new dataframe."""
    c = n.components[component]
    if (attr not in c.dynamic) or (c.dynamic[attr].empty):
        c.dynamic[attr] = df.reindex(n.snapshots)

    else:
        c.dynamic[attr] = df.combine_first(c.dynamic[attr])

    # Reindex to match network snapshots and component names
    result = c.dynamic[attr].reindex(n.snapshots, level="snapshot", axis=0)
    if n.has_scenarios:
        expected_columns = pd.MultiIndex.from_product(
            [n.scenarios, c.names], names=["scenario", "name"]
        )
        result = result.reindex(columns=expected_columns)
    else:
        # Preserve auxiliary dimensions (e.g. contingencies), using level="name"
        # Note that we don't have a case to preserve auxiliary dimensions
        # with stochastic dimension currently
        result = result.reindex(c.names, level="name", axis=1)

    c.dynamic[attr] = result.fillna(0.0)


def get_bus_counts(n: Network) -> pd.Series:
    """Count how often each bus appears in component bus columns.

    Parameters
    ----------
    n : Network
        The network to analyze.

    Returns
    -------
    pandas.Series
        Bus usage counts indexed by bus name.

    """
    all_buses = pd.Series(
        hstack([ravel(c.static.filter(regex=RE_PORTS.pattern)) for c in n.components])
    )
    all_buses = all_buses[all_buses != ""]
    return all_buses.value_counts()


@deprecated(
    deprecated_in="1.1.3",
    removed_in="2.0.0",
    details="Use `get_bus_counts(n).loc[lambda s: s > threshold].index` instead.",
)
def get_strongly_meshed_buses(n: Network, threshold: int = 45) -> pd.Series:
    """Get the buses which are strongly meshed in the network.

    Parameters
    ----------
    n : Network
        The network to analyze.
    threshold : int
        Number of attached components to be counted as strongly meshed.

    Returns
    -------
    pandas series of all meshed buses.

    """
    counts = get_bus_counts(n)
    results = counts.index[counts > threshold].rename("Bus")
    results = results.sort_values()
    return results
