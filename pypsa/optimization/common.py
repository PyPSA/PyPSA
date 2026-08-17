# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Use common methods for optimization problem definition with Linopy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
from deprecation import deprecated
from numpy import hstack, ravel, roll, zeros

from pypsa.constants import RE_PORTS

if TYPE_CHECKING:
    from linopy import Variable

    from pypsa import Network


def _period_start_mask(sns: pd.Index) -> xr.DataArray:
    """Mark the first snapshot of each investment period."""
    is_start = zeros(len(sns), dtype=bool)
    is_start[0] = True
    if isinstance(sns, pd.MultiIndex) and "period" in sns.names:
        periods = sns.get_level_values("period").to_numpy()
        is_start[1:] = periods[1:] != periods[:-1]
    return xr.DataArray(is_start, coords=[sns])


def _roll_within_periods(v: Variable) -> Variable:
    """Cyclically roll ``v`` by one snapshot within each investment period."""
    sns = v.indexes["snapshot"]
    positions = pd.Series(range(len(sns)), index=sns)
    roll_index = positions.groupby(level="period").transform(lambda s: roll(s, 1))
    coords = xr.Coordinates.from_pandas_multiindex(sns, "snapshot")
    return v.isel(snapshot=roll_index.to_numpy()).assign_coords(coords)


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
