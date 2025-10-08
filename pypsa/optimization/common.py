"""Use common methods for optimization problem definition with Linopy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from numpy import hstack, ravel

from pypsa.common import deprecated_in_next_major
from pypsa.constants import RE_PORTS

if TYPE_CHECKING:
    import xarray as xr

    from pypsa import Network


@deprecated_in_next_major(
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
        c.dynamic[attr].loc[df.index, df.columns] = df

    c.dynamic[attr] = (
        c.dynamic[attr]
        .reindex(n.snapshots, level="snapshot", axis=0)
        .reindex(c.names, level="name", axis=1)
        .fillna(0.0)
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
    all_buses = pd.Series(
        hstack([ravel(c.static.filter(regex=RE_PORTS.pattern)) for c in n.components])
    )
    all_buses = all_buses[all_buses != ""]
    counts = all_buses.value_counts()
    results = counts.index[counts > threshold].rename("Bus")
    results = results.sort_values()
    return results
