"""Use common methods for optimization problem definition with Linopy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from numpy import hstack, ravel

from pypsa.constants import RE_PORTS

if TYPE_CHECKING:
    import xarray as xr

    from pypsa import Network


def reindex(ds: xr.DataArray, dim: str, index: pd.Index) -> xr.DataArray:
    """Index a xarray.DataArray by a pandas.Index while renaming according to the
    new index name.

    Parameters
    ----------
    ds : xr.DataArray
    dim : str
    index : pd.Index

    Returns
    -------
    ds
        Reindexed dataarray with renamed dimension.

    """
    return ds.sel({dim: index}).rename({dim: index.name})


def set_from_frame(n: Network, c: str, attr: str, df: pd.DataFrame) -> None:
    """Update values in time-dependent attribute from new dataframe."""
    dynamic = n.dynamic(c)
    if (attr not in dynamic) or (dynamic[attr].empty):
        dynamic[attr] = df.reindex(n.snapshots).fillna(0.0)
    else:
        dynamic[attr].loc[df.index, df.columns] = df
        dynamic[attr] = dynamic[attr].fillna(0.0)


def get_strongly_meshed_buses(n: Network, threshold: int = 45) -> pd.Series:
    """Get the buses which are strongly meshed in the network.

    Parameters
    ----------
    n : Network
    threshhold: int
        number of attached components to be counted as strongly meshed

    Returns
    -------
    pandas series of all meshed buses.

    """
    all_buses = pd.Series(
        hstack(
            [
                ravel(c.static.filter(regex=RE_PORTS.pattern))
                for c in n.iterate_components()
            ]
        )
    )
    all_buses = all_buses[all_buses != ""]
    counts = all_buses.value_counts()
    results = counts.index[counts > threshold].rename("Bus-meshed")
    results = results.sort_values()
    return results
