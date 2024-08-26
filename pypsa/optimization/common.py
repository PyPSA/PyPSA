#!/usr/bin/env python3
"""
Use common methods for optimization problem definition with Linopy.
"""

from typing import TYPE_CHECKING

import pandas as pd
from numpy import hstack, ravel

if TYPE_CHECKING:
    import xarray as xr

    from pypsa.components import Network


def reindex(ds: "xr.DataArray", dim: str, index: pd.Index) -> "xr.DataArray":
    """
    Index a xarray.DataArray by a pandas.Index while renaming according to the
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


def set_from_frame(n: "Network", c: str, attr: str, df: pd.DataFrame) -> None:
    """
    Update values in time-dependent attribute from new dataframe.
    """
    pnl = n.pnl(c)
    if (attr not in pnl) or (pnl[attr].empty):
        pnl[attr] = df.reindex(n.snapshots).fillna(0.0)
    else:
        pnl[attr].loc[df.index, df.columns] = df
        pnl[attr] = pnl[attr].fillna(0.0)


def get_strongly_meshed_buses(n: "Network", threshold: int = 45) -> pd.Series:
    """
    Get the buses which are strongly meshed in the network.

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
        hstack([ravel(c.df.filter(like="bus")) for c in n.iterate_components()])
    )
    all_buses = all_buses[all_buses != ""]
    counts = all_buses.value_counts()
    return counts.index[counts > 20].rename("Bus-meshed")
