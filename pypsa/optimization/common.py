#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:29:33 2021

@author: fabian
"""


def reindex(ds, dim, index):
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


def get_var(n, c, key):
    """Get variables directly from network"""
    return n.model[f"{c}-{key}"]


def set_from_frame(n, c, attr, df):
    """Update values in time-dependent attribute from new dataframe."""
    pnl = n.pnl(c)
    if pnl[attr].empty:
        pnl[attr] = df.reindex(n.snapshots, fill_value=0)
    else:
        pnl[attr].update(df)
