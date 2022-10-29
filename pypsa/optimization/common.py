#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use common methods for optimization problem definition with Linopy.
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


def set_from_frame(n, c, attr, df):
    """
    Update values in time-dependent attribute from new dataframe.
    """
    pnl = n.pnl(c)
    if attr not in pnl:
        return
    if pnl[attr].empty:
        pnl[attr] = df.reindex(n.snapshots).fillna(0)
    else:
        pnl[attr].loc[df.index, df.columns] = df
        pnl[attr] = pnl[attr].fillna(0)
