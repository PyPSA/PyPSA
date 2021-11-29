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
