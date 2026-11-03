# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Use common methods for optimization problem definition with Linopy.

Snapshot naming convention
---------------------------
- `sns`: snapshots the current model build is constructed over, in the model's
  labelling (see `pypsa.optimization.window`).
- `window`: the build's `SnapshotWindow`, which maps those labels back to the
  network's.
- `period_sns`: per-investment-period subset of `sns`, as yielded by
  `window.iter_periods`.

`snapshots` is reserved for the public API (`n.optimize(snapshots=...)`,
`n.snapshots`); `"snapshot"` is the model dimension name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from deprecation import deprecated
from numpy import hstack, ravel

from pypsa.constants import RE_PORTS

if TYPE_CHECKING:
    import xarray as xr

    from pypsa import Network
    from pypsa.components import Components


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


_VARIABLE_FEATURE_DEFAULTS = (
    ("e_sum_min", -float("inf")),
    ("e_sum_max", float("inf")),
    ("marginal_cost", 0.0),
    ("marginal_cost_quadratic", 0.0),
)


def compute_constant_operation_assets(
    n: Network, c: Components, sns: pd.Index
) -> pd.Index:
    """Get the assets of a component that enter the model as constants.

    Only applies to components that opt in via `_constant_fixed_operation`
    (currently `Load`). An asset is a constant if no feature requires a
    decision variable, i.e. it is not extendable, committable, modular or
    maintainable, has no ramp limits, no `e_sum_min`/`e_sum_max` limits and no
    marginal costs, and either

    - its `p_set` is given for all snapshots `sns` (and all scenarios), or
    - it has no capacity (`p_nom == 0`), so its operation is zero wherever
      `p_set` is NaN. This keeps loads stored with the former default
      `p_set = 0` (which is not written to files) out of the model.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    c : pypsa.Components
        Components to evaluate.
    sns : pd.Index
        Snapshots of the model build.

    Returns
    -------
    pd.Index
        Names of the assets whose operation is a constant.

    """
    empty = pd.Index([], name="name")
    if not getattr(c, "_constant_fixed_operation", False) or c.empty:
        return empty
    names = c.active_assets
    if names.empty or "p_set" not in c.static:
        return empty

    def _any(da: xr.DataArray) -> pd.Series:
        dims = [d for d in da.dims if d != "name"]
        reduced = da.any(dims) if dims else da
        return reduced.to_series().reindex(names, fill_value=False)

    p_set = c.da.p_set.sel(name=names, snapshot=sns)
    constant = ~_any(p_set.isnull())
    nom_attr = c._operational_attrs["nom"]
    if nom_attr in c.static:
        constant |= ~_any(c.da[nom_attr].sel(name=names) != 0)

    needs_var = pd.Series(
        names.isin(c.extendables)
        | names.isin(c.committables)
        | names.isin(c.modulars)
        | names.isin(c.maintainables),
        index=names,
    )
    for attr in (
        "ramp_limit_up",
        "ramp_limit_down",
        "ramp_limit_start_up",
        "ramp_limit_shut_down",
    ):
        if attr in c.static:
            da = c.da[attr].sel(name=names)
            if "snapshot" in da.dims:
                da = da.sel(snapshot=sns)
            needs_var |= _any(da.notnull())
    for attr, default in _VARIABLE_FEATURE_DEFAULTS:
        if attr in c.static:
            da = c.da[attr].sel(name=names)
            if "snapshot" in da.dims:
                da = da.sel(snapshot=sns)
            needs_var |= _any(da != default)

    return names[(constant & ~needs_var).to_numpy()]


def get_constant_operation_assets(n: Network, c: Components) -> pd.Index:
    """Get the assets entering the current model build as constants.

    The split is computed once per model build in `create_model` and stored on
    the build's snapshot window. Falls back to computing it on the fly.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    c : pypsa.Components
        Components to evaluate.

    Returns
    -------
    pd.Index
        Names of the assets whose operation is a constant.

    """
    window = n.optimize._window
    if c.name in window.constant_assets:
        return window.constant_assets[c.name]
    return compute_constant_operation_assets(n, c, window.model_index)


def get_operational_assets(n: Network, c: Components) -> pd.Index:
    """Get the active assets that have an operational decision variable.

    These are all active assets except the ones entering the model as
    constants, see `get_constant_operation_assets`.

    Parameters
    ----------
    n : pypsa.Network
        Network instance.
    c : pypsa.Components
        Components to evaluate.

    Returns
    -------
    pd.Index
        Names of the assets with an operational variable.

    """
    if not getattr(c, "_constant_fixed_operation", False):
        return c.active_assets
    return c.active_assets.difference(get_constant_operation_assets(n, c))


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
