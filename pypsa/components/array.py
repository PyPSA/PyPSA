"""
Array module of PyPSA components.

Contains logic to combine static and dynamic pandas DataFrames to single xarray
DataArray for each variable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
import xarray

from pypsa.common import as_index

if TYPE_CHECKING:
    from pypsa import Components


# TODO: new-opt deprecation
def as_dynamic(
    c: Components,
    attr: str,
    snapshots: Sequence | None = None,
    inds: pd.Index | None = None,
) -> pd.DataFrame:
    """
    Get an attribute as a dynamic DataFrame.

    Return a Dataframe for a time-varying component attribute with values for
    all non-time-varying components filled in with the default values for the
    attribute.


    Parameters
    ----------
    c : pypsa.Components
        Components instance
    component : string
        Component object name, e.g. 'Generator' or 'Link'
    attr : string
        Attribute name
    snapshots : pandas.Index
        Restrict to these snapshots rather than n.snapshots.
    inds : pandas.Index
        Restrict to these components rather than n.components.index

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> import pypsa
    >>> n = pypsa.examples.ac_dc_meshed()
    >>> n.components.generators.as_dynamic('p_max_pu', n.snapshots[:2]) # doctest: +SKIP
    Generator            Manchester Wind  Manchester Gas  Norway Wind  Norway Gas  Frankfurt Wind  Frankfurt Gas
    snapshot
    2015-01-01 00:00:00         0.930020             1.0     0.974583         1.0        0.559078            1.0
    2015-01-01 01:00:00         0.485748             1.0     0.481290         1.0        0.752910            1.0

    """
    sns = as_index(c.n_save, snapshots, "snapshots")
    index = c.static.index
    empty_index = index[:0]  # keep index name and names
    empty_static = pd.Series([], index=empty_index)
    static = c.static.get(attr, empty_static)
    empty_dynamic = pd.DataFrame(index=sns, columns=empty_index)
    dynamic = c.dynamic.get(attr, empty_dynamic).loc[sns]

    if inds is not None:
        index = index.intersection(inds)

    diff = index.difference(dynamic.columns)
    static_to_dynamic = pd.DataFrame({**static[diff]}, index=sns)
    res = pd.concat([dynamic, static_to_dynamic], axis=1, names=sns.names)[index]
    res.index.name = sns.name
    if c.has_scenarios:
        res.columns.name = "Component"
        res.columns.names = static.index.names
    else:
        res.columns.name = c.name
    return res


def as_xarray(
    c: Components,
    attr: str,
    snapshots: Sequence | None = None,
    inds: pd.Index | None = None,
) -> xarray.DataArray:
    """
    Get an attribute as a xarray DataArray.

    Converts component data to a flexible xarray DataArray format, which is
    particularly useful for optimization routines. The method provides several
    conveniences:

    1. Supports short attribute name aliases through the `operational_attrs` mapping
        (e.g., "max_pu" instead of "p_max_pu")
    2. Automatically handles both static and time-varying attributes
    3. Creates activity masks with the special "active" attribute name
    4. Properly handles scenarios if present in the network

    Parameters
    ----------
    c : pypsa.Components
        Components instance
    attr : str
        Attribute name to retrieve, can be an operational shorthand (e.g., "max_pu")
        or the full attribute name (e.g., "p_max_pu")
    snapshots : Sequence | None, optional
        Snapshots to include. If None, uses all snapshots for time-varying data
        or returns static data as-is
    inds : pd.Index | None, optional
        Component indices to filter by. If None, includes all components

    Returns
    -------
    xarray.DataArray
        The requested attribute data as an xarray DataArray with appropriate dimensions

    Examples
    --------
    >>> import pypsa
    >>> n = pypsa.examples.ac_dc_meshed()

    # Get power output limits for generators for the first two snapshots
    >>> limit = n.components.generators.as_xarray('p_max_pu', n.snapshots[:2])

    # Use operational attribute shorthand
    >>> limit = n.components.generators.as_xarray('max_pu', n.snapshots[:2])

    # Get activity mask for lines
    >>> acitve = n.components.lines.as_xarray('active')

    # Get nominal capacity for specific generators
    >>> gens = pd.Index(['Manchester Wind', 'Norway Wind'], name='Generator')
    >>> p_nom = n.components.generators.as_xarray('p_nom', inds=gens)

    """
    if attr in c.operational_attrs.keys():
        attr = c.operational_attrs[attr]

    if attr == "active":
        res = xarray.DataArray(c.get_activity_mask(snapshots, inds))
    elif attr in c.dynamic.keys() or snapshots is not None:
        res = xarray.DataArray(c.as_dynamic(attr, snapshots, inds))
    else:
        if inds is not None:
            data = c.static[attr].reindex(inds)
        else:
            data = c.static[attr]
        res = xarray.DataArray(data)

    if c.has_scenarios:
        # untack the dimension that contains the scenarios
        res = res.unstack(res.indexes["scenario"].name)
    return res
