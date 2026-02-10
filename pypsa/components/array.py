# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Array module of PyPSA components.

Contains logic to combine static and dynamic pandas DataFrames to single xarray
DataArray for each variable.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pypsa._options import options
from pypsa.common import UnexpectedError, as_index, list_as_string
from pypsa.components.abstract import _ComponentsABC
from pypsa.guards import _assert_xarray_integrity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Components


def _from_xarray(da: xr.DataArray, c: Components) -> pd.DataFrame | pd.Series:
    """Convert component attribute xarray view back to pandas dataframe or series.

    Based on the dimensions the method returns the pandas format as stored in Network:

    - name: single-indexed series with names as rows and single column as attribute
    - name, snapshot: dataframe with snapshots as rows and names as columns
    - name, scenario: multi-index series with name and scenarios as rows
    - name, snapshot, scenario: multi-index dataframe with snapshots as rows and name/ scenarios as columns

    If name or scenarios (if stochastic) are missing, they will be expanded to cover them

    """
    # Add missing dimensions if needed
    if "name" not in da.dims:
        da = da.expand_dims(name=c.names)
    if "scenario" not in da.dims and c.has_scenarios:
        da = da.expand_dims(scenario=c.scenarios)

    dims = set(da.dims)
    if dims in ({"name"}, {"name", "snapshot"}):
        return da.transpose("snapshot", "name", missing_dims="ignore").to_pandas()

    # c.static with scenarios
    elif dims == {"name", "scenario"}:
        return da.transpose("scenario", "name").to_pandas().stack()

    # c.dynamic with scenarios
    elif dims == {"name", "snapshot", "scenario"}:
        df = (
            da.transpose("name", "scenario", "snapshot", ...)
            .stack(combined=("scenario", "name"))
            .to_pandas()
        )
        # Always return dataframes, also for one-column data
        if isinstance(df, pd.Series):
            df = df.to_frame()

        df.columns.name = None
        return df

    # Handle auxiliary dimensions (e.g. from security constrained optimization)
    elif len(dims) > 2:
        # Find auxiliary dimensions
        contingency_dims = [
            d for d in dims if d not in {"snapshot", "name", "scenario"}
        ]

        if contingency_dims:
            # Stack auxiliary dimensions with component dimension to create combined index
            if "scenario" in dims:
                stack_dims = ["name", "scenario"] + contingency_dims
            else:
                stack_dims = ["name"] + contingency_dims

            combined_name = "combined"
            df = da.stack({combined_name: stack_dims}).to_pandas()

            if hasattr(df, "columns"):
                df.columns.name = None

            return df

    # Handle cases with auxiliary dimensions but no component dimension (e.g. GlobalConstraint with cycle)
    elif len(dims) == 2 and "snapshot" in dims:
        # For 2D cases like ('snapshot', 'cycle'), just use to_pandas() directly
        return da.to_pandas()

    # Handle other cases
    available_dims = list_as_string(dims)
    msg = (
        f"Unexpected combination of dimensions: {available_dims}. "
        f"Expected some combination of 'snapshot', 'name', and 'scenario'."
    )
    raise UnexpectedError(msg)


class _XarrayAccessor:
    """Accessor class that provides property-like xarray access to all attributes.

    Attributes are lazy evaluated via _as_xarray method of the component.
    Supports both attribute access (c.da.p_max_pu) and item access (c.da['p_max_pu']).
    """

    # Use __slots__ to reduce memory footprint (no __dict__ and no dynamic attributes)
    __slots__ = ("_component",)

    def __init__(self, component: ComponentsArrayMixin) -> None:
        object.__setattr__(self, "_component", component)

    def _get_component(self) -> ComponentsArrayMixin:
        """Safely get the component reference to avoid recursion during unpickling."""
        return object.__getattribute__(self, "_component")

    def _get_array(self, attr: str) -> xr.DataArray:
        """Get an xarray DataArray for the specified attribute."""
        component = self._get_component()
        try:
            return component._as_xarray(attr=attr)
        except AttributeError as e:
            msg = f"'{component.__class__.__name__}' components has no attribute '{attr}'."
            raise AttributeError(msg) from e

    def __getattr__(self, attr: str) -> xr.DataArray:
        """Access component attributes as xarray DataArrays via dot notation."""
        return self._get_array(attr)

    def __getitem__(self, attr: str) -> xr.DataArray:
        """Access component attributes as xarray DataArrays via bracket notation."""
        return self._get_array(attr)

    def __dir__(self) -> list[str]:
        """List available attributes for tab-completion."""
        component = self._get_component()
        # Include all static and dynamic attributes
        attrs = set(component.static.columns)
        attrs.update(component.dynamic.keys())
        return sorted(attrs)

    def __str__(self) -> str:
        """Get string representation of the xarray accessor."""
        component = self._get_component()
        return f"'{component.ctype.name}' XarrayAccessor"

    def __repr__(self) -> str:
        """Get representation of the xarray accessor."""
        component = self._get_component()
        return f"'{component.ctype.name}' XarrayAccessor"


class ComponentsArrayMixin(_ComponentsABC):
    """Helper class for components array methods.

    Class inherits to [pypsa.Components][]. All attributes and methods can be used
    within any Components instance.
    """

    def __init__(self) -> None:
        """Initialize the ComponentsArrayMixin."""
        self.da = _XarrayAccessor(self)
        """
        xArray accessor to get component attributes as xarray DataArray.

        Examples
        --------
        >>> c = n.components.generators
        >>> c.da.p_max_pu
        <xarray.DataArray 'p_max_pu' (snapshot: 10, name: 6)> Size: 480B
        array([[0.93001988, 1.        , 0.9745832 , 1.        , 0.5590784 ,
              ...
                1.        ]])
        Coordinates:
        * snapshot  (snapshot) datetime64[ns] 80B 2015-01-01 ... 2015-01-01T09:00:00
        * name      (name) object 48B 'Manchester Wind' ... 'Frankfurt Gas'

        For stochastic networks the scenarios are unstacked automatically:
        >>> c = n_stoch.components.generators
        >>> c.da.p_max_pu
        <xarray.DataArray 'p_max_pu' (snapshot: 2920, scenario: 3, name: 4)> Size: 280kB
        array([[[0.    , 0.1566, 1.    , 1.    ],
              ...
                [0.    , 0.1082, 1.    , 1.    ]]], shape=(2920, 3, 4))
        Coordinates:
        * snapshot  (snapshot) datetime64[ns] 23kB 2015-01-01 ... 2015-12-31T21:00:00
        * scenario  (scenario) object 24B 'low' 'med' 'high'
        * name      (name) object 32B 'solar' 'wind' 'gas' 'lignite'

        String representation:
        >>> c.da
        <XarrayAccessor for Generators>
        """

    def __deepcopy__(
        self, memo: dict[int, object] | None = None
    ) -> ComponentsArrayMixin:
        """Create custom deepcopy which does not copy the xarray accessor."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result  # type: ignore
        for k, v in self.__dict__.items():
            setattr(
                result,
                k,
                _XarrayAccessor(result) if k == "da" else copy.deepcopy(v, memo),
            )
        return result

    def _as_dynamic(
        self,
        attr: str,
        snapshots: Sequence | None = None,
        inds: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Get an attribute as a dynamic DataFrame.

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
        >>> n.components.generators._as_dynamic('p_max_pu', n.snapshots[:2])
        name                 Manchester Wind  ...  Frankfurt Gas
        snapshot                              ...
        2015-01-01 00:00:00         0.930020  ...            1.0
        2015-01-01 01:00:00         0.485748  ...            1.0
        <BLANKLINE>
        [2 rows x 6 columns]

        """
        sns = as_index(self.n_save, snapshots, "snapshots")
        index = self.static.index

        static = self.static.get(attr, pd.Series([], index=index[:0]))
        dynamic = self.dynamic.get(attr, pd.DataFrame(index=sns, columns=index[:0]))

        # Filter snapshots
        if not dynamic.index.equals(sns):
            dynamic = dynamic.reindex(sns, fill_value=np.nan)

        # Filter names
        if inds is not None:
            index = index.intersection(inds)

        # Find columns that need to be filled from static data
        diff = index.difference(dynamic.columns)

        if len(diff) == 0:
            # No static data needed, just slice dynamic
            res = dynamic.reindex(columns=index, fill_value=np.nan)
        else:
            static_subset = static.reindex(diff, fill_value=np.nan)

            if len(static_subset) > 0:
                static_values = static_subset.values
                static_to_dynamic = pd.DataFrame(
                    data=static_values.reshape(1, -1).repeat(len(sns), axis=0),
                    index=sns,
                    columns=diff,
                )
            else:
                static_to_dynamic = pd.DataFrame(index=sns, columns=diff)

            # Concatenate only if there is existing dynamic data
            if len(dynamic) > 0:
                res = pd.concat([dynamic, static_to_dynamic], axis=1, copy=False)
                res = res[index]
            else:
                res = static_to_dynamic

        res.index.name = sns.name
        if self.has_scenarios:
            res.columns.names = static.index.names
            res.columns.name = None
        else:
            res.columns.name = "name"
        return res

    def _as_xarray(self, attr: str) -> xr.DataArray:
        """Get an attribute as a xarray DataArray.

        Converts component data to a flexible xarray DataArray format, which is
        particularly useful for optimization routines. The method provides several
        conveniences:

        1. Automatically handles both static and time-varying attributes
        2. Creates activity masks with the special "active" attribute name
        3. Properly handles scenarios if present in the network

        Parameters
        ----------
        c : pypsa.Components
            Components instance
        attr : str
            Attribute name to retrieve, can be an operational shorthand (e.g., "max_pu")
            or the full attribute name (e.g., "p_max_pu")

        Returns
        -------
        xr.DataArray
            The requested attribute data as an xarray DataArray with appropriate dimensions

        """
        if attr == "active":
            res = xr.DataArray(self.get_activity_mask())
        elif attr in self.dynamic.keys():
            res = xr.DataArray(self._as_dynamic(attr))
        else:
            res = xr.DataArray(self.static[attr])

        # Unstack the dimension that contains the scenarios
        if self.has_scenarios:
            res = (
                res.unstack(res.indexes["scenario"].name)
                .reindex(name=self.names)
                .reindex(scenario=self.scenarios)
            )
            # xr.unstack can leave .dims and .coords in different orders
            # Make sure dimensions match
            res = res.transpose(*[c for c in res.coords if c in res.dims])

        # Set attibute name as DataArray name
        res.name = attr

        # Optional runtime verification
        if options.debug.runtime_verification:
            _assert_xarray_integrity(self, res)

        return res
