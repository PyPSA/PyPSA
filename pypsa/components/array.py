# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Array module of PyPSA components.

Contains logic to combine static and dynamic pandas DataFrames to single xarray
DataArray for each variable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pandas as pd
import xarray as xr

from pypsa._options import options
from pypsa.common import (
    UnexpectedError,
    _is_output,
    as_index,
    experimental,
    list_as_string,
)
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


def _drop_default_rows(rows: pd.DataFrame, default: Any) -> pd.DataFrame:
    """Remove scalar rows that just repeat the attribute's default value."""
    if pd.isnull(default):
        return rows[rows["value"].notna()]
    return rows[rows["value"] != default]


def _concat_long(series_rows: pd.DataFrame, scalar_rows: pd.DataFrame) -> pd.DataFrame:
    """Concatenate series rows and null-snapshot scalar rows of one attribute.

    The scalar rows carry plain NaN in the dimension columns, which pandas
    would flag as an all-NA entry with a conflicting dtype -- take the dtypes
    from the series rows instead.
    """
    if scalar_rows.empty:
        return series_rows
    if series_rows.empty:
        return scalar_rows
    scalar_rows = scalar_rows.copy()
    for col in ("snapshot", "period"):
        scalar_rows[col] = series_rows[col].iloc[:0].reindex(scalar_rows.index)
    return pd.concat([series_rows, scalar_rows[series_rows.columns]], ignore_index=True)


class _ComponentAccessor:
    """Base class for lazy per-attribute accessors (`c.da`, `c.long`)."""

    # Use __slots__ to reduce memory footprint (no __dict__ and no dynamic attributes)
    __slots__ = ("_component",)

    def __init__(self, component: ComponentsArrayMixin) -> None:
        object.__setattr__(self, "_component", component)

    def _get_component(self) -> ComponentsArrayMixin:
        """Safely get the component reference to avoid recursion during unpickling."""
        return object.__getattribute__(self, "_component")

    def _get(self, attr: str) -> Any:
        raise NotImplementedError

    def __getattr__(self, attr: str) -> Any:
        """Access a component attribute via dot notation."""
        return self._get(attr)

    def __getitem__(self, attr: str) -> Any:
        """Access a component attribute via bracket notation."""
        return self._get(attr)

    def __iter__(self) -> NoReturn:
        """Raise a clear error, as accessor objects are not iterable."""
        msg = f"{type(self).__name__.removeprefix('_')} objects are not iterable."
        raise TypeError(msg)

    def __repr__(self) -> str:
        """Get representation of the accessor."""
        name = type(self).__name__.removeprefix("_")
        return f"'{self._get_component().ctype.name}' {name}"


class _XarrayAccessor(_ComponentAccessor):
    """Lazy xarray access to all static and dynamic attributes (see `Components.da`)."""

    __slots__ = ()

    def _get(self, attr: str) -> xr.DataArray:
        """Get an xarray DataArray for the specified attribute."""
        component = self._get_component()
        try:
            return component._as_xarray(attr=attr)
        except (AttributeError, KeyError) as e:
            msg = f"'{component.__class__.__name__}' components has no attribute '{attr}'."
            raise AttributeError(msg) from e

    def __dir__(self) -> list[str]:
        """List available attributes for tab-completion."""
        component = self._get_component()
        # Include all static and dynamic attributes
        attrs = set(component.static.columns)
        attrs.update(component.dynamic.keys())
        return sorted(attrs)


class _LongAccessor(_ComponentAccessor):
    """Lazy long-format access to varying and output attributes (see `Components.long`)."""

    __slots__ = ()

    def _get(self, attr: str) -> pd.DataFrame:
        """Get a dynamic attribute as a long DataFrame."""
        return self._get_component()._as_long(attr)

    def __dir__(self) -> list[str]:
        """List the varying and output attributes for tab-completion."""
        defaults = self._get_component().defaults
        outputs = defaults["status"].astype(str).str.startswith("Output")
        return sorted(defaults.index[defaults.varying | outputs])


class ComponentsArrayMixin(_ComponentsABC):
    """Helper class for components array methods.

    Class inherits to [pypsa.Components][]. All attributes and methods can be used
    within any Components instance.
    """

    @property
    def da(self) -> _XarrayAccessor:
        """XArray accessor to get component attributes as xarray DataArray.

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
        >>> c = n_stochastic.components.generators
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
        'Generator' XarrayAccessor

        """
        return _XarrayAccessor(self)

    @property
    @experimental
    def long(self) -> _LongAccessor:
        """Long/tidy accessor for varying and output attributes.

        <!-- md:badge-experimental --> | <!-- md:badge-version v1.3.0 -->

        Each `c.long.<attr>` is a DataFrame of the four dimensions plus `value`:
        `name, snapshot, scenario, period, value`. Nulls follow the parquet
        store convention, see [pypsa.Network.export_to_parquet][]: what is
        available here mirrors what the parquet long tables hold — varying
        attributes (including ones that currently hold only static input) and
        static outputs like `p_nom_opt`, which appear as null-snapshot scalar
        rows.

        Recomputed on every access, since `c.static` and `c.dynamic` are currently
        mutable with no change hook to invalidate a cache. This might change in future.

        Examples
        --------
        >>> c = n.components.generators
        >>> c.long.p_max_pu.head(3)
                      name   snapshot  scenario  period     value
        0  Manchester Wind 2015-01-01       NaN     NaN  0.930020
        1   Frankfurt Wind 2015-01-01       NaN     NaN  0.559078
        2      Norway Wind 2015-01-01       NaN     NaN  0.974583

        """
        return _LongAccessor(self)

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
            Values of `attr` for every component, indexed by snapshot.

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
                res = pd.concat([dynamic, static_to_dynamic], axis=1)
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

    def _stack_dynamic(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Melt a wide `snapshots x components` frame into long rows.

        Examples
        --------
        >>> c._stack_dynamic(c.dynamic['p_max_pu']).head(3)
            snapshot             name     value  scenario  period
        0 2015-01-01  Manchester Wind  0.930020       NaN     NaN
        1 2015-01-01   Frankfurt Wind  0.559078       NaN     NaN
        2 2015-01-01      Norway Wind  0.974583       NaN     NaN

        """
        stochastic = isinstance(wide.columns, pd.MultiIndex)
        wide = wide.rename_axis(columns=["scenario", "name"] if stochastic else "name")

        stacked = wide.stack(level=list(range(wide.columns.nlevels)), future_stack=True)
        long = stacked.rename("value").reset_index()
        if isinstance(self.snapshots, pd.MultiIndex):
            long = long.rename(columns={"timestep": "snapshot"})
        if "scenario" not in long.columns:
            long["scenario"] = np.nan
        if "period" not in long.columns:
            long["period"] = np.nan
        return long

    def _scalar_rows(self, attr: str, exclude: pd.Index) -> pd.DataFrame:
        """Build null snapshot long rows from the static scalars of `attr`.

        Examples
        --------
        Components with a time series are excluded and the remaining static
        scalars become null snapshot rows:

        >>> c._scalar_rows('p_max_pu', c.dynamic['p_max_pu'].columns)
                     name  value  scenario  snapshot  period
        0  Manchester Gas    1.0       NaN       NaN     NaN
        1      Norway Gas    1.0       NaN       NaN     NaN
        2   Frankfurt Gas    1.0       NaN       NaN     NaN

        """
        static = self.static[attr]
        if len(exclude):
            static = static[~static.index.isin(exclude)]
        out = static.rename("value").reset_index()
        if "scenario" not in out.columns:
            out["scenario"] = np.nan
        out["snapshot"] = np.nan
        out["period"] = np.nan
        return out

    def _as_long(self, attr: str, *, drop_defaults: bool = False) -> pd.DataFrame:
        """Get an attribute as a long/tidy DataFrame.

        See the `long` accessor. With `drop_defaults`, scalar rows equal to
        the attribute's default are omitted.
        """
        varying = attr in self.dynamic or (
            attr in self.defaults.index and bool(self.defaults.at[attr, "varying"])
        )
        if not varying and not _is_output(self.defaults, attr):
            msg = (
                f"'{attr}' of '{self.__class__.__name__}' components has no "
                f"long representation."
            )
            raise AttributeError(msg)

        cols = ["name", "snapshot", "scenario", "period", "value"]
        if not varying:
            out = self._scalar_rows(attr, pd.Index([]))[cols]
            if drop_defaults:
                out = _drop_default_rows(out, self.defaults.at[attr, "default"])
            return out
        wide = self.dynamic[attr]
        out = self._stack_dynamic(wide)[cols]
        # scalar rows import into `c.static[attr]`, so only attrs with a
        # static column get them
        static_backed = attr in self.defaults.index and bool(
            self.defaults.at[attr, "static"]
        )
        if static_backed:
            scalars = self._scalar_rows(attr, wide.columns)[cols]
            if drop_defaults:
                scalars = _drop_default_rows(scalars, self.defaults.at[attr, "default"])
            out = _concat_long(out, scalars)
        return out

    def _export_long_frames(self, static_outputs: list[str]) -> list[pd.DataFrame]:
        """Build the parquet store's long frame for every exported attribute."""
        frames = []
        for attr in dict.fromkeys([*self.dynamic, *static_outputs]):
            long = self._as_long(attr, drop_defaults=True)
            if long.empty:
                continue
            frames.append(long.assign(component_type=self.name, attribute=attr))
        return frames
