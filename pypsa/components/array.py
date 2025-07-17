"""Array module of PyPSA components.

Contains logic to combine static and dynamic pandas DataFrames to single xarray
DataArray for each variable.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray

from pypsa.common import as_index
from pypsa.components.abstract import _ComponentsABC

if TYPE_CHECKING:
    from collections.abc import Sequence


class _XarrayAccessor:
    """Accessor class that provides property-like xarray access to all attributes.

    Attributes are lazy evaluated via _as_xarray method of the component.
    """

    def __init__(self, component: ComponentsArrayMixin) -> None:
        self._component = component

    def __getattr__(self, attr: str) -> xarray.DataArray:
        try:
            return self._component._as_xarray(attr=attr)
        except AttributeError as e:
            msg = (
                f"'{self._component.__class__.__name__}' components has no "
                "attribute '{attr}'"
            )
            raise AttributeError(msg) from e

    def __getitem__(self, attr: str) -> xarray.DataArray:
        try:
            return self._component._as_xarray(attr=attr)
        except AttributeError as e:
            msg = (
                f"'{self._component.__class__.__name__}' components has no "
                "attribute '{attr}'"
            )
            raise AttributeError(msg) from e


class ComponentsArrayMixin(_ComponentsABC):
    """Helper class for components array methods.

    Class only inherits to Components and should not be used directly.
    """

    def __init__(self) -> None:
        """Initialize the ComponentsArrayMixin."""
        self.da = _XarrayAccessor(self)

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
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
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
        else:
            res.columns.name = "name"
        return res

    def _as_xarray(
        self,
        attr: str,
        snapshots: Sequence | None = None,
        inds: Sequence | None = None,
        drop_scenarios: bool = False,  # TODO: Do we need that?
    ) -> xarray.DataArray:
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
        snapshots : Sequence | None, optional
            Snapshots to include. If None, uses all snapshots for time-varying data
            or returns static data as-is
        inds : pd.Index | None, optional
            Component indices to filter by. If None, includes all components
        drop_scenarios : bool, default False
            If True, drops the scenario dimension from the resulting DataArray
            by selecting the first scenario.

        Returns
        -------
        xarray.DataArray
            The requested attribute data as an xarray DataArray with appropriate dimensions

        """
        # Strip any index name information
        # snapshots = getattr(snapshots, "values", snapshots) # TODO # noqa: ERA001
        inds = getattr(inds, "values", inds)

        if attr == "active":
            res = xarray.DataArray(self.get_activity_mask(snapshots))
        elif attr in self.dynamic.keys() or snapshots is not None:
            res = self._as_dynamic(attr, snapshots)
            if self.has_scenarios:
                # TODO implement this better
                res.columns.name = None
            res = xarray.DataArray(res)
        else:
            res = xarray.DataArray(self.static[attr])

        # Rename dimension
        # res = res.rename({self.name: "component"}) # noqa: ERA001

        if self.has_scenarios:
            # untack the dimension that contains the scenarios
            res = res.unstack(res.indexes["scenario"].name)
            # Ensure scenario order matches network's scenario order
            if "scenario" in res.dims:
                res = res.reindex(scenario=self.scenarios)
            if drop_scenarios:
                res = res.isel(scenario=0, drop=True)

        if inds is not None:
            res = res.sel(name=inds)

        if self.has_periods:
            try:
                res = res.rename(dim_0="snapshot")
            except ValueError:
                pass

        res.name = attr

        return res
