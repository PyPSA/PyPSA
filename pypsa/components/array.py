"""Array module of PyPSA components.

Contains logic to combine static and dynamic pandas DataFrames to single xarray
DataArray for each variable.
"""

from __future__ import annotations

import copy
import inspect
import os
from typing import TYPE_CHECKING

import pandas as pd
import xarray

from pypsa.common import as_index
from pypsa.components.abstract import _ComponentsABC

if TYPE_CHECKING:
    from collections.abc import Sequence


class _XarrayAccessor:
    """Accessor class that provides property-like xarray access to all attributes.

    Attributes are lazy evaluated via as_xarray method of the component.
    """

    def __init__(self, component: ComponentsArrayMixin) -> None:
        self._component = component

    def __getattr__(self, attr: str) -> xarray.DataArray:
        try:
            return self._component.as_xarray(attr=attr)
        except AttributeError as e:
            msg = (
                f"'{self._component.__class__.__name__}' components has no "
                "attribute '{attr}'"
            )
            raise AttributeError(msg) from e

    def __getitem__(self, attr: str) -> xarray.DataArray:
        try:
            return self._component.as_xarray(attr=attr)
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
        component            Manchester Wind  ...  Frankfurt Gas
        snapshot                              ...
        2015-01-01 00:00:00         0.930020  ...            1.0
        2015-01-01 01:00:00         0.485748  ...            1.0
        <BLANKLINE>
        [2 rows x 6 columns]

        """
        # Check if we are in a power flow calculation
        stack = inspect.stack()
        in_pf = any(os.path.basename(frame.filename) == "pf.py" for frame in stack)  # noqa: PTH119

        sns = as_index(self.n_save, snapshots, "snapshots")
        index = self.static.index
        empty_index = index[:0]  # keep index name and names
        empty_static = pd.Series([], index=empty_index)
        static = self.static.get(attr, empty_static)
        empty_dynamic = pd.DataFrame(index=sns, columns=empty_index)
        dynamic = self.dynamic.get(attr, empty_dynamic).loc[sns]

        if inds is not None:
            index = index.intersection(inds)

        diff = index.difference(dynamic.columns)
        static_to_dynamic = pd.DataFrame({**static[diff]}, index=sns)
        res = pd.concat([dynamic, static_to_dynamic], axis=1, names=sns.names)[index]

        # power flow calculations in pf.py require a starting point for the algorithm, while p_set default is n/a
        if attr == "p_set" and in_pf:
            res = res.fillna(0)

        res.index.name = sns.name
        res.columns.name = "component"
        return res

    def as_xarray(
        self,
        attr: str,
        snapshots: Sequence | None = None,
        inds: Sequence | None = None,
    ) -> xarray.DataArray:
        """Get an attribute as a xarray DataArray.

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
        >>> active = n.components.lines.as_xarray('active')

        # Get nominal capacity for specific generators
        >>> gens = pd.Index(['Manchester Wind', 'Norway Wind'], name='Generator')
        >>> p_nom = n.components.generators.as_xarray('p_nom', inds=gens)

        """
        # Strip any index name information
        # snapshots = getattr(snapshots, "values", snapshots) # TODO # noqa: ERA001
        inds = getattr(inds, "values", inds)

        if attr in self.operational_attrs.keys():
            attr = self.operational_attrs[attr]

        if attr == "active":
            res = xarray.DataArray(self.get_activity_mask(snapshots, inds))
        elif attr in self.dynamic.keys() or snapshots is not None:
            res = xarray.DataArray(self._as_dynamic(attr, snapshots, inds))
        else:
            if inds is not None:
                data = self.static[attr].reindex(inds)
                data.index.name = "component"
            else:
                data = self.static[attr]
                data.index.name = "component"
            res = xarray.DataArray(data)

        # Rename dimension
        # res = res.rename({self.name: "component"}) # noqa: ERA001

        return res
