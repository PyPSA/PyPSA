"""Stores components module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import pandas as pd
import xarray as xr
from xarray import DataArray

from pypsa.components.components import Components


class Stores(Components):
    """
    Stores components class.

    This class is used for store components. All functionality specific to
    stores is implemented here. Functionality for all components is implemented in
    the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    See Also
    --------
    pypsa.components.abstract.Components : Base class for all components.

    """

    base_attr = "e"

    @overload
    def get_bounds_pu(
        self,
        sns: Sequence,
        index: pd.Index | None = None,
        attr: str | None = None,
        as_xarray: Literal[True] = True,
    ) -> tuple[xr.DataArray, xr.DataArray]: ...

    @overload
    def get_bounds_pu(
        self,
        sns: Sequence,
        index: pd.Index | None = None,
        attr: str | None = None,
        as_xarray: Literal[False] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    def get_bounds_pu(
        self,
        sns: Sequence,
        index: pd.Index | None = None,
        attr: str | None = None,
        as_xarray: bool = False,
    ) -> tuple[pd.DataFrame | DataArray]:
        """
        Get per unit bounds for stores.

        Parameters
        ----------
        sns : pandas.Index/pandas.DateTimeIndex
            Set of snapshots for the bounds
        index : pd.Index, optional
            Subset of the component elements
        attr : string, optional
            Attribute name for the bounds, e.g. "e"
        as_xarray : bool, default False
            If True, return xarray DataArrays instead of pandas DataFrames

        Returns
        -------
        tuple[pd.DataFrame | DataArray, pd.DataFrame | DataArray]
            Tuple of (min_pu, max_pu) DataFrames or DataArrays.

        """
        min_pu = self.as_dynamic("e_min_pu", sns)
        max_pu = self.as_dynamic("e_max_pu", sns)

        if index is not None:
            min_pu = min_pu.reindex(columns=index)
            max_pu = max_pu.reindex(columns=index)

        if as_xarray:
            min_pu = xr.DataArray(min_pu)
            max_pu = xr.DataArray(max_pu)

        return min_pu, max_pu
