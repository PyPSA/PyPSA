"""Storage units components module."""

from collections.abc import Sequence
from typing import Any

import pandas as pd
import xarray as xr

from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components


@patch_add_docstring
class StorageUnits(Components):
    """StorageUnits components class.

    This class is used for storage unit components. All functionality specific to
    storage units is implemented here. Functionality for all components is implemented
    in the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    See Also
    --------
    [pypsa.Components][] : Base class for all components.

    Examples
    --------
    >>> n.components.storage_units
    Empty 'StorageUnit' Components

    """

    base_attr = "p"
    nominal_attr = "p_nom"

    def get_bounds_pu(
        self,
        sns: Sequence,
        index: pd.Index | None = None,
        attr: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for storage units.

        Parameters
        ----------
        sns : pandas.Index/pandas.DateTimeIndex
            Set of snapshots for the bounds
        index : pd.Index, optional
            Subset of the component elements
        attr : string, optional
            Attribute name for the bounds, e.g. "p", "p_store", "state_of_charge"

        Returns
        -------
        tuple[pd.DataFrame | DataArray, pd.DataFrame | DataArray]
            Tuple of (min_pu, max_pu) DataFrames or DataArrays.

        """
        max_pu = self.as_xarray("p_max_pu", sns, inds=index)

        if attr == "p_store":
            max_pu = -self.as_xarray("p_min_pu", snapshots=sns, inds=index)
            min_pu = xr.zeros_like(max_pu)
        elif attr == "state_of_charge":
            max_pu = self.as_xarray("max_hours", snapshots=sns, inds=index)
            min_pu = xr.zeros_like(max_pu)
        else:
            max_pu = self.as_xarray("p_max_pu", snapshots=sns, inds=index)
            min_pu = xr.zeros_like(max_pu)

        return min_pu, max_pu

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str = "",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> pd.Index:
        """Wrap Components.add() and docstring is patched via decorator."""
        return super().add(
            name=name,
            suffix=suffix,
            overwrite=overwrite,
            **kwargs,
        )
