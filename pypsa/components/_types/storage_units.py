"""Storage units components module."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import xarray as xr

from pypsa.components.components import Components


class StorageUnits(Components):
    """
    StorageUnits components class.

    This class is used for storage unit components. All functionality specific to
    storage units is implemented here. Functionality for all components is implemented
    in the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    See Also
    --------
    pypsa.components.abstract.Components : Base class for all components.

    """

    base_attr = "p"
    nominal_attr = "p_nom"

    def get_bounds_pu(
        self,
        sns: Sequence,
        index: pd.Index | None = None,
        attr: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Get per unit bounds for storage units.

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
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

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
