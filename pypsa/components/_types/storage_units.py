# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Storage units components module."""

from collections.abc import Sequence
from typing import Any

import pandas as pd
import xarray as xr

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components


@patch_add_docstring
class StorageUnits(Components):
    """StorageUnits components class.

    This class is used for storage unit components. All functionality specific to
    storage units is implemented here. Functionality for all components is implemented
    in the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.storage_units
    Empty 'StorageUnit' Components

    """

    _operational_variables = ["p_dispatch", "p_store", "state_of_charge"]

    def get_bounds_pu(
        self,
        attr: str = "p_store",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for storage units.

        <!-- md:badge-version v1.0.0 -->

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "p", "p_store", "state_of_charge"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = f"Bounds can only be retrieved for operational attributes. For storage_units those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        max_pu = self.da.p_max_pu

        if attr == "p_store":
            max_pu = -self.da.p_min_pu
            min_pu = xr.zeros_like(max_pu)
        elif attr == "state_of_charge":
            max_pu = self.da.max_hours
            min_pu = xr.zeros_like(max_pu)
        else:
            max_pu = self.da.p_max_pu
            min_pu = xr.zeros_like(max_pu)

        return min_pu, max_pu

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str = "",
        overwrite: bool = False,
        return_names: bool | None = None,
        **kwargs: Any,
    ) -> pd.Index | None:
        """Wrap Components.add() and docstring is patched via decorator."""
        return super().add(
            name=name,
            suffix=suffix,
            overwrite=overwrite,
            return_names=return_names,
            **kwargs,
        )
