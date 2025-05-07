"""Generators components module."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import xarray as xr

from pypsa.components.components import Components


class Generators(Components):
    """
    Generators components class.

    This class is used for generator components. All functionality specific to
    generators is implemented here. Functionality for all components is implemented in
    the abstract base class.

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
        Get per unit bounds for generators.

        Parameters
        ----------
        sns : pandas.Index/pandas.DateTimeIndex
            Set of snapshots for the bounds
        index : pd.Index, optional
            Subset of the component elements
        attr : string, optional
            Attribute name for the bounds, e.g. "p"

        Returns
        -------
        tuple[pd.DataFrame | DataArray, pd.DataFrame | DataArray]
            Tuple of (min_pu, max_pu) DataFrames or DataArrays.

        """
        min_pu = self.as_xarray("p_min_pu", sns, inds=index)
        max_pu = self.as_xarray("p_max_pu", sns, inds=index)

        return min_pu, max_pu
