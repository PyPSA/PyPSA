"""Lines components module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray

from pypsa.components.components import Components
from pypsa.geo import haversine_pts


class Lines(Components):
    """
    Lines components class.

    This class is used for line components. All functionality specific to
    lines is implemented here. Functionality for all components is implemented in
    the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    See Also
    --------
    pypsa.components.abstract.Components : Base class for all components.

    """

    base_attr = "s"
    nominal_attr = "s_nom"

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
    ) -> tuple[pd.DataFrame | DataArray, pd.DataFrame | DataArray]:
        """
        Get per unit bounds for lines.

        For passive branch components, min_pu is the negative of max_pu.

        Parameters
        ----------
        sns : pandas.Index/pandas.DateTimeIndex
            Set of snapshots for the bounds
        index : pd.Index, optional
            Subset of the component elements
        attr : string, optional
            Attribute name for the bounds, e.g. "s"
        as_xarray : bool, default False
            If True, return xarray DataArrays instead of pandas DataFrames

        Returns
        -------
        tuple[pd.DataFrame | DataArray, pd.DataFrame | DataArray]
            Tuple of (min_pu, max_pu) DataFrames or DataArrays.

        """
        max_pu = self.as_xarray("s_max_pu", sns, inds=index)
        min_pu = -max_pu  # Lines specific: min_pu is the negative of max_pu

        if not as_xarray:
            max_pu = max_pu.to_dataframe()
            min_pu = min_pu.to_dataframe()

        return min_pu, max_pu

    def calculate_line_length(self) -> pd.Series:
        """
        Get length of the lines in meters.

        Based on coordinates of attached buses. Buses must have 'x' and 'y' attributes,
        otherwise no line length can be calculated. By default the haversine formula is
        used to calculate the distance between two points.

        Returns
        -------
        pd.Series
            Length of the lines.

        See Also
        --------
        pypsa.geo.haversine : Function to calculate distance between two points.

        Examples
        --------
        >>> c = pypsa.examples.scigrid_de().c.lines
        >>> ds = c.calculate_line_length()
        >>> ds.head()
        0    34432.796096
        1    59701.666027
        2    32242.741010
        3    30559.154647
        4    21574.543367
        dtype: float64

        """
        return (
            pd.Series(
                haversine_pts(
                    a=np.array(
                        [
                            self.static.bus0.map(self.n_save.buses.x),
                            self.static.bus0.map(self.n_save.buses.y),
                        ]
                    ).T,
                    b=np.array(
                        [
                            self.static.bus1.map(self.n_save.buses.x),
                            self.static.bus1.map(self.n_save.buses.y),
                        ]
                    ).T,
                )
            )
            * 1_000
        )
