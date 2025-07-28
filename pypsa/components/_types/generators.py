"""Generators components module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import xarray as xr


@patch_add_docstring
class Generators(Components):
    """Generators components class.

    This class is used for generator components. All functionality specific to
    generators is implemented here. Functionality for all components is implemented in
    the abstract base class.

    See Also
    --------
    [pypsa.Components][] : Base class for all components.

    Examples
    --------
    >>> n.components.generators
    'Generator' Components
    ----------------------
    Attached to PyPSA Network 'AC-DC-Meshed'
    Components: 6

    """

    _operational_variables = ["p"]

    def get_bounds_pu(
        self,
        attr: str = "p",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for generators.

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "p"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = f"Bounds can only be retrieved for operational attributes. For generators those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        return self.da.p_min_pu, self.da.p_max_pu

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
