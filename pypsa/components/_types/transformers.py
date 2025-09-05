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
class Transformers(Components):
    """Transformers components class.

    This class is used for transformer components. All functionality specific to
    transformers is implemented here. Functionality for all components is implemented in
    the abstract base class.

    See Also
    --------
    [pypsa.Components][] : Base class for all components.

    Examples
    --------
    >>> n.components.transformers
    Empty 'Transformer' Components

    """

    _operational_variables = ["s"]

    def get_bounds_pu(
        self,
        attr: str = "s",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for transformers.

        For passive branch components, min_pu is the negative of max_pu.

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "s"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = f"Bounds can only be retrieved for operational attributes. For transformers those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        max_pu = self.da.s_max_pu
        min_pu = -max_pu  # Transformers specific: min_pu is the negative of max_pu

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
