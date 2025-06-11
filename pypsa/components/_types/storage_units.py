"""Storage units components module."""

from collections.abc import Sequence
from typing import Any

import pandas as pd

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
