"""Links components module."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


@patch_add_docstring
class Links(Components):
    """
    Links components class.

    This class is used for link components. All functionality specific to
    links is implemented here. Functionality for all components is implemented in
    the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    See Also
    --------
    pypsa.components.abstract.Components : Base class for all components.

    """

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str = "",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> pd.Index:
        """Wrapper for Components.add() and docstring is patched via decorator."""
        return super().add(
            name=name,
            suffix=suffix,
            overwrite=overwrite,
            **kwargs,
        )

    @property
    def additional_ports(self) -> list[str]:
        """
        Identify additional link ports (bus connections) beyond predefined ones.

        Parameters
        ----------
        n : pypsa.Network
            Network instance.
        where : iterable of strings, default None
            Subset of columns to consider. Takes link columns by default.

        Returns
        -------
        list of strings
            List of additional link ports. E.g. ["2", "3"] for bus2, bus3.

        Also see
        ---------
        pypsa.Components.ports

        """
        return [match.group(1) for col in self.static.columns if (match := re.search(r"^bus([2-9]\d*)$", col))]
