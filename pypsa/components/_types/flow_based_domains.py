# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Flow-based domain components module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


@patch_add_docstring
class FlowBasedDomains(Components):
    """Flow-based domain components class.

    A non-physical component holding a flow-based market-coupling domain: a set of
    linear constraints on the net positions of the market zones (buses), of the form
    ``PTDF . NP <= RAM``. Each entity is one critical network element (CNEC). The zonal
    PTDF sensitivities are stored as extra columns named by bus in the static frame; the
    RAM is a (possibly time-varying) attribute. See the abstract base class for
    functionality shared across all components.

    See Also
    --------
    [pypsa.Components][]

    """

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str | Sequence[str] = "",
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
