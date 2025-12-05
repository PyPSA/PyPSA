# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Buses components module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components
from pypsa.constants import RE_PORTS_FILTER

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@patch_add_docstring
class Buses(Components):
    """Buses components class.

    This class is used for bus components. All functionality specific to
    buses is implemented here. Functionality for all components is implemented in
    the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.buses
    'Bus' Components
    ----------------
    Attached to PyPSA Network 'AC-DC-Meshed'
    Components: 9

    """

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

    def add_missing_buses(self, **kwargs: Any) -> pd.Index:
        """Add buses that are referenced by components but not yet defined.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments for new buses (e.g., v_nom, carrier).

        Returns
        -------
        pd.Index
            Index of newly added bus names.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.components.generators.add("my_gen", bus="my_bus")
        >>> n.components.buses.add_missing_buses(v_nom=100.0)
        Index(['my_bus'], dtype='object')

        Buses are added without the need to extra call `n.add`:

        >>> n.components.buses.static  # doctest: +ELLIPSIS
                v_nom type    x    y  ...
        name                          ...
        my_bus  100.0       0.0  0.0  ...
        ...

        """
        # Collect all unknown buses from all components
        all_buses = set()
        for c in self.n_save.c.values():
            if c.static.empty:
                continue
            # Get bus columns for this component
            bus_cols = c.static.columns[c.static.columns.str.contains(RE_PORTS_FILTER)]
            for attr in bus_cols:
                buses = c.static[attr].astype(str)
                # Filter empty strings for branch components (bus2, bus3 can be empty)
                if c.name in self.n_save.branch_components and int(attr[-1]) > 1:
                    buses = buses[buses != ""]
                # Filter empty strings for global constraints
                if c.name == "GlobalConstraint":
                    buses = buses[buses != ""]
                # Find missing buses
                missing = ~buses.isin(self.n_save.c.buses.names)
                all_buses.update(buses[missing & (buses != "") & (buses != "nan")])

        missing_buses = sorted(all_buses)

        if not missing_buses:
            logger.debug("No missing buses found.")
            return pd.Index([], name="name")

        logger.info("Adding %d missing buses: %s", len(missing_buses), missing_buses)
        result = self.add(missing_buses, return_names=True, **kwargs)

        return result
