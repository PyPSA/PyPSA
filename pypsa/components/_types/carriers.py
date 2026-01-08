# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Carriers components module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.common import generate_colors
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


@patch_add_docstring
class Carriers(Components):
    """Carriers components class.

    This class is used for carrier components. All functionality specific to
    carriers is implemented here. Functionality for all components is implemented in
    the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.carriers
    'Carrier' Components
    --------------------
    Attached to PyPSA Network 'AC-DC-Meshed'
    Components: 6

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

    def assign_colors(
        self,
        carriers: str | Sequence[str] | None = None,
        palette: str = "tab10",
        overwrite: bool = False,
    ) -> None:
        """Assign colors to carriers using a matplotlib color palette.

        <!-- md:badge-version v1.1.0 -->

        Parameters
        ----------
        carriers : str, Sequence[str], or None, default None
            Carrier name(s) to assign colors to. If None, assigns colors to all
            carriers that don't have a color set (or all if overwrite=True).
        palette : str, default "tab10"
            Matplotlib color palette to use for assigning colors. Options include:
            - "tab10" (10 colors, default)
            - "tab20" (20 colors)
            - "Set1", "Set2", "Set3" (qualitative palettes)
            - "Pastel1", "Pastel2" (soft colors)
            - Any other matplotlib colormap name
        overwrite : bool, default False
            If True, overwrite existing colors. If False, only assign colors to
            carriers that don't have a color set (empty string or NaN).

        See Also
        --------
        add_missing_carriers : Add carriers that are used but not yet defined

        """
        # Determine which carriers to color
        if carriers is None:
            if overwrite:
                target_carriers = self.names
            else:
                # Only carriers without colors
                mask = (self.static["color"] == "") | self.static["color"].isna()
                filtered_static = self.static[mask]
                if self.n_save.has_scenarios:
                    target_carriers = filtered_static.index.get_level_values(
                        "name"
                    ).unique()
                else:
                    target_carriers = filtered_static.index
        else:
            if isinstance(carriers, str):
                carriers = [carriers]
            target_carriers = pd.Index(carriers)

            unknown = target_carriers.difference(self.names)
            if len(unknown) > 0:
                msg = f"Cannot assign colors to unknown carriers: {list(unknown)}"
                raise ValueError(msg)

        if len(target_carriers) == 0:
            logger.debug("No carriers to assign colors to.")
            return

        # Sort carriers for deterministic color assignment
        target_carriers = sorted(target_carriers)

        colors = generate_colors(len(target_carriers), palette)

        for carrier, color in zip(target_carriers, colors, strict=False):
            # Update color for all scenarios
            if self.n_save.has_scenarios:
                for scenario in self.n_save.scenarios:
                    self.static.loc[(scenario, carrier), "color"] = color
            else:
                self.static.loc[carrier, "color"] = color

        logger.info(
            "Assigned colors to %d carriers using '%s' palette.",
            len(target_carriers),
            palette,
        )

    def add_missing_carriers(
        self,
        **kwargs: Any,
    ) -> pd.Index:
        """Add carriers that are used in the network but not yet defined.

        <!-- md:badge-version v1.1.0 -->

        This function iterates over all components that have a carrier attribute,
        collects all unique carrier values, and adds any carriers that are not yet
        defined in the network.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments to pass to the add() method for the new
            carriers (e.g., color, co2_emissions, nice_name).

        Returns
        -------
        pd.Index
            Index of newly added carrier names.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.components.buses.add('my_bus', carrier='my_carrier')
        >>> n.c.carriers.add_missing_carriers()
        Index(['my_carrier'], dtype='object')

        Carriers are added without needing to call `n.add` separately:

        >>> n.components.carriers.static  # doctest: +ELLIPSIS
                    co2_emissions color nice_name  max_growth  max_relative_growth
        name
        my_carrier            0.0                         inf                  0.0

        """
        # Get all unique carrier values
        all_carriers = set()
        for c in self.n_save.c.values():
            all_carriers.update(c.unique_carriers)

        existing_carriers = set(self.names)
        missing_carriers = sorted(all_carriers - existing_carriers)

        if not missing_carriers:
            logger.debug("No missing carriers found. All carriers are already defined.")
            return pd.Index([], name="name")

        logger.info(
            "Adding %d missing carriers: %s", len(missing_carriers), missing_carriers
        )

        result = self.add(missing_carriers, return_names=True, **kwargs)

        return result
