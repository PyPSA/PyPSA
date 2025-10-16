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

        Examples
        --------
        Assign colors to all carriers without colors:

        >>> n.c.carriers.assign_colors()

        Assign colors using a different palette:

        >>> n.c.carriers.assign_colors(palette="Set3")

        Assign colors to specific carriers:

        >>> n.c.carriers.assign_colors(["wind", "solar"], palette="tab20")

        Overwrite existing colors:

        >>> n.c.carriers.assign_colors(overwrite=True)

        See Also
        --------
        add_missing_carriers : Add carriers that are used but not yet defined

        """
        # Determine which carriers to color
        if carriers is None:
            if overwrite:
                target_carriers = (
                    self.static.index.get_level_values("name").unique()
                    if self.n.has_scenarios
                    else self.static.index
                )
            else:
                # Only carriers without colors
                mask = (self.static["color"] == "") | self.static["color"].isna()
                if self.n.has_scenarios:
                    target_carriers = (
                        self.static[mask].index.get_level_values("name").unique()
                    )
                else:
                    target_carriers = self.static[mask].index
        else:
            if isinstance(carriers, str):
                carriers = [carriers]
            target_carriers = pd.Index(carriers)

        if len(target_carriers) == 0:
            logger.debug("No carriers to assign colors to.")
            return

        # Sort carriers for deterministic color assignment
        target_carriers = sorted(target_carriers)

        # Generate colors
        colors = generate_colors(len(target_carriers), palette)

        # Assign colors
        for carrier, color in zip(target_carriers, colors, strict=False):
            if self.n.has_scenarios:
                # Update color for all scenarios
                for scenario in self.n.scenarios:
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
        palette: str | None = "tab10",
        **kwargs: Any,
    ) -> pd.Index:
        """Add carriers that are used in the network but not yet defined.

        This function iterates over all components that have a carrier attribute,
        collects all unique carrier values, and adds any carriers that are not yet
        defined in the network. Colors are automatically assigned to the new
        carriers using the specified color palette unless disabled.

        Parameters
        ----------
        palette : str or None, default "tab10"
            Matplotlib color palette to use for assigning colors to carriers.
            If None, no colors are assigned automatically. Options include:
            - "tab10" (10 colors, default)
            - "tab20" (20 colors)
            - "Set1", "Set2", "Set3" (qualitative palettes)
            - "Pastel1", "Pastel2" (soft colors)
            - Any other matplotlib colormap name
            - None (no automatic color assignment)
        **kwargs : Any
            Additional keyword arguments to pass to the add() method for the new
            carriers (e.g., co2_emissions, nice_name). If 'color' is provided,
            it will override the automatic color assignment from the palette.

        Returns
        -------
        pd.Index
            Index of newly added carrier names.

        Raises
        ------
        ValueError
            If both palette and color are provided in kwargs.

        Examples
        --------
        Add missing carriers with default tab10 colors:

        >>> n.c.carriers.add_missing_carriers()
        Index(['AC', 'gas', 'solar', 'wind'], dtype='object', name='name')

        Add missing carriers without automatic color assignment:

        >>> n.c.carriers.add_missing_carriers(palette=None)
        Index(['AC', 'gas', 'solar', 'wind'], dtype='object', name='name')

        Add missing carriers with a custom palette:

        >>> n.c.carriers.add_missing_carriers(palette="Set3")
        Index(['AC', 'gas', 'solar', 'wind'], dtype='object', name='name')

        Add missing carriers with additional attributes:

        >>> n.c.carriers.add_missing_carriers(
        ...     palette="tab20",
        ...     co2_emissions=0.0
        ... )
        Index(['AC', 'solar', 'wind'], dtype='object', name='name')

        Add missing carriers with explicit colors:

        >>> n.c.carriers.add_missing_carriers(
        ...     color=["red", "blue", "green"]
        ... )
        Index(['AC', 'gas', 'solar'], dtype='object', name='name')

        Notes
        -----
        - Components checked for carrier attributes: generators, loads, storage_units,
          stores, links, lines, sub_networks
        - Empty strings, None, and NaN values are ignored
        - Colors are assigned deterministically based on alphabetically sorted carrier
          names
        - If more carriers exist than colors in the palette, colors will be cycled
        - If you provide explicit colors via kwargs, they will override palette colors

        See Also
        --------
        add : Add carriers manually to the network
        assign_colors : Assign colors to existing carriers

        """
        # Check for conflicting arguments
        if palette is not None and "color" in kwargs:
            msg = "Cannot specify both 'palette' and 'color' in kwargs. Use either palette for automatic assignment or color for explicit values."
            raise ValueError(msg)

        # Collect all unique carrier values from components
        all_carriers = set()
        for c in self.n_save.c.values():
            all_carriers.update(c.unique_carriers)

        # Get existing carriers
        if self.n.has_scenarios:
            # For stochastic networks, extract carrier names from MultiIndex
            existing_carriers = set(self.static.index.get_level_values("name").unique())
        else:
            existing_carriers = set(self.static.index)

        # Find missing carriers
        missing_carriers = sorted(all_carriers - existing_carriers)

        if not missing_carriers:
            logger.debug("No missing carriers found. All carriers are already defined.")
            return pd.Index([], name="name")

        # Add missing carriers
        logger.info(
            "Adding %d missing carriers: %s", len(missing_carriers), missing_carriers
        )

        # For stochastic networks, add carriers with scenario wrapping
        if self.n.has_scenarios:
            # Create a temporary network to leverage the standard add() method
            temp_static = self.static.copy()
            self.static = self.static.iloc[:0]  # Temporarily clear for clean add

            # Add carriers normally to get proper defaults
            self.add(missing_carriers, **kwargs)
            new_carriers_df = self.static.copy()

            # Restore original static
            self.static = temp_static

            # Wrap new carriers across all scenarios
            wrapped_df = pd.concat(
                dict.fromkeys(self.n.scenarios, new_carriers_df), names=["scenario"]
            )

            # Append to existing carriers
            self.static = pd.concat([self.static, wrapped_df], axis=0)

            # Assign colors if palette is specified and colors not explicitly provided
            if palette is not None and "color" not in kwargs:
                self.assign_colors(
                    carriers=missing_carriers, palette=palette, overwrite=False
                )

            return pd.Index(missing_carriers, name="name")

        # For non-stochastic networks, add carriers
        result = self.add(missing_carriers, return_names=True, **kwargs)

        # Assign colors if palette is specified and colors not explicitly provided
        if palette is not None and "color" not in kwargs:
            self.assign_colors(
                carriers=missing_carriers, palette=palette, overwrite=False
            )

        return result
