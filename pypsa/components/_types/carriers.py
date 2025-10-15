"""Carriers components module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

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
    [pypsa.Components][] : Base class for all components.

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

    def add_missing_carriers(
        self,
        assign_colors: bool = True,
        color_palette: str = "tab10",
        **kwargs: Any,
    ) -> pd.Index:
        """Add carriers that are used in the network but not yet defined.

        This function iterates over all components that have a carrier attribute,
        collects all unique carrier values, and adds any carriers that are not yet
        defined in the network. Colors can optionally be automatically assigned
        to the new carriers using the specified color palette.

        Parameters
        ----------
        assign_colors : bool, default True
            Whether to automatically assign colors to the new carriers. If False,
            carriers will be added without color values (unless provided in kwargs).
        color_palette : str, default "tab10"
            Matplotlib color palette to use for assigning colors to carriers.
            Only used if assign_colors=True. Options include:
            - "tab10" (10 colors, default)
            - "tab20" (20 colors)
            - "Set1", "Set2", "Set3" (qualitative palettes)
            - "Pastel1", "Pastel2" (soft colors)
            - Any other matplotlib colormap name
        **kwargs : Any
            Additional keyword arguments to pass to the add() method for the new
            carriers (e.g., co2_emissions, nice_name, color).

        Returns
        -------
        pd.Index
            Index of newly added carrier names.

        Examples
        --------
        Add missing carriers with default tab10 colors:

        >>> n.components.carriers.add_missing_carriers()
        Index(['wind', 'solar', 'gas'], dtype='object', name='name')

        Add missing carriers without automatic color assignment:

        >>> n.components.carriers.add_missing_carriers(assign_colors=False)
        Index(['wind', 'solar', 'gas'], dtype='object', name='name')

        Add missing carriers with a custom palette:

        >>> n.components.carriers.add_missing_carriers(color_palette="Set3")
        Index(['wind', 'solar', 'gas'], dtype='object', name='name')

        Add missing carriers with additional attributes:

        >>> n.components.carriers.add_missing_carriers(
        ...     color_palette="tab20",
        ...     co2_emissions=0.0
        ... )
        Index(['wind', 'solar'], dtype='object', name='name')

        Notes
        -----
        - Components checked for carrier attributes: generators, loads, storage_units,
          stores, links, lines, sub_networks
        - Empty strings, None, and NaN values are ignored
        - Colors are assigned deterministically based on alphabetically sorted carrier
          names
        - If more carriers exist than colors in the palette, colors will be cycled

        See Also
        --------
        add : Add carriers manually to the network

        """
        # Collect all unique carrier values from components
        all_carriers = set()
        for c in self.n.components:
            # Check if component type exists in this network
            if c is None or c.static.empty:
                continue
            if "carrier" not in c.static.columns:
                continue

            carriers = c.static["carrier"].dropna()
            # Filter out empty strings
            carriers = carriers[carriers != ""]
            all_carriers.update(carriers.unique())

        # Get existing carriers
        if self.n.has_scenarios:
            # For stochastic networks, extract carrier names from MultiIndex
            existing_carriers = set(self.static.index.get_level_values("name").unique())
        else:
            existing_carriers = set(self.static.index)

        # Find missing carriers
        missing_carriers = sorted(all_carriers - existing_carriers)

        if not missing_carriers:
            logger.info("No missing carriers found. All carriers are already defined.")
            return pd.Index([], name="name")

        # Generate colors for missing carriers if requested
        if assign_colors and "color" not in kwargs:
            colors = self._generate_colors(len(missing_carriers), color_palette)
            kwargs["color"] = colors

        # Add missing carriers
        logger.info(
            "Adding %d missing carriers: %s", len(missing_carriers), missing_carriers
        )

        # Add carriers normally first
        result = self.add(missing_carriers, return_names=True, **kwargs)

        # For stochastic networks, wrap the newly added carriers across all scenarios
        if self.n.has_scenarios:
            # Get only the newly added carriers
            new_carriers_df = self.static.loc[missing_carriers]

            # Remove them from static (they were added without scenarios)
            self.static = self.static.drop(missing_carriers)

            # Wrap them with scenarios using pd.concat (same as set_scenarios does)
            wrapped_df = pd.concat(
                dict.fromkeys(self.n.scenarios, new_carriers_df), names=["scenario"]
            )

            # Combine with existing carriers, preserving MultiIndex structure
            if self.static.empty:
                # If static was empty, just use the wrapped dataframe
                self.static = wrapped_df
            else:
                # Concatenate preserving the MultiIndex
                combined = pd.concat([self.static, wrapped_df], axis=0)
                # Ensure result is a proper MultiIndex, not an Index of tuples
                if not isinstance(combined.index, pd.MultiIndex):
                    # Reconstruct as MultiIndex
                    combined.index = pd.MultiIndex.from_tuples(
                        combined.index, names=["scenario", "name"]
                    )
                self.static = combined

        return result

    def _generate_colors(self, n_colors: int, palette: str = "tab10") -> list[str]:
        """Generate a list of colors from a matplotlib palette.

        Parameters
        ----------
        n_colors : int
            Number of colors to generate.
        palette : str, default "tab10"
            Matplotlib color palette name.

        Returns
        -------
        list of str
            List of hex color strings.

        """
        try:
            cmap = plt.get_cmap(palette)
        except ValueError:
            logger.warning(
                "Color palette '%s' not found. Using 'tab10' as fallback.", palette
            )
            cmap = plt.get_cmap("tab10")

        # Get the number of colors in the palette
        if hasattr(cmap, "N"):
            n_palette_colors = cmap.N
        else:
            # For continuous colormaps, use a reasonable number
            n_palette_colors = 256

        # Generate colors
        colors = []
        for i in range(n_colors):
            # Cycle through palette if we have more carriers than colors
            idx = i % n_palette_colors
            if n_palette_colors <= 20:
                # For discrete palettes, use integer indices
                rgba = cmap(idx)
            else:
                # For continuous palettes, normalize to [0, 1]
                rgba = cmap(idx / n_palette_colors)
            # Convert RGBA to hex
            colors.append(mcolors.to_hex(rgba))

        return colors
