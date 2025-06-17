"""Links components module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components
from pypsa.constants import RE_PORTS_GE_2

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import xarray as xr


@patch_add_docstring
class Links(Components):
    """Links components class.

    This class is used for link components. All functionality specific to
    links is implemented here. Functionality for all components is implemented in
    the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    See Also
    --------
    [pypsa.Components][] : Base class for all components.

    Examples
    --------
    >>> n.components.links
    'Link' Components
    -----------------
    Attached to PyPSA Network 'AC-DC'
    Components: 4

    """

    base_attr = "p"
    nominal_attr = "p_nom"

    def get_bounds_pu(
        self,
        sns: Sequence,
        index: pd.Index | None = None,
        attr: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for links.

        Parameters
        ----------
        sns : pandas.Index/pandas.DateTimeIndex
            Set of snapshots for the bounds
        index : pd.Index, optional
            Subset of the component elements
        attr : string, optional
            Attribute name for the bounds, e.g. "p"

        Returns
        -------
        tuple[pd.DataFrame | DataArray, pd.DataFrame | DataArray]
            Tuple of (min_pu, max_pu) DataFrames or DataArrays.

        """
        min_pu = self.as_xarray("p_min_pu", sns, inds=index)
        max_pu = self.as_xarray("p_max_pu", sns, inds=index)

        return min_pu, max_pu

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

    @property
    def additional_ports(self) -> list[str]:
        """Identify additional link ports (bus connections) beyond predefined ones.

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

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Link", "link1", bus0="bus1", bus1="bus2", bus2="bus3")  # doctest: +ELLIPSIS
        Index...
        >>> n.components.links.additional_ports
        ['2']

        """
        return [
            match.group(1)
            for col in self.static.columns
            if (match := RE_PORTS_GE_2.search(col))
        ]
