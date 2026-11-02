# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Storage units components module."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.common import list_as_string
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components


@patch_add_docstring
class StorageUnits(Components):
    """StorageUnits components class.

    This class is used for storage unit components. All functionality specific to
    storage units is implemented here. Functionality for all components is implemented
    in the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.storage_units
    Empty 'StorageUnit' Components

    """

    _operational_variables = ["p_dispatch", "p_store", "state_of_charge"]

    def get_bounds_pu(
        self,
        attr: str = "p_store",
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Get per unit bounds for storage units.

        <!-- md:badge-version v1.0.0 -->

        Parameters
        ----------
        attr : string, optional
            Attribute name for the bounds, e.g. "p", "p_store", "state_of_charge"

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Tuple of (min_pu, max_pu) DataArrays.

        """
        if attr not in self._operational_variables:
            msg = f"Bounds can only be retrieved for operational attributes. For storage_units those are: {list_as_string(self._operational_variables)}."
            raise ValueError(msg)

        max_pu = self.da.p_max_pu

        if attr == "p_store":
            max_pu = -self.da.p_min_pu
            min_pu = xr.zeros_like(max_pu)
        elif attr == "state_of_charge":
            max_pu = self.da.max_hours
            min_pu = xr.zeros_like(max_pu)
        else:
            max_pu = self.da.p_max_pu
            min_pu = xr.zeros_like(max_pu)

        return min_pu, max_pu

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

    def convert_to_stores(self, names: Sequence[str] | None = None) -> pd.Index:
        """Convert storage units into equivalent stores.

        Stores cover all functionality of storage units and both formulations
        yield identical optimisation results. Storage units whose conversion
        would change results raise a `ValueError`: those with `p_dispatch_set`,
        `p_store_set`, piecewise costs or `marginal_cost_quadratic`, which stores
        apply to the net dispatch instead of the discharge only.

        Parameters
        ----------
        names : Sequence[str] | None, optional
            Names of the storage units to convert. Defaults to all.

        Returns
        -------
        pd.Index
            Names of the added stores.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", "bus")
        >>> n.add("StorageUnit", "su", bus="bus", p_nom=10, max_hours=4, capital_cost=8)
        >>> n.c.storage_units.convert_to_stores()
        Index(['su'], dtype='str', name='name')
        >>> n.stores[["e_nom", "max_hours", "capital_cost"]]
              e_nom  max_hours  capital_cost
        name
        su     40.0        4.0           2.0

        """
        n = self.n_save
        if n.has_scenarios:
            msg = "Converting storage units is not supported for stochastic networks."
            raise NotImplementedError(msg)

        index = self.static.index if names is None else pd.Index(names)
        clash = index.intersection(n.c.stores.static.index)
        if not clash.empty:
            msg = f"Stores with names {clash.tolist()} already exist."
            raise ValueError(msg)

        max_hours = self.static.loc[index, "max_hours"]
        if not np.isfinite(max_hours).all():
            msg = "Storage units must have finite `max_hours` to be converted."
            raise ValueError(msg)

        piecewise = [
            attr
            for attr, df in self.piecewise.items()
            if not df.columns.get_level_values("name").intersection(index).empty
        ]
        if piecewise:
            msg = f"Storage units with piecewise {piecewise} cannot be converted to stores."
            raise ValueError(msg)

        mc_quadratic = n.get_switchable_as_dense(
            "StorageUnit", "marginal_cost_quadratic", inds=index
        )
        if (mc_quadratic != 0).any().any():
            msg = (
                "Storage units with `marginal_cost_quadratic` cannot be converted to "
                "stores, since stores apply it to the net dispatch."
            )
            raise ValueError(msg)

        renamed = {
            "cyclic_state_of_charge": "e_cyclic",
            "cyclic_state_of_charge_per_period": "e_cyclic_per_period",
            "marginal_cost": "marginal_cost_dispatch",
        }
        factors = {
            **dict.fromkeys(
                ["p_nom", "p_nom_min", "p_nom_max", "p_nom_set", "p_nom_mod"], max_hours
            ),
            **dict.fromkeys(
                ["capital_cost", "overnight_cost", "fom_cost"], 1 / max_hours
            ),
        }
        outputs = self.defaults.index[self.defaults.status.str.startswith("Output")]
        kwargs: dict[str, Any] = {}
        for attr in self.static.columns.difference(outputs):
            store_attr = renamed.get(
                attr, attr.replace("p_nom", "e_nom").replace("state_of_charge", "e")
            )
            varying = (
                attr in self.dynamic
                and not self.dynamic[attr].columns.intersection(index).empty
            )
            if (
                attr in self.defaults.index
                and store_attr not in n.c.stores.defaults.index
            ):
                if varying or self.static.loc[index, attr].notna().any():
                    msg = f"Storage units with `{attr}` cannot be converted to stores."
                    raise ValueError(msg)
                continue
            if varying:
                kwargs[store_attr] = n.get_switchable_as_dense(
                    "StorageUnit", attr, inds=index
                )
            else:
                kwargs[store_attr] = self.static.loc[index, attr] * factors.get(attr, 1)

        n.remove("StorageUnit", index)
        return n.add("Store", index, return_names=True, **kwargs)
