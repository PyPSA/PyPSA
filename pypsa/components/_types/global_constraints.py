# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Global constraints components module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

if TYPE_CHECKING:
    from collections.abc import Sequence

PTDF_PREFIX = "ptdf_"


@patch_add_docstring
class GlobalConstraints(Components):
    """Global constraints components class.

    This class is used for global constraint components. All functionality specific to
    global constraints is implemented here. Functionality for all components is implemented in
    the abstract base class.

    See Also
    --------
    [pypsa.Components][]

    Examples
    --------
    >>> n.components.global_constraints
    'GlobalConstraint' Components
    -----------------------------
    Attached to PyPSA Network 'AC-DC-Meshed'
    Components: 1

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

    def _ptdf_columns(self) -> pd.Index:
        """Names of the zonal PTDF columns, without the `ptdf_` prefix.

        Static columns come first, then columns that only exist as dynamic data.
        """
        static = [c for c in self.static.columns if str(c).startswith(PTDF_PREFIX)]
        dynamic = [
            k
            for k, df in self.dynamic.items()
            if k.startswith(PTDF_PREFIX) and k not in static and not df.columns.empty
        ]
        return pd.Index([c[len(PTDF_PREFIX) :] for c in static + dynamic], dtype=object)

    def _flow_based_names(self) -> pd.Index:
        """Names of the `flow_based` global constraints."""
        static = self.static
        return static.index[static["type"] == "flow_based"].unique("name")

    def add_flow_based(
        self,
        zonal_ptdf: pd.DataFrame | pd.Series,
        ram: float | pd.Series | pd.DataFrame,
        investment_period: int | None = None,
        overwrite: bool = False,
    ) -> None:
        """Add a flow-based domain, one `flow_based` global constraint per CNEC.

        Each critical network element with contingency (CNEC) becomes a row with
        `type="flow_based"`, `sense="<="` and the right-hand side `ram`. The zonal
        PTDF is stored in the columns `ptdf_<name>`, where `<name>` is a bus (a
        bidding zone) or a Link (a corridor whose flow `p` enters the domain).
        In the optimisation, the domain constrains the zonal net positions:
        `sum_z PTDF[c, z] * NP[z] + sum_l PTDF[c, l] * p[l] <= ram[c]`.

        Parameters
        ----------
        zonal_ptdf : pandas.DataFrame or pandas.Series
            Sensitivities with one column per zone or link. A DataFrame indexed by
            CNEC is a static domain. A DataFrame with a `(snapshot, CNEC)`
            MultiIndex is time-varying, its leading level(s) must equal
            `n.snapshots`. A Series is a single CNEC named by its `name`.
        ram : float, pandas.Series or pandas.DataFrame
            Remaining available margin. A scalar applies to every CNEC, a Series
            is indexed by CNEC and a DataFrame is `snapshot x CNEC`.
        investment_period : int, optional
            Investment period the domain binds in. By default it binds in all
            snapshots.
        overwrite : bool, default False
            Replace existing global constraints of the same names.

        Notes
        -----
        `sense` must be `<=` and `constant` is ignored for `flow_based` rows. A link
        column carries `h - (eta * PTDF_bus1 - PTDF_bus0)`, where `h` is the
        CNEC's sensitivity to the link flow, since the net position of a zone
        includes the flows of its links. On a stochastic network this writes the
        same domain to every scenario. A scenario-specific domain is set
        afterwards by editing `ptdf_<name>` or `ram` per scenario in the static
        or dynamic data.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", ["A", "B"])
        >>> ptdf = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=["AB (+)"])
        >>> n.c.global_constraints.add_flow_based(ptdf, ram=1000)
        >>> n.c.global_constraints.zonal_ptdf
        bus       A    B
        name
        AB (+)  0.5 -0.5

        """
        if isinstance(zonal_ptdf, pd.Series):
            if not isinstance(zonal_ptdf.name, str) or not zonal_ptdf.name:
                msg = "A Series `zonal_ptdf` needs a non-empty string `name`, the CNEC."
                raise ValueError(msg)
            zonal_ptdf = zonal_ptdf.to_frame().T

        sns = self.n_save.snapshots
        if isinstance(zonal_ptdf.index, pd.MultiIndex):
            cnecs = zonal_ptdf.index.unique(-1)
            leading = zonal_ptdf.index.droplevel(-1).unique()
            if not leading.equals(sns):
                msg = (
                    "The leading index level(s) of a time-varying `zonal_ptdf` "
                    "must equal `n.snapshots`."
                )
                raise ValueError(msg)
            ptdf = {
                f"{PTDF_PREFIX}{col}": zonal_ptdf[col]
                .unstack(-1)
                .reindex(index=sns, columns=cnecs)
                for col in zonal_ptdf.columns
            }
        else:
            cnecs = zonal_ptdf.index
            ptdf = {f"{PTDF_PREFIX}{col}": zonal_ptdf[col] for col in zonal_ptdf}

        if isinstance(ram, pd.Series | pd.DataFrame):
            labels = ram.index if isinstance(ram, pd.Series) else ram.columns
            missing = cnecs.difference(labels)
            if not missing.empty:
                msg = f"`ram` misses the CNECs {missing.tolist()}."
                raise ValueError(msg)
            if isinstance(ram, pd.Series):
                ram = ram.reindex(cnecs)
            else:
                if not ram.index.equals(sns):
                    msg = "The index of a `ram` DataFrame must equal `n.snapshots`."
                    raise ValueError(msg)
                ram = ram.reindex(columns=cnecs)

        kwargs: dict[str, Any] = {}
        if investment_period is not None:
            kwargs["investment_period"] = investment_period
        self.add(
            list(cnecs),
            type="flow_based",
            sense="<=",
            ram=ram,
            overwrite=overwrite,
            **kwargs,
            **ptdf,
        )

    def _as_xarray(self, attr: str) -> xr.DataArray:
        """Get an attribute as a DataArray, including the derived `zonal_ptdf`.

        `zonal_ptdf` stacks the `ptdf_<name>` columns of the `flow_based` rows
        along a `bus` dim, with 0 where a row has no value.
        """
        if attr != "zonal_ptdf":
            return super()._as_xarray(attr)

        names = self._flow_based_names()
        columns = self._ptdf_columns()
        if names.empty or columns.empty:
            return xr.DataArray(
                np.zeros((len(names), 0)),
                coords={"name": names, "bus": pd.Index([], dtype=object)},
                dims=("name", "bus"),
                name="zonal_ptdf",
            )

        # Bind outside the comprehension: zero-arg super() fails inside one on 3.11
        as_xarray = super()._as_xarray
        arrays = xr.broadcast(*[as_xarray(f"{PTDF_PREFIX}{c}") for c in columns])
        res = xr.concat(arrays, dim=pd.Index(columns, name="bus"), coords="minimal")
        res = res.sel(name=names).fillna(0.0).astype(float)
        lead = [d for d in res.dims if d not in ("name", "bus")]
        return res.transpose(*lead, "name", "bus").rename("zonal_ptdf")

    @property
    def zonal_ptdf(self) -> pd.DataFrame:
        """Zonal PTDF of the `flow_based` rows, one column per zone or link.

        Indexed by CNEC for a static domain and by `(snapshot, CNEC)` for a
        time-varying one, with a leading `scenario` level on a stochastic
        network. See `add_flow_based` for an example.
        """
        da = self._as_xarray("zonal_ptdf")
        if da.sizes["bus"] == 0:
            return pd.DataFrame(index=da.indexes["name"], columns=[], dtype=float)
        return (
            da.to_series().unstack("bus", sort=False).reindex(columns=da.indexes["bus"])
        )
