# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Flow-based domain components module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

from pypsa.common import check_optional_dependency
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

_EXCEL_HINT = (
    "Missing optional dependencies to read Excel files. Install them via "
    "`pip install pypsa[excel]`."
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network
    from pypsa.components.types import ComponentType


@patch_add_docstring
class FlowBasedDomains(Components):
    """Flow-based domain components class.

    A non-physical component holding a flow-based market-coupling domain: a set of
    linear constraints on the net positions of the market zones (buses), of the form
    ``zonal_ptdf . NP <= RAM``. Each entity is one critical network element (CNEC).

    Unlike the scalar attributes (``ram``, ``mu_domain``, ...), the zonal PTDF
    sensitivities form a matrix (cnec x zone) and are stored in a dedicated frame
    ``c.zonal_ptdf`` (rows = CNECs, columns = zone buses), analogous to ``c.piecewise``.
    The name distinguishes it from the *nodal* PTDF computed per sub-network. Pass it
    directly to ``add`` via the ``zonal_ptdf`` argument; read it back as a pandas
    DataFrame from ``c.zonal_ptdf`` or as an xarray DataArray from ``c.da.zonal_ptdf``.

    See Also
    --------
    [pypsa.Components][]

    """

    frame_attrs: tuple[str, ...] = ("zonal_ptdf",)

    def __init__(
        self,
        ctype: ComponentType,
        n: Network | None = None,
        names: str | int | Sequence[int | str] | None = None,
        suffix: str = "",
    ) -> None:
        """Initialise the component and its (empty) zonal PTDF frame."""
        super().__init__(ctype=ctype, n=n, names=names, suffix=suffix)
        self._zonal_ptdf = pd.DataFrame()

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

    def from_eraa(
        self,
        path: str,
        year: str | int,
        season: str,
        *,
        buses: dict[str, str] | None = None,
        ptdf_sheet: str | None = None,
        ram_sheet: str | None = None,
    ) -> pd.Index | None:
        """Add a flow-based domain from an ERAA ``FB-Domain-CORE`` Excel workbook.

        The zonal PTDF sheet carries two header rows: a *kind* row (``PTDF_SZ`` for the
        study-zone sensitivities) and a *label* row (the zone names plus ``FB_ID`` and
        ``CNEC_ID``). Only the ``PTDF_SZ`` columns are read here; AHC/EvFB columns are
        ignored for now. The domain is assumed time-invariant: one ``season`` is selected.

        Parameters
        ----------
        path : str
            Path to the ERAA workbook (requires the ``openpyxl`` extra to read ``.xlsx``).
        year : str or int
            Target year, selecting the ``PTDF {year}`` and ``RAM {year}`` sheets.
        season : str
            Seasonal domain to select (the ``FB_ID`` value and the RAM column, e.g.
            ``"winter1"``).
        buses : dict, optional
            Explicit mapping from ERAA zone labels to network bus names. Labels are used
            as-is where unmapped; no fuzzy matching is done.
        ptdf_sheet, ram_sheet : str, optional
            Override the sheet names (default ``"PTDF {year}"`` / ``"RAM {year}"``).

        Returns
        -------
        pandas.Index or None
            Names of the added CNECs (see [`add`][pypsa.Network.add]).

        """
        check_optional_dependency("openpyxl", _EXCEL_HINT)
        ptdf_sheet = ptdf_sheet or f"PTDF {year}"
        ram_sheet = ram_sheet or f"RAM {year}"

        raw = pd.read_excel(path, sheet_name=ptdf_sheet, header=None)
        kind, label = raw.iloc[0], raw.iloc[1]
        zones = label[kind == "PTDF_SZ"].tolist()
        body = raw.iloc[2:].copy()
        body.columns = list(label)
        rows = body[body["FB_ID"] == season].set_index("CNEC_ID")

        ram = pd.read_excel(path, sheet_name=ram_sheet).set_index("CNEC_ID")[season]
        keep = rows.index[rows.index.isin(ram.dropna().index)]

        return self._add_domain_frame(rows.loc[keep, zones], ram.loc[keep], buses)

    def from_jao(
        self,
        path: str,
        *,
        presolved: bool = True,
        buses: dict[str, str] | None = None,
        name_col: str = "Id",
        sep: str = ";",
    ) -> pd.Index | None:
        """Add a flow-based domain from a JAO ``finalComputation`` CSV.

        The zonal PTDF is given directly as ``Ptdf_<hub>`` columns; the ``Ptdf_`` prefix
        is stripped to obtain the hub (bus) name. The ``Direction`` (DIRECT/OPPOSITE) is
        already baked into the PTDF sign, so no sign duplication is applied. The domain is
        assumed time-invariant (one CSV = one market hour).

        Parameters
        ----------
        path : str
            Path to the JAO ``finalComputation`` CSV.
        presolved : bool, default True
            Keep only the presolved rows (the actual domain); if False, all rows are read.
        buses : dict, optional
            Explicit mapping from hub names (after stripping ``Ptdf_``) to network bus
            names. Hubs are used as-is where unmapped; no fuzzy matching is done.
        name_col : str, default "Id"
            Column used as the unique CNEC name. ``CneName`` is not unique across
            directions and contingencies, so the numeric ``Id`` is the default.
        sep : str, default ";"
            CSV field separator.

        Returns
        -------
        pandas.Index or None
            Names of the added CNECs (see [`add`][pypsa.Network.add]).

        """
        raw = pd.read_csv(path, sep=sep, low_memory=False)
        rows = (raw[raw["Presolved"]] if presolved else raw).copy()
        rows[name_col] = rows[name_col].astype(str)
        rows = rows.set_index(name_col)

        ptdf_cols = [c for c in rows.columns if c.startswith("Ptdf_")]
        zonal_ptdf = rows[ptdf_cols].rename(columns=lambda c: c.removeprefix("Ptdf_"))

        return self._add_domain_frame(zonal_ptdf, rows["Ram"], buses)

    def _add_domain_frame(
        self,
        zonal_ptdf: pd.DataFrame,
        ram: pd.Series,
        buses: dict[str, str] | None,
    ) -> pd.Index | None:
        """Rename zones, validate they are buses, and add the parsed domain."""
        if buses is not None:
            zonal_ptdf = zonal_ptdf.rename(columns=buses)
        self._require_buses(zonal_ptdf.columns)
        ram = ram.reindex(zonal_ptdf.index)
        return self.add(zonal_ptdf.index, zonal_ptdf=zonal_ptdf, ram=ram.values)

    def _require_buses(self, zones: pd.Index) -> None:
        """Fail fast if any zone column is not a bus in the network."""
        missing = sorted(set(zones) - set(self.n_save.c.buses.static.index))
        if missing:
            msg = (
                f"Flow-based domain references zones that are not network buses: "
                f"{missing}. Add these buses or pass a `buses` mapping to the importer."
            )
            raise ValueError(msg)

    @property
    def zonal_ptdf(self) -> pd.DataFrame:
        """Zonal PTDF sensitivities of the domain.

        Returns
        -------
        pandas.DataFrame
            Zonal power transfer distribution factors with one row per CNEC (the
            component index) and one column per zone bus. This is the stored frame
            itself, so in-place edits write through; assign a new frame or use ``add``
            to replace it. The xarray view used internally by the optimisation is
            ``c.da.zonal_ptdf``.

        """
        return self._zonal_ptdf

    def _set_frame(self, attr: str, value: Any, names: pd.Index) -> None:
        """Store a matrix-valued attribute (currently only ``zonal_ptdf``).

        ``value`` is a Series over zones (single CNEC) or a DataFrame (cnec x zone).
        Rows already present are overwritten; missing zone entries default to zero.
        """
        if attr != "zonal_ptdf":
            super()._set_frame(attr, value, names)
            return
        if isinstance(value, pd.Series):
            df = value.to_frame(names[0]).T
        else:
            df = pd.DataFrame(value).reindex(names)
        df = df.rename_axis(index="name", columns="bus").astype(float)
        keep = self._zonal_ptdf.drop(index=df.index, errors="ignore")
        self._zonal_ptdf = pd.concat([keep, df]).fillna(0.0)

    def _as_xarray(self, attr: str) -> xr.DataArray:
        """Expose ``zonal_ptdf`` as a (name, bus) DataArray; defer otherwise."""
        if attr == "zonal_ptdf":
            da = xr.DataArray(self._zonal_ptdf.rename_axis(index="name", columns="bus"))
            da.name = "zonal_ptdf"
            return da
        return super()._as_xarray(attr)
