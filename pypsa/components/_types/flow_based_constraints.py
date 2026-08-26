# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Flow-based domain components module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

from pypsa.common import check_optional_dependency
from pypsa.components._types._patch import patch_add_docstring
from pypsa.components.components import Components

logger = logging.getLogger(__name__)

_EXCEL_HINT = (
    "Missing optional dependencies to read Excel files. Install them via "
    "`pip install pypsa[excel]`."
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa import Network
    from pypsa.components.types import ComponentType


@patch_add_docstring
class FlowBasedConstraints(Components):
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
    A time-varying domain passes a ``(snapshot, cnec)`` MultiIndex frame instead, and the
    frame keeps that MultiIndex; the whole domain is static or time-varying, not mixed.

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
        season: str | pd.Series,
        *,
        buses: dict[str, str] | None = None,
        links: dict[str, str] | None = None,
        ptdf_sheet: str | None = None,
        ram_sheet: str | None = None,
    ) -> pd.Index | None:
        """Add a flow-based domain from an ERAA ``FB-Domain-CORE`` Excel workbook.

        The zonal PTDF sheet carries two header rows: a *kind* row (``PTDF_SZ`` for the
        study-zone sensitivities, ``PTDF*_AHC,SZ`` and ``PTDF_EvFB`` for the advanced
        hybrid-coupling and evolved corridors) and a *label* row (the zone / border names
        plus ``FB_ID`` and ``CNEC_ID``).

        Parameters
        ----------
        path : str
            Path to the ERAA workbook (requires the ``openpyxl`` extra to read ``.xlsx``).
        year : str or int
            Target year, selecting the ``PTDF {year}`` and ``RAM {year}`` sheets.
        season : str or pandas.Series
            A single ``FB_ID`` value (e.g. ``"winter1"``) selects one time-invariant
            domain. Passing a Series indexed by the network snapshots, whose values are
            season names, builds a *time-varying* domain: each snapshot's domain is the
            season it maps to. The per-season CNEC sets are unioned; where a CNEC is
            absent in a snapshot's season, its PTDF row is zero and its RAM infinite, so
            that constraint never binds that hour.
        buses : dict, optional
            Explicit mapping from ERAA zone labels to network bus names. Labels are used
            as-is where unmapped; no fuzzy matching is done.
        links : dict, optional
            Explicit mapping from AHC/EvFB border labels (of the form ``"A-B"``) to
            network link names, to include those corridors as ``Link-p`` terms. The sign
            is aligned automatically to the link's ``bus0 -> bus1`` orientation (the ERAA
            PTDF is defined for flow from ``A`` to ``B``). Endpoints are resolved through
            ``buses``. Unmapped AHC/EvFB columns are dropped.
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
        corridors = label[kind.isin(["PTDF*_AHC,SZ", "PTDF_EvFB"])].tolist()
        body = raw.iloc[2:].copy()
        body.columns = list(label)
        ram_all = pd.read_excel(path, sheet_name=ram_sheet).set_index("CNEC_ID")

        unknown = [border for border in (links or {}) if border not in corridors]
        if unknown:
            msg = f"{unknown} are not ERAA AHC/EvFB columns; available: {corridors}."
            raise ValueError(msg)
        if dropped := [c for c in corridors if c not in (links or {})]:
            logger.warning(
                "Dropping %d unmapped ERAA AHC/EvFB corridor(s): %s. Pass them in "
                "`links` to include them as link terms.",
                len(dropped),
                dropped,
            )

        def parse(s: str) -> tuple[pd.DataFrame, pd.Series]:
            """Parse one ERAA season into its ``(zonal_ptdf, ram)`` pair (CNEC-indexed)."""
            rows = body[body["FB_ID"] == s].set_index("CNEC_ID")
            ram = ram_all[s]
            keep = rows.index[rows.index.isin(ram.dropna().index)]
            zonal_ptdf = rows.loc[keep, zones].astype(float)
            for border, link in (links or {}).items():
                sign = self._link_sign(border, link, buses or {})
                zonal_ptdf[link] = rows.loc[keep, border].astype(float) * sign
            return zonal_ptdf, ram.loc[keep]

        if isinstance(season, pd.Series):
            return self._add_eraa_dynamic(season, parse, buses)
        zonal_ptdf, ram = parse(season)
        return self._add_domain_frame(zonal_ptdf, ram, buses)

    def _add_eraa_dynamic(
        self,
        season: pd.Series,
        parse: Any,
        buses: dict[str, str] | None,
    ) -> pd.Index | None:
        """Assemble a time-varying domain from a ``snapshot -> ERAA season`` mapping.

        Each season is parsed once; the per-season CNEC sets are unioned. Where a CNEC is
        absent in the season a snapshot maps to, its PTDF row is zero and its RAM infinite,
        so the constraint is present but never binds that hour.
        """
        sns = self.n_save.snapshots
        season = season.reindex(sns)
        if season.isna().any():
            msg = "`season` Series must map every network snapshot to an ERAA season."
            raise ValueError(msg)
        parsed = {s: parse(s) for s in season.unique()}
        cnecs = pd.Index(sorted(set().union(*(zp.index for zp, _ in parsed.values()))))
        ptdf = pd.concat(
            {t: parsed[s][0].reindex(cnecs).fillna(0.0) for t, s in season.items()},
            names=["snapshot", "name"],
        )
        ram = pd.DataFrame(
            {t: parsed[s][1].reindex(cnecs) for t, s in season.items()}
        ).T.fillna(float("inf"))
        if buses:
            ptdf = ptdf.rename(columns=buses)
        self._require_components(ptdf.columns)
        return self.add(cnecs, zonal_ptdf=ptdf, ram=ram)

    def _link_sign(self, border: str, link: str, buses: dict[str, str]) -> float:
        """Sign aligning an ERAA border ``"A-B"`` (flow A->B) to a link's bus0->bus1."""
        frm, to = (buses.get(x, x) for x in border.split("-", 1))
        static = self.n_save.c.links.static
        if link not in static.index:
            msg = f"{link!r} is not a network link."
            raise ValueError(msg)
        ends = (static.at[link, "bus0"], static.at[link, "bus1"])
        if ends == (frm, to):
            return 1.0
        if ends == (to, frm):
            return -1.0
        msg = (
            f"Link {link!r} ({ends[0]} -> {ends[1]}) does not connect the border "
            f"{border!r} endpoints ({frm}, {to}); check the `buses` mapping."
        )
        raise ValueError(msg)

    def from_jao(
        self,
        path: str,
        *,
        presolved: bool = True,
        buses: dict[str, str] | None = None,
        links: dict[str, str] | None = None,
        name_col: str = "Id",
        sep: str = ";",
    ) -> pd.Index | None:
        """Add a flow-based domain from a JAO ``finalComputation`` CSV.

        The zonal PTDF is given directly as ``Ptdf_<hub>`` columns; the ``Ptdf_`` prefix
        is stripped to obtain the hub name. The ``Direction`` (DIRECT/OPPOSITE) is already
        baked into the PTDF sign, so no sign duplication is applied. The domain is assumed
        time-invariant (one CSV = one market hour).

        Parameters
        ----------
        path : str
            Path to the JAO ``finalComputation`` CSV.
        presolved : bool, default True
            Keep only the presolved rows (the actual domain); if False, all rows are read.
        buses : dict, optional
            Explicit mapping from hub names (after stripping ``Ptdf_``) to network bus
            names. Hubs are used as-is where unmapped; no fuzzy matching is done.
        links : dict, optional
            Explicit mapping from hub names to network link names, to include an external
            virtual hub as a ``Link-p`` term. JAO hubs are undirected labels, so the sign
            is not adjusted: ensure the link's ``bus0 -> bus1`` orientation matches the
            hub's net-position convention, or flip the column.
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

        return self._add_domain_frame(zonal_ptdf, rows["Ram"], {**(buses or {}), **(links or {})})

    def from_tso(
        self,
        path: str,
        *,
        buses: dict[str, str] | None = None,
        links: dict[str, str] | None = None,
        encoding: str = "latin-1",
        decimal: str | None = None,
    ) -> pd.Index | None:
        """Add a flow-based domain from a TSO ``MS_FBMC`` domain CSV.

        The file has a ``!``/``!!`` metadata header ending in an ``!!OBJEKTTYP`` row that
        types every column, followed by the column header and the data. Only this one file
        is read: ``RAM_MW`` (``FB_RAM``) is the RAM, ``FB_DOMAIN``/``FB_DOMAIN_AHC`` columns
        are the zones, and ``HGUE``/``HGUE_AHC`` columns are the HVDC converters. The domain
        is assumed time-invariant (one ``Domain_TS`` file).

        Parameters
        ----------
        path : str
            Path to the ``MS_FBMC`` domain CSV (semicolon-separated).
        buses : dict, optional
            Explicit mapping from zone labels to network bus names. Labels are used as-is
            where unmapped; no fuzzy matching is done.
        links : dict, optional
            Explicit mapping from converter column names (e.g. ``"KONV_BE-DE1_DE"``) to
            network link names, to include them as ``Link-p`` terms. The sign is not
            adjusted: ensure the link's ``bus0 -> bus1`` orientation matches the converter
            convention, or flip the column. Unmapped converters are dropped.
        encoding : str, default "latin-1"
            File encoding (TSO files are typically Latin-1).
        decimal : str, optional
            Decimal separator; auto-detected from the ``RAM_MW`` column (German ``","`` vs
            English ``"."``) when not given.

        Returns
        -------
        pandas.Index or None
            Names of the added CNECs (see [`add`][pypsa.Network.add]).

        """
        raw = Path(path).read_text(encoding=encoding).splitlines()
        meta = [i for i, line in enumerate(raw) if line.startswith("!")]
        objtyp = raw[meta[-1]].split(";")[1:]  # !!OBJEKTTYP row, aligns to header[1:]
        header = raw[meta[-1] + 1].split(";")
        types = {
            header[i + 1]: t
            for i, t in enumerate(objtyp)
            if i + 1 < len(header) and header[i + 1]
        }

        ram_col = next(c for c in header if types.get(c) == "FB_RAM")
        zones = [c for c in header if types.get(c) in ("FB_DOMAIN", "FB_DOMAIN_AHC")]
        converters = [c for c in header if types.get(c) in ("HGUE", "HGUE_AHC")]

        if decimal is None:
            cells = (line.split(";")[header.index(ram_col)] for line in raw[meta[-1] + 2 :])
            decimal = "," if any("," in c for c in cells) else "."

        df = pd.read_csv(
            path, sep=";", skiprows=meta, encoding=encoding, decimal=decimal
        ).set_index("CNEC_ID")

        zonal_ptdf = df[zones].astype(float)
        for col, link in (links or {}).items():
            if col not in converters:
                msg = f"{col!r} is not a TSO converter column; available: {converters}."
                raise ValueError(msg)
            zonal_ptdf[link] = df[col].astype(float)

        if dropped := [c for c in converters if c not in (links or {})]:
            logger.warning(
                "Dropping %d unmapped TSO converter(s): %s. Pass them in `links` to "
                "include them as link terms.",
                len(dropped),
                dropped,
            )

        return self._add_domain_frame(zonal_ptdf, df[ram_col], buses)

    def _add_domain_frame(
        self,
        zonal_ptdf: pd.DataFrame,
        ram: pd.Series,
        mapping: dict[str, str] | None,
    ) -> pd.Index | None:
        """Rename columns, validate they are buses/links, and add the parsed domain."""
        if mapping:
            zonal_ptdf = zonal_ptdf.rename(columns=mapping)
        self._require_components(zonal_ptdf.columns)
        ram = ram.reindex(zonal_ptdf.index)
        return self.add(zonal_ptdf.index, zonal_ptdf=zonal_ptdf, ram=ram.values)

    def _require_components(self, columns: pd.Index) -> None:
        """Fail fast if any domain column is neither a bus (zone) nor a link."""
        n = self.n_save
        known = set(n.c.buses.static.index) | set(n.c.links.static.index)
        missing = sorted(set(columns) - known)
        if missing:
            msg = (
                f"Flow-based domain references columns that are not network buses or "
                f"links: {missing}. Add them or pass a `buses` mapping to the importer."
            )
            raise ValueError(msg)

    @property
    def zonal_ptdf(self) -> pd.DataFrame:
        """Zonal PTDF sensitivities of the domain.

        Returns
        -------
        pandas.DataFrame
            Zonal power transfer distribution factors, one column per zone bus. For a
            static domain the index is the CNEC (component index); for a time-varying
            domain it is a ``(snapshot, CNEC)`` MultiIndex, so ``c.zonal_ptdf.loc[sns]``
            selects one snapshot's matrix. This is the stored frame itself, so in-place
            edits write through; assign a new frame or use ``add`` to replace it. The
            xarray view used internally by the optimisation is ``c.da.zonal_ptdf``
            (dims ``(name, bus)``, gaining a ``snapshot`` dim when time-varying).

        """
        return self._zonal_ptdf

    def _set_frame(self, attr: str, value: Any, names: pd.Index) -> None:
        """Store a matrix-valued attribute (currently only ``zonal_ptdf``).

        ``value`` is a Series over zones (single CNEC), a DataFrame (cnec x zone), or a
        time-varying DataFrame with a ``(snapshot, cnec)`` MultiIndex and zones as
        columns. Rows already present are overwritten; missing zone entries default to
        zero. A domain is static or time-varying as a whole; mixing the two raises.
        """
        if attr != "zonal_ptdf":
            super()._set_frame(attr, value, names)
            return
        if isinstance(value, pd.DataFrame) and isinstance(value.index, pd.MultiIndex):
            df = self._time_varying_frame(value)
        elif isinstance(value, pd.Series):
            df = value.to_frame(names[0]).T.rename_axis(index="name", columns="bus")
        else:
            frame = pd.DataFrame(value).reindex(names)
            df = frame.rename_axis(index="name", columns="bus")
        df = df.astype(float)
        existing = self._zonal_ptdf
        if not existing.empty and existing.index.nlevels != df.index.nlevels:
            msg = "Cannot mix static and time-varying zonal PTDF rows in one domain."
            raise ValueError(msg)
        keep = existing.drop(index=df.index, errors="ignore")
        self._zonal_ptdf = pd.concat([keep, df]).fillna(0.0)

    def _time_varying_frame(self, value: pd.DataFrame) -> pd.DataFrame:
        """Validate and label a time-varying ``(snapshot, cnec) x zone`` frame."""
        if not value.index.get_level_values(0).isin(self.n_save.snapshots).all():
            msg = (
                "Time-varying zonal_ptdf must be indexed by (snapshot, CNEC); its outer "
                "index level must be network snapshots."
            )
            raise ValueError(msg)
        return value.rename_axis(index=["snapshot", "name"], columns="bus")

    def _as_xarray(self, attr: str) -> xr.DataArray:
        """Expose ``zonal_ptdf`` as a (name, bus) or (snapshot, name, bus) DataArray."""
        if attr == "zonal_ptdf":
            z = self._zonal_ptdf
            if isinstance(z.index, pd.MultiIndex):
                da = z.stack(future_stack=True).to_xarray()
            else:
                da = xr.DataArray(z.rename_axis(index="name", columns="bus"))
            da.name = "zonal_ptdf"
            return da
        return super()._as_xarray(attr)
