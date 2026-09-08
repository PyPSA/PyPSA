# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Importers for published flow-based domains (ERAA, JAO, TSO).

Mixed into [pypsa.components.FlowBasedConstraints][]; each parser builds a
``(zonal_ptdf, ram)`` pair and forwards it to ``add``. See the flow-based constraint
user guide for the file formats and mappings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.common import check_optional_dependency
from pypsa.components.abstract import _ComponentsABC

logger = logging.getLogger(__name__)

_EXCEL_HINT = (
    "Missing optional dependencies to read Excel files. Install them via "
    "`pip install pypsa[excel]`."
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class FlowBasedImportersMixin(_ComponentsABC):
    """Mixin adding the ``from_eraa`` / ``from_jao`` / ``from_tso`` domain importers."""

    if TYPE_CHECKING:

        def add(
            self,
            name: str | int | Sequence[int | str],
            suffix: str | Sequence[str] = "",
            overwrite: bool = False,
            return_names: bool | None = None,
            **kwargs: Any,
        ) -> pd.Index | None: ...

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
        """Add a domain from an ERAA ``FB-Domain-CORE`` Excel workbook (needs the ``excel`` extra).

        The PTDF sheet has a *kind* header row (``PTDF_SZ`` zones, ``PTDF*_AHC,SZ`` and
        ``PTDF_EvFB`` corridors) and a *label* row.

        Parameters
        ----------
        path : str
            Path to the ERAA workbook.
        year : str or int
            Selects the ``PTDF {year}`` / ``RAM {year}`` sheets.
        season : str or pandas.Series
            A season name (e.g. ``"winter1"``) selects one static domain. A Series indexed
            by the snapshots (values = season names) builds a time-varying domain from the
            union of the seasons' CNECs; a CNEC absent in a snapshot's season gets PTDF 0
            and RAM infinite, so it never binds that hour.
        buses, links : dict, optional
            Map ERAA zone / border labels (``"A-B"``) to network bus / link names. The link
            sign is aligned to the link's ``bus0 -> bus1`` orientation; unmapped corridors
            are dropped.
        ptdf_sheet, ram_sheet : str, optional
            Override the default sheet names.

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

        if unknown := [b for b in (links or {}) if b not in corridors]:
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
        """Stack a ``snapshot -> season`` mapping into a time-varying domain (union of CNECs)."""
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
        """Sign aligning an ERAA border ``"A-B"`` (flow A->B) to a link's ``bus0 -> bus1``."""
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
        """Add a domain from a JAO ``finalComputation`` CSV (one static market hour).

        The zonal PTDF is read from the ``Ptdf_<hub>`` columns (the prefix is stripped); the
        ``Direction`` is already baked into the sign.

        Parameters
        ----------
        path : str
            Path to the JAO ``finalComputation`` CSV.
        presolved : bool, default True
            Keep only the presolved rows (the actual domain).
        buses, links : dict, optional
            Map hub names to network bus / link names. JAO hubs are undirected, so a link
            column is only renamed, not sign-adjusted; flip it if your link orientation
            differs.
        name_col : str, default "Id"
            Unique CNEC name column (``CneName`` is not unique across directions).
        sep : str, default ";"
            CSV field separator.

        """
        raw = pd.read_csv(path, sep=sep, low_memory=False)
        rows = (raw[raw["Presolved"]] if presolved else raw).copy()
        rows[name_col] = rows[name_col].astype(str)
        rows = rows.set_index(name_col)

        ptdf_cols = [c for c in rows.columns if c.startswith("Ptdf_")]
        zonal_ptdf = rows[ptdf_cols].rename(columns=lambda c: c.removeprefix("Ptdf_"))
        mapping = {**(buses or {}), **(links or {})}
        return self._add_domain_frame(zonal_ptdf, rows["Ram"], mapping)

    def from_tso(
        self,
        path: str,
        *,
        buses: dict[str, str] | None = None,
        links: dict[str, str] | None = None,
        encoding: str = "latin-1",
        decimal: str | None = None,
    ) -> pd.Index | None:
        """Add a domain from a TSO ``MS_FBMC`` domain CSV (one static typical situation).

        A ``!!OBJEKTTYP`` header row types every column: ``RAM_MW`` is the RAM,
        ``FB_DOMAIN``/``FB_DOMAIN_AHC`` are zones, ``HGUE``/``HGUE_AHC`` are HVDC converters.

        Parameters
        ----------
        path : str
            Path to the ``MS_FBMC`` domain CSV (semicolon-separated).
        buses, links : dict, optional
            Map zone labels / converter columns (e.g. ``"KONV_BE-DE1_DE"``) to network bus /
            link names. Converter columns are not sign-adjusted; unmapped ones are dropped.
        encoding : str, default "latin-1"
            File encoding.
        decimal : str, optional
            Decimal separator; auto-detected from the ``RAM_MW`` column when omitted.

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
            cells = (
                line.split(";")[header.index(ram_col)] for line in raw[meta[-1] + 2 :]
            )
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
        """Rename columns, check they are buses/links, and add the parsed domain."""
        if mapping:
            zonal_ptdf = zonal_ptdf.rename(columns=mapping)
        self._require_components(zonal_ptdf.columns)
        ram = ram.reindex(zonal_ptdf.index)
        return self.add(zonal_ptdf.index, zonal_ptdf=zonal_ptdf, ram=ram.values)

    def _require_components(self, columns: pd.Index) -> None:
        """Fail fast if any domain column is neither a bus (zone) nor a link."""
        n = self.n_save
        known = set(n.c.buses.static.index) | set(n.c.links.static.index)
        if missing := sorted(set(columns) - known):
            msg = (
                f"Flow-based domain references columns that are not network buses or "
                f"links: {missing}. Add them or pass a `buses` mapping to the importer."
            )
            raise ValueError(msg)
