# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Importers for published flow-based domains (ERAA, JAO, TSO).

Mixed into [pypsa.components.GlobalConstraints][]; the domain is added as global
constraints of ``type="flow_based"``. Each importer reads the zone columns, the RAM and
the corridor columns of its format and hands over one hub sensitivity per corridor (MW on
the CNEC per MW injected at the corridor's landing node, for a flow from one named end to
the other). ``_add_domain`` turns it into PyPSA's link column: the sensitivity to the link
flow with the zone net positions held fixed, ``h - (eta PTDF_bus1 - PTDF_bus0)`` over the
link's zone ends.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pypsa.common import check_optional_dependency
from pypsa.components.abstract import _ComponentsABC

logger = logging.getLogger(__name__)

_EXCEL_HINT = (
    "Missing optional dependencies to read Excel files. Install them via "
    "`pip install pypsa[excel]`."
)

if TYPE_CHECKING:
    from collections.abc import Callable

    # corridor name -> (column for a flow frm -> to, frm zone, to zone)
    Corridors = dict[str, tuple[pd.Series, str | None, str | None]]


class FlowBasedImportersMixin(_ComponentsABC):
    """Mixin adding the ``flow_based_from_eraa`` / ``_jao`` / ``_tso`` domain importers."""

    if TYPE_CHECKING:

        def add_flow_based(
            self,
            zonal_ptdf: pd.DataFrame,
            ram: pd.Series | pd.DataFrame,
            *,
            investment_period: int | None = None,
            overwrite: bool = False,
        ) -> pd.Index | None: ...

    def flow_based_from_eraa(
        self,
        path: str,
        year: str | int,
        season: str | pd.Series,
        *,
        buses: dict[str, str] | None = None,
        links: dict[str, str] | None = None,
        investment_period: int | None = None,
    ) -> pd.Index | None:
        """Add a domain from an ERAA ``FB-Domain-CORE`` Excel workbook (needs the ``excel`` extra).

        The PTDF sheet has a header row with the column kind (``PTDF_SZ`` zones,
        ``PTDF*_AHC,SZ`` and ``PTDF_EvFB`` corridors) and a row with the labels. A corridor
        label ``"A-B"`` is a flow from A to B. ERAA publishes AHC columns already in
        PyPSA's form, ``PTDF*_AHC = h - PTDF_B`` (``FB_README.xlsx``, expression 5), so they
        arrive unchanged for a link into B; the EvFB columns are hub differences and get the
        zone PTDFs subtracted.

        Parameters
        ----------
        path : str
            Path to the ERAA workbook.
        year : str or int
            Selects the ``PTDF {year}`` / ``RAM {year}`` sheets.
        season : str or pandas.Series
            A season name (e.g. ``"winter1"``) selects one static domain. A Series indexed
            by the snapshots (values = season names) builds a time-varying domain from the
            union of the seasons' CNECs; a CNEC absent in a season gets PTDF 0 and RAM
            infinite, so it never binds.
        buses : dict, optional
            Map ERAA zone labels to bus names (default: same name).
        links : dict, optional
            Map corridor labels (e.g. ``"BE00-DE00"``) to link names (default: same name).
            Corridors without a link are dropped.
        investment_period : int, optional
            Apply the domain only in this investment period; its label is appended to the
            CNEC names, so domains of several years can coexist. By default the domain
            applies in all periods.

        """
        check_optional_dependency("openpyxl", _EXCEL_HINT)
        suffix = "" if investment_period is None else f" {investment_period}"
        raw = pd.read_excel(path, sheet_name=f"PTDF {year}", header=None)
        kind, label = raw.iloc[0], raw.iloc[1]
        zones = label[kind == "PTDF_SZ"].tolist()
        ahc = label[kind == "PTDF*_AHC,SZ"].tolist()
        evfb = label[kind == "PTDF_EvFB"].tolist()
        body = raw.iloc[2:].copy()
        body.columns = list(label)
        ram_all = pd.read_excel(path, sheet_name=f"RAM {year}").set_index("CNEC_ID")

        def parse(s: str) -> tuple[pd.DataFrame, pd.Series]:
            rows = body[body["FB_ID"] == s].set_index("CNEC_ID")
            ram = ram_all[s].dropna()
            keep = rows.index.intersection(ram.index)
            ptdf = rows.loc[keep, zones + ahc + evfb].astype(float)
            return ptdf.rename(lambda c: c + suffix), ram.loc[keep].rename(
                lambda c: c + suffix
            )

        if isinstance(season, pd.Series):
            ptdf, ram = self._stack_seasons(season, parse)
        else:
            ptdf, ram = parse(season)

        corridors: Corridors = {}
        for border in ahc + evfb:
            frm, to = border.split("-", 1)
            to = to.split("_")[0]  # numbered borders, e.g. "UK00-FR00_1"
            # recover the hub sensitivity h from PTDF*_AHC = h - PTDF_to
            col = ptdf[border] + ptdf[to] if border in ahc else ptdf[border]
            corridors[border] = (col, frm, to)
        return self._add_domain(
            ptdf[zones], corridors, ram, buses, links, investment_period
        )

    def _stack_seasons(
        self, season: pd.Series, parse: Callable[[str], tuple[pd.DataFrame, pd.Series]]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stack a ``snapshot -> season`` mapping into a time-varying PTDF and RAM."""
        season = season.reindex(self.n_save.snapshots)
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
        return ptdf, ram

    def flow_based_from_jao(
        self,
        path: str,
        *,
        buses: dict[str, str] | None = None,
        links: dict[str, str] | None = None,
    ) -> pd.Index | None:
        """Add a domain from a JAO ``finalComputation`` CSV (one static market hour).

        Only the presolved rows are read (the others never bind), named by their ``Id``.
        The PTDF is read from the ``Ptdf_<hub>`` columns; the ``Direction`` is already in
        the sign. Two kinds of hub are corridors:

        - ``ALBE`` and ``ALDE``, the Belgian and German ends of the ALEGrO HVDC, form one
          corridor ``"ALBE-ALDE"`` (a flow from BE to DE).
        - AHC hubs ``<zone>_<external>_<name>`` (e.g. ``DE_SE4_Baltic``) are a flow from
          ``<external>`` into ``<zone>``.

        ``CH`` is dropped: its PTDFs are published for transparency only, as the Swiss net
        position is a fixed forecast inside the RAM. All other hubs are zones.

        Parameters
        ----------
        path : str
            Path to the JAO ``finalComputation`` CSV.
        buses : dict, optional
            Map zone hubs to bus names (default: same name).
        links : dict, optional
            Map corridor names to link names (default: same name). Corridors without a
            link are dropped.

        """
        raw = pd.read_csv(path, sep=";", low_memory=False)
        rows = raw[raw["Presolved"]]
        rows = rows.set_index(rows["Id"].astype(str))
        ptdf_cols = [c for c in rows.columns if c.startswith("Ptdf_")]
        ptdf = rows[ptdf_cols].rename(columns=lambda c: c.removeprefix("Ptdf_"))

        corridors: Corridors = {}
        if {"ALBE", "ALDE"} <= set(ptdf.columns):
            corridors["ALBE-ALDE"] = (ptdf["ALDE"] - ptdf["ALBE"], "BE", "DE")
        for hub in ptdf.columns[ptdf.columns.str.contains("_")]:
            zone, external = hub.split("_")[:2]
            corridors[hub] = (ptdf[hub], external, zone)
        not_zones = {"ALBE", "ALDE", "CH"}
        zones = [h for h in ptdf.columns if "_" not in h and h not in not_zones]
        return self._add_domain(ptdf[zones], corridors, rows["Ram"], buses, links)

    def flow_based_from_tso(
        self,
        path: str,
        *,
        buses: dict[str, str] | None = None,
        links: dict[str, str] | None = None,
    ) -> pd.Index | None:
        """Add a domain from a TSO ``MS_FBMC`` domain CSV (one static typical situation).

        A ``!!OBJEKTTYP`` row gives each column's type: ``FB_RAM`` is the RAM and
        ``FB_DOMAIN`` are zones. Corridors are:

        - ``HGUE``: the two converters of an HVDC between two zones,
          ``KONV_<A>-<B><n>_A`` and ``KONV_<A>-<B><n>_B``, form one corridor
          ``KONV_<A>-<B><n>`` (a flow from A to B).
        - ``HGUE_AHC``: the converter ``KONV_AHC_<pair>_<zone>`` of an HVDC to a
          non-flow-based zone is corridor ``KONV_AHC_<pair>``, a flow into ``<zone>``.
        - ``FB_DOMAIN_AHC``: the AC exchange of a non-flow-based zone (e.g. ``DKW``) with the
          flow-based region, a flow into the region.

        Parameters
        ----------
        path : str
            Path to the ``MS_FBMC`` domain CSV (semicolon-separated).
        buses : dict, optional
            Map zone labels to bus names (default: same name).
        links : dict, optional
            Map corridor names to link names (default: same name). Corridors without a
            link are dropped. ``FB_DOMAIN_AHC`` corridors are named like their zone, so
            map them to a link name that is not a bus name (e.g. ``{"DKW": "DKW-DE"}``).

        """
        raw = Path(path).read_text(encoding="latin-1").splitlines()
        meta = [i for i, line in enumerate(raw) if line.startswith("!")]
        header = raw[meta[-1] + 1].split(";")
        types = dict(zip(header[1:], raw[meta[-1]].split(";")[1:], strict=False))
        ram_col = next(c for c, t in types.items() if t == "FB_RAM")
        cells = (line.split(";")[header.index(ram_col)] for line in raw[meta[-1] + 2 :])
        decimal = "," if any("," in c for c in cells) else "."
        df = pd.read_csv(
            path, sep=";", skiprows=meta, encoding="latin-1", decimal=decimal
        ).set_index("CNEC_ID")

        def of_type(t: str) -> list[str]:
            return [c for c in header if types.get(c) == t]

        corridors: Corridors = {}
        for col in of_type("HGUE"):
            name, zone = col.rsplit("_", 1)
            frm = name.split("_")[-1].split("-")[0]
            if zone != frm:  # each pair once, from the receiving end
                corridors[name] = (df[col] - df[f"{name}_{frm}"], frm, zone)
        for col in of_type("HGUE_AHC"):
            name, zone = col.rsplit("_", 1)
            corridors[name] = (df[col], None, zone)
        for col in of_type("FB_DOMAIN_AHC"):
            corridors[col] = (df[col], None, None)
        zones = of_type("FB_DOMAIN")
        return self._add_domain(df[zones], corridors, df[ram_col], buses, links)

    def _add_domain(
        self,
        zonal_ptdf: pd.DataFrame,
        corridors: Corridors,
        ram: pd.Series | pd.DataFrame,
        buses: dict[str, str] | None,
        links: dict[str, str] | None,
        investment_period: int | None = None,
    ) -> pd.Index | None:
        """Add zone columns plus one column per corridor that has a network link.

        A corridor's hub sensitivity describes a flow ``frm -> to``; its sign is flipped if
        the link runs ``to -> frm``, and the zone PTDFs of the link's ends are subtracted so
        the column is the sensitivity with the zone net positions held fixed. A ``None``
        zone is not checked; ``to=None`` means the link's only flow-based end.
        """
        buses, links = buses or {}, links or {}
        if unknown := sorted(set(links) - set(corridors)):
            msg = f"{unknown} are not corridors of this domain; available: {list(corridors)}."
            raise ValueError(msg)

        n = self.n_save
        zonal_ptdf = zonal_ptdf.rename(columns=buses)
        if missing := sorted(set(zonal_ptdf.columns) - set(n.c.buses.static.index)):
            msg = f"Zones {missing} are not network buses; pass a `buses` mapping."
            raise ValueError(msg)

        zones = set(zonal_ptdf.columns)
        static = n.c.links.static
        dropped = []
        for name, (col, frm, to) in corridors.items():
            link = links.get(name, name)
            if link not in static.index:
                dropped.append(name)
                continue
            if link in n.c.buses.static.index:
                msg = f"Link name {link!r} is also a bus name; rename it via `links`."
                raise ValueError(msg)
            bus0, bus1 = static.loc[link, ["bus0", "bus1"]]
            ends = {bus0, bus1} & zones
            frm, to = (buses.get(x, x) if x else None for x in (frm, to))
            to = to or (next(iter(ends)) if len(ends) == 1 else None)
            if to not in (bus0, bus1) or ends != {frm, to} & zones:
                msg = (
                    f"Link {link!r} ({bus0} -> {bus1}) does not fit corridor {name!r} "
                    f"({frm} -> {to}); check the `buses` mapping."
                )
                raise ValueError(msg)
            h = col if to == bus1 else -col
            eta = static.at[link, "efficiency"]
            shift = sum(
                a * zonal_ptdf[b] for b, a in ((bus0, -1), (bus1, eta)) if b in zones
            )
            zonal_ptdf[link] = h - shift
        if dropped:
            logger.warning(
                "Dropping %d corridor(s) without a network link: %s. Add links of the "
                "same name or pass a `links` mapping.",
                len(dropped),
                dropped,
            )

        return self.add_flow_based(zonal_ptdf, ram, investment_period=investment_period)
