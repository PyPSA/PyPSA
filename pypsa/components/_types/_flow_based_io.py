# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Importers for published flow-based domains (ERAA, JAO, TSO).

Mixed into [pypsa.components.GlobalConstraints][]; the domain is added as global
constraints of ``type="flow_based"``. Each importer reads the zone PTDFs, the RAM and the
hub sensitivities of its format and hands them to ``add_flow_based_from_hubs``, which
turns the hubs into PyPSA's link columns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

from pypsa.common import check_optional_dependency
from pypsa.components.abstract import _ComponentsABC

logger = logging.getLogger(__name__)


def _corridors(rows: list[tuple[str | None, ...]]) -> pd.DataFrame:
    """Corridor table from ``(name, bus0, bus1, hub0, hub1)`` rows."""
    columns = ["name", "bus0", "bus1", "hub0", "hub1"]
    return pd.DataFrame(rows, columns=columns).set_index("name")


_EXCEL_HINT = (
    "Missing optional dependencies to read Excel files. Install them via "
    "`pip install pypsa[excel]`."
)

if TYPE_CHECKING:
    from collections.abc import Callable

    # corridor name -> link name, or several (e.g. a pair of one-way links)
    LinkMap = dict[str, str | list[str]]


class FlowBasedImportersMixin(_ComponentsABC):
    """Mixin adding ``add_flow_based_from_hubs`` and the ERAA, JAO and TSO importers."""

    if TYPE_CHECKING:

        def add_flow_based(
            self,
            zonal_ptdf: pd.DataFrame | pd.Series,
            ram: float | pd.Series | pd.DataFrame,
            investment_period: int | None = None,
            domain: str | None = None,
            overwrite: bool = False,
        ) -> None: ...

    def flow_based_from_eraa(
        self,
        path: str,
        year: str | int,
        fb_id: str | int | pd.Series | None = None,
        *,
        weather_scenario: str | None = None,
        buses: dict[str, str] | None = None,
        links: LinkMap | None = None,
        investment_period: int | None = None,
        domain: str | None = None,
    ) -> None:
        """Add a domain from an ERAA ``FB-Domain-*`` Excel workbook (needs the ``excel`` extra).

        Reads the layout of ERAA 2025 (Core and Nordic). The PTDF sheet has a header row
        with the column kind (``PTDF_SZ`` zones, ``PTDF*_AHC,SZ`` and ``PTDF_EvFB``
        corridors) and a row with the labels. If the sheet holds a single ``FB_ID``
        (Nordic), its PTDF serves every RAM column. A corridor label ``"A-B"`` is a flow
        from A to B. An AHC column is ``h - PTDF_B`` (``FB_README.xlsx``, expression 5), so
        the hub sensitivity at B is recovered as ``h``. An EvFB column is the difference of
        its two hubs; it is exact for a lossless link only.

        Parameters
        ----------
        path : str
            Path to the ERAA workbook.
        year : str or int
            Selects the ``PTDF {year}`` / ``RAM {year}`` sheets.
        fb_id : str, int or pandas.Series, optional
            A domain ID, i.e. a RAM column such as ``"winter1"`` (Core) or ``3`` (Nordic),
            selects one static domain. A Series indexed by the snapshots (values = domain
            IDs) builds a time-varying domain from the union of the IDs' CNECs; a CNEC
            absent under an ID gets NaN RAM, so it is dropped in those snapshots.
        weather_scenario : str, optional
            Instead of ``fb_id``, take the domain ID of each snapshot from the column of
            this weather scenario (e.g. ``"WS01"``) in the ``Domain Assignment`` sheet,
            matched by month, day and hour of the target ``year``. Needs datetime
            snapshots.
        buses : dict, optional
            Map ERAA zone labels to bus names (default: same name).
        links : dict, optional
            Map corridor labels (e.g. ``"BE00-DE00"``) to link names (default: same name).
            A list maps one corridor to several links, e.g. a pair of one-way links.
            Corridors without a link are dropped.
        investment_period : int, optional
            Apply the domain only in this investment period; its label is appended to the
            CNEC names, so domains of several years can coexist. By default the domain
            applies in all periods.
        domain : str, optional
            Name of the flow-based domain, e.g. ``"Core"``; see
            [add_flow_based][pypsa.components.GlobalConstraints.add_flow_based]. It
            prefixes the CNEC names, e.g. ``"Core cnec_18"``.

        """
        check_optional_dependency("openpyxl", _EXCEL_HINT)
        if (fb_id is None) == (weather_scenario is None):
            msg = "Pass exactly one of `fb_id` and `weather_scenario`."
            raise ValueError(msg)
        if weather_scenario is not None:
            fb_id = self._eraa_assignment(path, int(year), weather_scenario)
        prefix = f"{domain} " if domain else ""
        suffix = "" if investment_period is None else f" {investment_period}"
        raw = pd.read_excel(path, sheet_name=f"PTDF {year}", header=None)
        kind, label = raw.iloc[0], raw.iloc[1]
        zones = label[kind == "PTDF_SZ"].tolist()
        ahc = label[kind == "PTDF*_AHC,SZ"].tolist()
        evfb = label[kind == "PTDF_EvFB"].tolist()
        body = raw.iloc[2:].copy()
        body.columns = list(label)
        ram_all = pd.read_excel(path, sheet_name=f"RAM {year}").set_index("CNEC_ID")

        fb_ids = body["FB_ID"].unique()

        def parse(s: str | int) -> tuple[pd.DataFrame, pd.Series]:
            ptdf_id = fb_ids[0] if len(fb_ids) == 1 else s
            rows = body[body["FB_ID"] == ptdf_id].set_index("CNEC_ID")
            ram = ram_all[s].dropna()
            keep = rows.index.intersection(ram.index)
            ptdf = rows.loc[keep, zones + ahc + evfb].astype(float)
            name = f"{prefix}{{}}{suffix}".format
            return ptdf.rename(name), ram.loc[keep].rename(name)

        if isinstance(fb_id, pd.Series):
            ptdf, ram = self._stack_domains(fb_id, parse)
        else:
            ptdf, ram = parse(cast("str | int", fb_id))

        corridors: list[tuple[str | None, ...]] = []
        for border in ahc + evfb:
            frm, to = border.split("-", 1)
            to = to.split("_")[0]  # numbered borders, e.g. "UK00-FR00_1"
            corridors.append((border, frm, to, None, border))
            if border in ahc:  # PTDF*_AHC = h - PTDF_to
                ptdf[border] += ptdf[to]
        self.add_flow_based_from_hubs(
            ptdf[zones],
            ptdf,
            _corridors(corridors),
            ram,
            buses=buses,
            links=links,
            investment_period=investment_period,
            domain=domain,
        )

    def _eraa_assignment(self, path: str, year: int, scenario: str) -> pd.Series:
        """Map each snapshot to its ERAA domain ID in one weather scenario."""
        timesteps = self.n_save.snapshots.get_level_values(-1)
        if not isinstance(timesteps, pd.DatetimeIndex):
            msg = "`weather_scenario` needs datetime snapshots."
            raise TypeError(msg)
        sheet = pd.read_excel(path, sheet_name="Domain Assignment")
        ids = sheet[sheet["Year"] == year].set_index(["Month", "Day", "Hour"])[scenario]
        keys = pd.MultiIndex.from_arrays(
            [timesteps.month, timesteps.day, timesteps.hour]
        )
        return pd.Series(ids.reindex(keys).to_numpy(), index=self.n_save.snapshots)

    def _stack_domains(
        self,
        fb_id: pd.Series,
        parse: Callable[[str | int], tuple[pd.DataFrame, pd.Series]],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stack a ``snapshot -> domain ID`` mapping into a time-varying PTDF and RAM."""
        fb_id = fb_id.reindex(self.n_save.snapshots)
        if fb_id.isna().any():
            msg = "The domain IDs must map every network snapshot to an ERAA domain."
            raise ValueError(msg)
        parsed = {s: parse(s) for s in fb_id.unique()}
        cnecs = pd.Index(sorted(set().union(*(zp.index for zp, _ in parsed.values()))))
        ptdf = pd.concat(
            {t: parsed[s][0].reindex(cnecs).fillna(0.0) for t, s in fb_id.items()},
            names=["snapshot", "name"],
        )
        ram = pd.DataFrame({t: parsed[s][1].reindex(cnecs) for t, s in fb_id.items()}).T
        return ptdf, ram

    def flow_based_from_jao(
        self,
        path: str,
        *,
        buses: dict[str, str] | None = None,
        links: LinkMap | None = None,
        domain: str | None = None,
    ) -> None:
        """Add a domain from a JAO ``finalComputation`` CSV (one static market hour).

        Only the presolved rows are read (the others never bind), named by their ``Id``.
        The PTDF is read from the ``Ptdf_<hub>`` columns; the ``Direction`` is already in
        the sign. Two kinds of hub are corridors:

        - ``ALBE`` and ``ALDE``, the Belgian and German hubs of the ALEGrO HVDC, form one
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
            Map corridor names to link names (default: same name). A list maps one
            corridor to several links, e.g. a pair of one-way links. Corridors without a
            link are dropped.
        domain : str, optional
            Name of the flow-based domain, e.g. ``"Core"``; see
            [add_flow_based][pypsa.components.GlobalConstraints.add_flow_based]. It
            prefixes the CNEC names, e.g. ``"Core 12345"``.

        """
        raw = pd.read_csv(path, sep=";", low_memory=False)
        rows = raw[raw["Presolved"]]
        prefix = f"{domain} " if domain else ""
        rows = rows.set_index(prefix + rows["Id"].astype(str))
        ptdf_cols = [c for c in rows.columns if c.startswith("Ptdf_")]
        ptdf = rows[ptdf_cols].rename(columns=lambda c: c.removeprefix("Ptdf_"))

        corridors: list[tuple[str | None, ...]] = []
        if {"ALBE", "ALDE"} <= set(ptdf.columns):
            corridors.append(("ALBE-ALDE", "BE", "DE", "ALBE", "ALDE"))
        for hub in ptdf.columns[ptdf.columns.str.contains("_")]:
            zone, external = hub.split("_")[:2]
            corridors.append((hub, external, zone, None, hub))
        not_zones = {"ALBE", "ALDE", "CH"}
        zones = [h for h in ptdf.columns if "_" not in h and h not in not_zones]
        self.add_flow_based_from_hubs(
            ptdf[zones],
            ptdf,
            _corridors(corridors),
            rows["Ram"],
            buses=buses,
            links=links,
            domain=domain,
        )

    def flow_based_from_tso(
        self,
        path: str,
        *,
        buses: dict[str, str] | None = None,
        links: LinkMap | None = None,
        domain: str | None = None,
    ) -> None:
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
            Map corridor names to link names (default: same name). A list maps one
            corridor to several links, e.g. a pair of one-way links. Corridors without a
            link are dropped. ``FB_DOMAIN_AHC`` corridors are named like their zone, so
            map them to a link name that is not a bus name (e.g. ``{"DKW": "DKW-DE"}``).
        domain : str, optional
            Name of the flow-based domain, e.g. ``"Core"``; see
            [add_flow_based][pypsa.components.GlobalConstraints.add_flow_based]. It
            prefixes the CNEC names.

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
        df.index = (f"{domain} " if domain else "") + df.index.astype(str)

        def of_type(t: str) -> list[str]:
            return [c for c in header if types.get(c) == t]

        corridors: list[tuple[str | None, ...]] = []
        for col in of_type("HGUE"):
            name, zone = col.rsplit("_", 1)
            frm = name.split("_")[-1].split("-")[0]
            if zone != frm:  # each pair once, from the receiving end
                corridors.append((name, frm, zone, f"{name}_{frm}", col))
        for col in of_type("HGUE_AHC"):
            name, zone = col.rsplit("_", 1)
            corridors.append((name, None, zone, None, col))
        corridors += [(c, None, None, None, c) for c in of_type("FB_DOMAIN_AHC")]
        self.add_flow_based_from_hubs(
            df[of_type("FB_DOMAIN")],
            df,
            _corridors(corridors),
            df[ram_col],
            buses=buses,
            links=links,
            domain=domain,
        )

    def add_flow_based_from_hubs(
        self,
        zonal_ptdf: pd.DataFrame,
        hub_ptdf: pd.DataFrame,
        corridors: pd.DataFrame,
        ram: float | pd.Series | pd.DataFrame,
        *,
        buses: dict[str, str] | None = None,
        links: LinkMap | None = None,
        investment_period: int | None = None,
        domain: str | None = None,
    ) -> None:
        """Add a flow-based domain whose corridors are given by hub sensitivities.

        Published domains (Core, JAO, TSO files) describe a controllable corridor, e.g.
        an HVDC link, by its hubs: the CNEC's sensitivity to 1 MW injected where the
        corridor lands in a zone. This method turns them into PyPSA's link columns for
        [add_flow_based][pypsa.components.GlobalConstraints.add_flow_based]. For a link
        ``bus0 -> bus1`` with efficiency ``eta``, the sensitivity to its flow is
        ``h = eta * hub1 - hub0``, and its column is ``h - (eta * PTDF_bus1 - PTDF_bus0)``
        over the link's zone ends, the sensitivity with zone net positions held fixed.

        Parameters
        ----------
        zonal_ptdf : pandas.DataFrame
            Zone-to-slack PTDF, CNEC x zone, or ``(snapshot, CNEC)`` x zone if
            time-varying.
        hub_ptdf : pandas.DataFrame
            Hub sensitivities with the same index; the ``corridors`` name its columns.
        corridors : pandas.DataFrame
            One row per corridor, indexed by its name, for a flow ``bus0 -> bus1``.
            Columns ``bus0`` and ``bus1`` are the zones at the ends (NaN outside the
            domain; a NaN ``bus1`` means the link's only zone end), ``hub0`` and ``hub1``
            the ``hub_ptdf`` columns at the ends (NaN: no hub, sensitivity 0). A link
            with ``efficiency != 1`` needs a hub at each of its zone ends.
        ram : float, pandas.Series or pandas.DataFrame
            Remaining available margin, as in ``add_flow_based``.
        buses : dict, optional
            Map zone labels to bus names (default: same name).
        links : dict, optional
            Map corridor names to link names (default: same name). A list maps one
            corridor to several links, e.g. a pair of one-way links. Corridors without a
            link are dropped.
        investment_period : int, optional
            Investment period the domain binds in, as in ``add_flow_based``.
        domain : str, optional
            Name of the flow-based domain, as in ``add_flow_based``.

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.add("Bus", ["BE", "DE"])
        >>> n.add("Link", "ALEGrO", bus0="BE", bus1="DE", p_nom=1000, p_min_pu=-1)
        >>> zonal_ptdf = pd.DataFrame({"BE": [0.3], "DE": [-0.2]}, index=["c1"])
        >>> hub_ptdf = pd.DataFrame({"ALBE": [0.4], "ALDE": [-0.4]}, index=["c1"])
        >>> corridors = pd.DataFrame(
        ...     {"bus0": "BE", "bus1": "DE", "hub0": "ALBE", "hub1": "ALDE"},
        ...     index=["ALEGrO"],
        ... )
        >>> n.c.global_constraints.add_flow_based_from_hubs(
        ...     zonal_ptdf, hub_ptdf, corridors, ram=500
        ... )
        >>> n.c.global_constraints.zonal_ptdf.round(3)
        bus      BE   DE  ALEGrO
        name
        c1      0.3 -0.2    -0.3

        """
        buses, links = buses or {}, links or {}
        if unknown := sorted(set(links) - set(corridors.index)):
            msg = (
                f"{unknown} are not corridors of this domain; "
                f"available: {corridors.index.tolist()}."
            )
            raise ValueError(msg)

        n = self.n_save
        zonal_ptdf = zonal_ptdf.rename(columns=buses)
        if missing := sorted(set(zonal_ptdf.columns) - set(n.c.buses.static.index)):
            msg = f"Zones {missing} are not network buses; pass a `buses` mapping."
            raise ValueError(msg)

        cnecs = zonal_ptdf.index.unique(-1)
        if not (clash := cnecs.intersection(self.static.index.unique("name"))).empty:
            msg = (
                f"CNECs {clash.tolist()[:5]} already exist, e.g. from another domain; "
                "rename the existing global constraints first."
            )
            raise ValueError(msg)

        zones = set(zonal_ptdf.columns)
        static = n.c.links.static
        dynamic_eta = n.c.links.dynamic.efficiency.columns
        dropped = []
        targets = {name: links.get(name, name) for name in corridors.index}
        pairs = [
            (name, link)
            for name, target in targets.items()
            for link in ([target] if isinstance(target, str) else target)
        ]
        for name, link in pairs:
            frm, to, hub0, hub1 = corridors.loc[name, ["bus0", "bus1", "hub0", "hub1"]]
            if link not in static.index:
                dropped += [name] if name not in dropped else []
                continue
            if link in n.c.buses.static.index:
                msg = f"Link name {link!r} is also a bus name; rename it via `links`."
                raise ValueError(msg)
            if link in dynamic_eta:
                msg = f"Corridor link {link!r} has a time-varying `efficiency`."
                raise ValueError(msg)
            bus0, bus1 = static.loc[link, ["bus0", "bus1"]]
            ends = {bus0, bus1} & zones
            frm, to = (buses.get(x, x) if pd.notna(x) else None for x in (frm, to))
            to = to or (next(iter(ends)) if len(ends) == 1 else None)
            if to not in (bus0, bus1) or ends != {frm, to} & zones:
                msg = (
                    f"Link {link!r} ({bus0} -> {bus1}) does not fit corridor {name!r} "
                    f"({frm} -> {to}); check the `buses` mapping."
                )
                raise ValueError(msg)
            eta = static.at[link, "efficiency"]
            if to != bus1:
                hub0, hub1 = hub1, hub0
            sides = ((bus0, hub0, -1), (bus1, hub1, eta))  # (bus, hub, incidence)
            if eta != 1 and any(pd.isna(hub) and b in zones for b, hub, _ in sides):
                msg = (
                    f"Corridor {name!r} needs a hub at each zone end for the lossy link "
                    f"{link!r} (efficiency {eta})."
                )
                raise ValueError(msg)
            h = sum(a * hub_ptdf[hub] for _, hub, a in sides if pd.notna(hub))
            shift = sum(a * zonal_ptdf[b] for b, _, a in sides if b in zones)
            zonal_ptdf[link] = h - shift
        if dropped:
            logger.warning(
                "Dropping %d corridor(s) without a network link: %s. Add links of the "
                "same name or pass a `links` mapping.",
                len(dropped),
                dropped,
            )

        self.add_flow_based(zonal_ptdf, ram, investment_period, domain)
