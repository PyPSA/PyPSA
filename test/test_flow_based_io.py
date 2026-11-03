# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the flow-based domain importers (from_eraa, ...)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pypsa

FB = {"type": "flow_based", "sense": "<="}  # a flow-based global constraint row

ZONES = ["Z1", "Z2", "Z3"]


@pytest.fixture
def eraa_workbook(tmp_path):
    """A minimal ERAA-shaped workbook: two header rows, two seasons, one NaN RAM."""
    pytest.importorskip("openpyxl")
    header_kind = [
        None,
        None,
        "PTDF_SZ",
        "PTDF_SZ",
        "PTDF_SZ",
        "PTDF*_AHC,SZ",
        "PTDF_EvFB",
        "PTDF*_AHC,SZ",
    ]
    header_label = ["FB_ID", "CNEC_ID", *ZONES, "EXT-Z1", "Z1-Z2", "EXT-Z2_1"]
    data = [
        ["winter1", "c1", 0.4, -0.2, 0.1, 0.9, 0.5, 0.3],
        ["winter1", "c2", -0.3, 0.5, 0.2, 0.1, 0.3, -0.1],
        ["winter1", "c3", 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],  # NaN RAM below -> dropped
        ["summer1", "c1", 0.9, 0.9, 0.9, 0.0, 0.0, 0.0],  # other season -> ignored
    ]
    ptdf = pd.DataFrame([header_kind, header_label, *data])
    ram = pd.DataFrame(
        {
            "CNEC_ID": ["c1", "c2", "c3"],
            "winter1": [1000.0, 800.0, None],
            "summer1": [500.0, 500.0, 500.0],
        }
    )
    assignment = pd.DataFrame(
        {
            "Year": [2028, 2030, 2030],
            "Month": 1,
            "Day": 1,
            "Hour": [0, 0, 1],
            "WS01": ["summer1", "winter1", "summer1"],
        }
    )
    path = tmp_path / "eraa.xlsx"
    with pd.ExcelWriter(path) as xl:
        ptdf.to_excel(xl, sheet_name="PTDF 2030", header=False, index=False)
        ram.to_excel(xl, sheet_name="RAM 2030", index=False)
        assignment.to_excel(xl, sheet_name="Domain Assignment", index=False)
    return str(path)


@pytest.fixture
def eraa_nordic_workbook(tmp_path):
    """An ERAA Nordic-shaped workbook: one PTDF for all domain IDs, integer IDs."""
    pytest.importorskip("openpyxl")
    ptdf = pd.DataFrame(
        [
            [None, None, None, "PTDF_SZ", "PTDF_SZ", "PTDF_SZ"],
            ["FB_ID", "CNEC_ID", "CNEC_Name", *ZONES],
            [1, -5, "line a - reversed", 0.4, -0.2, 0.1],
            [1, 3, "line b", -0.3, 0.5, 0.2],
        ]
    )
    ram = pd.DataFrame(
        {
            "CNEC_ID": [-5, 3],
            "CNEC_Name": ["line a - reversed", "line b"],
            0: [1000.0, 800.0],
            1: [600.0, 700.0],
        }
    )
    path = tmp_path / "eraa_nordic.xlsx"
    with pd.ExcelWriter(path) as xl:
        ptdf.to_excel(xl, sheet_name="PTDF 2030", header=False, index=False)
        ram.to_excel(xl, sheet_name="RAM 2030", index=False)
    return str(path)


def _network(buses=ZONES):
    n = pypsa.Network()
    n.add("Bus", buses)
    return n


@pytest.mark.parametrize(("fb_id", "ram"), [(0, [1000.0, 800.0]), (1, [600.0, 700.0])])
def test_from_eraa_nordic_shares_ptdf_across_domain_ids(
    eraa_nordic_workbook, fb_id, ram
):
    """A single PTDF serves every RAM column; integer CNEC IDs become string names."""
    n = _network()
    n.c.global_constraints.flow_based_from_eraa(
        eraa_nordic_workbook, year=2030, fb_id=fb_id
    )
    c = n.c.global_constraints
    assert list(c.static.index) == ["-5", "3"]
    assert c.zonal_ptdf.loc["-5", "Z1"] == pytest.approx(0.4)
    assert c.static.ram.tolist() == ram


@pytest.mark.parametrize(("first", "second"), [(None, None), ("A", "A")])
def test_from_eraa_cnec_name_clash_raises(eraa_workbook, first, second):
    """Importing the same CNEC names twice fails instead of skipping them."""
    n = _network()
    c = n.c.global_constraints
    c.flow_based_from_eraa(eraa_workbook, year="2030", fb_id="winter1", domain=first)
    with pytest.raises(ValueError, match="already exist"):
        c.flow_based_from_eraa(
            eraa_workbook, year="2030", fb_id="winter1", domain=second
        )


def test_from_eraa_domains_prefix_cnec_names(eraa_workbook):
    """Named domains prefix their CNEC names, so equal IDs of two domains coexist."""
    n = _network()
    c = n.c.global_constraints
    for domain in ["A", "B"]:
        c.flow_based_from_eraa(
            eraa_workbook, year="2030", fb_id="winter1", domain=domain
        )
    assert list(c.static.index) == ["A c1", "A c2", "B c1", "B c2"]


def test_from_eraa_time_varying_efficiency_raises(eraa_workbook):
    """A corridor link with time-varying efficiency fails fast."""
    n = _network()
    n.set_snapshots([0, 1])
    n.add("Link", "Z1-Z2", bus0="Z1", bus1="Z2", p_nom=100, efficiency=[0.9, 0.95])
    with pytest.raises(ValueError, match="time-varying `efficiency`"):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year="2030", fb_id="winter1"
        )


def test_from_eraa_parses_zones_season_and_ram(eraa_workbook):
    """Only PTDF_SZ columns and the selected season are read; NaN-RAM CNECs are dropped."""
    n = _network()
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", fb_id="winter1"
    )
    c = n.c.global_constraints

    assert list(c.static.index) == ["c1", "c2"]  # c3 dropped (NaN RAM), summer ignored
    assert list(c.zonal_ptdf.columns) == ZONES  # AHC column X-Z1 ignored
    assert c.zonal_ptdf.loc["c1", "Z1"] == pytest.approx(0.4)
    assert c.static.ram.to_dict() == {"c1": 1000.0, "c2": 800.0}


def test_from_eraa_bus_mapping(eraa_workbook):
    """An explicit buses mapping renames zone columns to network bus names."""
    n = _network(["ZoneOne", "Z2", "Z3"])
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", fb_id="winter1", buses={"Z1": "ZoneOne"}
    )
    assert list(n.c.global_constraints.zonal_ptdf.columns) == [
        "ZoneOne",
        "Z2",
        "Z3",
    ]


def test_from_eraa_fails_fast_on_unknown_zone(eraa_workbook):
    """A zone that is not a network bus raises instead of being silently mapped."""
    n = _network(["Z1", "Z2"])  # missing Z3
    with pytest.raises(ValueError, match="not network buses"):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year="2030", fb_id="winter1"
        )


def test_from_eraa_unknown_season_raises(eraa_workbook):
    """Selecting a season absent from the RAM sheet fails rather than adding an empty domain."""
    n = _network()
    with pytest.raises(KeyError):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year="2030", fb_id="autumn9"
        )


def test_from_eraa_maps_corridors_to_links(eraa_workbook):
    """Corridors go to links of the same name; EvFB columns get the zone PTDFs subtracted."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "EXT-Z1", bus0="EXT", bus1="Z1", p_nom=100)
    n.add("Link", "EXT-Z2_1", bus0="EXT", bus1="Z2", p_nom=100)  # numbered border
    n.add("Link", "Z1-Z2", bus0="Z1", bus1="Z2", p_nom=100)
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", fb_id="winter1"
    )
    z = n.c.global_constraints.zonal_ptdf
    assert set(z.columns) == {*ZONES, "EXT-Z1", "EXT-Z2_1", "Z1-Z2"}
    assert z.loc["c1", "EXT-Z1"] == pytest.approx(0.9)  # PTDF*_AHC as published
    assert z.loc["c2", "EXT-Z2_1"] == pytest.approx(-0.1)  # PTDF*_AHC as published
    assert z.loc["c1", "Z1-Z2"] == pytest.approx(
        0.5 - (-0.2 - 0.4)
    )  # h - (PTDF_Z2 - PTDF_Z1)


@pytest.mark.parametrize(
    ("border", "bus0", "bus1", "expected"),
    [("EXT-Z1", "Z1", "EXT", -0.9), ("Z1-Z2", "Z2", "Z1", -1.1)],
)
def test_from_eraa_corridor_orientation_flips_sign(
    eraa_workbook, border, bus0, bus1, expected
):
    """A link oriented opposite to the border label flips the column sign."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "rev", bus0=bus0, bus1=bus1, p_nom=100)
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", fb_id="winter1", links={border: "rev"}
    )
    z = n.c.global_constraints.zonal_ptdf
    assert z.loc["c1", "rev"] == pytest.approx(expected)


def test_from_eraa_corridor_endpoint_mismatch_raises(eraa_workbook):
    """A mapped link must connect the border's flow-based zone."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "bad", bus0="Z2", bus1="Z3", p_nom=100)  # not EXT<->Z1
    with pytest.raises(ValueError, match="does not fit"):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year="2030", fb_id="winter1", links={"EXT-Z1": "bad"}
        )


def test_from_eraa_unknown_corridor_raises(eraa_workbook):
    """A links key that is not a corridor fails fast."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "ahc", bus0="EXT", bus1="Z1", p_nom=100)
    with pytest.raises(ValueError, match="not corridors"):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year="2030", fb_id="winter1", links={"NOPE": "ahc"}
        )


@pytest.mark.parametrize(
    ("period", "names"), [(None, ["c1", "c2"]), (2030, ["c1 2030", "c2 2030"])]
)
def test_from_eraa_investment_period(eraa_workbook, period, names):
    """By default the domain applies in all periods; a period ties (and suffixes) its rows."""
    n = _network()
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", fb_id="winter1", investment_period=period
    )
    static = n.c.global_constraints.static
    assert list(static.index) == names
    assert list(n.c.global_constraints.zonal_ptdf.index) == names
    assert static["investment_period"].isna().all() == (period is None)


def test_from_eraa_time_varying_by_season(eraa_workbook):
    """A snapshot->season Series builds a time-varying domain; unmatched CNECs are dropped."""
    n = _network()
    n.set_snapshots([0, 1])
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", fb_id=pd.Series({0: "winter1", 1: "summer1"})
    )
    c = n.c.global_constraints
    z = c.zonal_ptdf
    assert isinstance(z.index, pd.MultiIndex)
    assert sorted(c.static.index) == ["c1", "c2"]  # union of both seasons
    assert z.loc[(0, "c1"), "Z1"] == pytest.approx(0.4)  # winter1 PTDF
    assert z.loc[(1, "c1"), "Z1"] == pytest.approx(0.9)  # summer1 PTDF
    assert z.loc[(1, "c2")].eq(0.0).all()  # c2 absent in summer1 -> zero row
    ram = c.dynamic["ram"]
    assert ram.loc[0, "c2"] == pytest.approx(800.0)
    assert pd.isna(ram.loc[1, "c2"])  # dropped that hour


def test_from_eraa_weather_scenario_assigns_domains(eraa_workbook):
    """A weather scenario maps each snapshot to its domain ID by month, day and hour."""
    n = _network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=2, freq="h"))
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year=2030, weather_scenario="WS01"
    )
    ram = n.c.global_constraints.dynamic["ram"]
    assert ram["c1"].tolist() == [1000.0, 500.0]  # winter1, then summer1


@pytest.mark.parametrize(
    ("snapshots", "kwargs", "error"),
    [
        (pd.date_range("2013-01-01", periods=3, freq="h"), {}, ValueError),
        ([0, 1], {}, TypeError),
        ([0, 1], {"fb_id": "winter1"}, ValueError),
    ],
)
def test_from_eraa_weather_scenario_raises(eraa_workbook, snapshots, kwargs, error):
    """A snapshot without assignment, non-datetime snapshots or two selectors fail."""
    n = _network()
    n.set_snapshots(snapshots)
    with pytest.raises(error):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year=2030, weather_scenario="WS01", **kwargs
        )


def test_from_eraa_time_varying_incomplete_mapping_raises(eraa_workbook):
    """A domain ID Series that misses a snapshot fails fast."""
    n = _network()
    n.set_snapshots([0, 1])
    with pytest.raises(ValueError, match="every network snapshot"):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year="2030", fb_id=pd.Series({0: "winter1"})
        )


@pytest.fixture
def jao_csv(tmp_path):
    """A minimal JAO finalComputation CSV with one non-presolved row."""
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "Direction": ["DIRECT", "OPPOSITE", "DIRECT"],
            "Presolved": [True, True, False],  # row 3 is filtered out by default
            "Ram": [1000.0, 900.0, 500.0],
            "Ptdf_ALBE": [0.3, -0.3, 0.0],  # ALEGrO end in BE
            "Ptdf_ALDE": [0.1, -0.1, 0.0],  # ALEGrO end in DE
            "Ptdf_Z1": [0.4, -0.4, 0.1],
            "Ptdf_Z2": [-0.2, 0.2, 0.1],
            "Ptdf_Z3": [0.1, -0.1, 0.1],
            "Ptdf_Z1_X_Cable": [0.25, -0.25, 0.0],  # AHC hub
            "Ptdf_CH": [0.05, -0.05, 0.0],  # published for transparency only
        }
    )
    path = tmp_path / "jao.csv"
    df.to_csv(path, sep=";", index=False)
    return str(path)


def test_from_jao_strips_prefix_and_filters_presolved(jao_csv, caplog):
    """Ptdf_ is stripped to hub names; only presolved rows are kept; Id is the name."""
    n = _network()
    with caplog.at_level("WARNING"):
        n.c.global_constraints.flow_based_from_jao(jao_csv)
    c = n.c.global_constraints

    assert list(c.static.index) == ["1", "2"]  # non-presolved row 3 dropped; Id as name
    assert list(c.zonal_ptdf.columns) == ZONES  # prefix removed; corridors, CH dropped
    assert c.zonal_ptdf.loc["1", "Z1"] == pytest.approx(0.4)
    assert c.static.ram.to_dict() == {"1": 1000.0, "2": 900.0}
    assert "without a network link" in caplog.text


@pytest.mark.parametrize(
    ("bus0", "bus1", "expected"), [("X", "Z1", 0.25 - 0.4), ("Z1", "X", 0.4 - 0.25)]
)
def test_from_jao_ahc_hub_sign_follows_link(jao_csv, bus0, bus1, expected):
    """An AHC hub column becomes h - PTDF_zone, positive for power entering its zone."""
    n = _network([*ZONES, "X"])
    n.add("Link", "Z1_X_Cable", bus0=bus0, bus1=bus1, p_nom=100)
    n.c.global_constraints.flow_based_from_jao(jao_csv)
    z = n.c.global_constraints.zonal_ptdf
    assert set(z.columns) == {*ZONES, "Z1_X_Cable"}
    assert z.loc["1", "Z1_X_Cable"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("bus0", "bus1", "expected"),
    [("Z1", "Z2", (0.1 - 0.3) - (-0.2 - 0.4)), ("Z2", "Z1", (0.3 - 0.1) - (0.4 + 0.2))],
)
def test_from_jao_evfb_hubs_contract_to_one_link(jao_csv, bus0, bus1, expected):
    """ALBE and ALDE contract to one column: (h_DE - h_BE) - (PTDF_DE - PTDF_BE)."""
    n = _network()
    n.add("Link", "ALBE-ALDE", bus0=bus0, bus1=bus1, p_nom=1000)
    n.c.global_constraints.flow_based_from_jao(jao_csv, buses={"BE": "Z1", "DE": "Z2"})
    z = n.c.global_constraints.zonal_ptdf
    assert set(z.columns) == {*ZONES, "ALBE-ALDE"}
    assert z.loc["1", "ALBE-ALDE"] == pytest.approx(expected)


@pytest.fixture
def tso_domain(tmp_path):
    """A minimal TSO MS_FBMC domain CSV factory (English or German decimals)."""

    def build(decimal="."):
        def n(x):
            return str(x).replace(".", decimal)

        lines = [
            "!DATEITYP;MS_FBMC_Domain_TS*",
            "!!FORMAT_NAME;FORMAT_FLOW_BASED_DOMAIN",
            "!!OBJEKTTYP;FB_RAM;FB_DOMAIN;FB_DOMAIN;FB_DOMAIN_AHC;HGUE;HGUE;HGUE_AHC",
            "CNEC_ID;RAM_MW;Z1;Z2;EXT;KONV_Z1-Z21_Z1;KONV_Z1-Z21_Z2;KONV_AHC_Z1-EXT_Z1",
            f"c1;{n(1000.0)};{n(0.4)};{n(-0.2)};{n(0.1)};{n(0.3)};{n(-0.1)};{n(0.25)}",
            f"c2;{n(800.0)};{n(0.1)};{n(0.5)};{n(0.2)};{n(-0.1)};{n(0.4)};{n(0.15)}",
        ]
        path = tmp_path / f"tso_{decimal!r}.csv"
        path.write_text("\n".join(lines), encoding="latin-1")
        return str(path)

    return build


TSO_ZONES = ["Z1", "Z2"]


@pytest.mark.parametrize("decimal", [".", ","])
def test_from_tso_parses_domain(tso_domain, decimal, caplog):
    """OBJEKTTYP types the columns; the decimal locale is auto-detected; corridors drop."""
    n = pypsa.Network()
    n.add("Bus", TSO_ZONES)
    with caplog.at_level("WARNING"):
        n.c.global_constraints.flow_based_from_tso(tso_domain(decimal))
    c = n.c.global_constraints
    assert list(c.static.index) == ["c1", "c2"]
    assert list(c.zonal_ptdf.columns) == TSO_ZONES  # AHC zone EXT is not a zone
    assert c.zonal_ptdf.loc["c1", "Z1"] == pytest.approx(0.4)
    assert c.static.ram.to_dict() == {"c1": 1000.0, "c2": 800.0}
    assert "without a network link" in caplog.text


@pytest.mark.parametrize(
    ("corridor", "bus0", "bus1", "expected"),
    [
        ("KONV_AHC_Z1-EXT", "EXT", "Z1", 0.25 - 0.4),
        ("KONV_AHC_Z1-EXT", "Z1", "EXT", 0.4 - 0.25),
        ("EXT", "EXT", "Z1", 0.1 - 0.4),  # AC exchange of the AHC zone
    ],
)
def test_from_tso_ahc_sign_follows_link(tso_domain, corridor, bus0, bus1, expected):
    """An AHC column becomes h - PTDF_zone, positive for power entering the region."""
    n = pypsa.Network()
    n.add("Bus", [*TSO_ZONES, "EXT"])
    n.add("Link", "dc", bus0=bus0, bus1=bus1, p_nom=100)
    n.c.global_constraints.flow_based_from_tso(tso_domain(), links={corridor: "dc"})
    z = n.c.global_constraints.zonal_ptdf
    assert set(z.columns) == {*TSO_ZONES, "dc"}
    assert z.loc["c1", "dc"] == pytest.approx(expected)


def test_from_tso_corridor_maps_to_one_way_link_pair(tso_domain):
    """A corridor mapped to two one-way links gives each the column in its direction."""
    n = pypsa.Network()
    n.add("Bus", [*TSO_ZONES, "EXT"])
    n.add(
        "Link", ["EXT-Z1", "Z1-EXT"], bus0=["EXT", "Z1"], bus1=["Z1", "EXT"], p_nom=100
    )
    n.c.global_constraints.flow_based_from_tso(
        tso_domain(), links={"EXT": ["EXT-Z1", "Z1-EXT"]}
    )
    z = n.c.global_constraints.zonal_ptdf
    assert z.loc["c1", "EXT-Z1"] == pytest.approx(0.1 - 0.4)
    assert z.loc["c1", "Z1-EXT"] == pytest.approx(0.4 - 0.1)


@pytest.mark.parametrize(
    ("bus0", "bus1", "expected"),
    [
        ("Z1", "Z2", (-0.1 - 0.3) - (-0.2 - 0.4)),
        ("Z2", "Z1", (0.3 + 0.1) - (0.4 + 0.2)),
    ],
)
def test_from_tso_evfb_converters_contract_to_one_link(
    tso_domain, bus0, bus1, expected
):
    """Both converters contract to one column: (h_Z2 - h_Z1) - (PTDF_Z2 - PTDF_Z1)."""
    n = pypsa.Network()
    n.add("Bus", TSO_ZONES)
    n.add("Link", "KONV_Z1-Z21", bus0=bus0, bus1=bus1, p_nom=100)
    n.c.global_constraints.flow_based_from_tso(tso_domain())
    z = n.c.global_constraints.zonal_ptdf
    assert z.loc["c1", "KONV_Z1-Z21"] == pytest.approx(expected)


def test_from_tso_ahc_zone_name_clash_raises(tso_domain):
    """An AC-AHC corridor named like its zone bus needs a distinct link name."""
    n = pypsa.Network()
    n.add("Bus", [*TSO_ZONES, "EXT"])
    n.add("Link", "EXT", bus0="EXT", bus1="Z1", p_nom=100)
    with pytest.raises(ValueError, match="also a bus name"):
        n.c.global_constraints.flow_based_from_tso(tso_domain())


# Two domains: D1 (zones A, B, C) and D2 (zones X, Y). Corridors: EvFB A -> B, AHC from
# the outside bus E into C, and C -> X between the domains. Hubs are sensitivities to
# 1 MW injected at a corridor end.
_VH_DOMAINS = {
    "D1": (
        pd.DataFrame(
            {"A": [0.3, -0.2, 0.1], "B": [-0.1, 0.4, 0.0], "C": [0.0, -0.3, 0.2]},
            index=["k1", "k2", "k3"],
        ),
        pd.DataFrame(
            {
                "hA": [0.35, -0.1, 0.05],
                "hB": [-0.2, 0.3, 0.1],
                "hCE": [0.1, -0.4, 0.25],
                "hCX": [0.05, -0.2, 0.3],
            },
            index=["k1", "k2", "k3"],
        ),
        pd.DataFrame(
            {
                "bus0": ["A", "E", "C"],
                "bus1": ["B", "C", "X"],
                "hub0": ["hA", None, "hCX"],
                "hub1": ["hB", "hCE", None],
            },
            index=["AB", "EC", "CX"],
        ),
        pd.Series([40.0, 30.0, 50.0], index=["k1", "k2", "k3"]),
    ),
    "D2": (
        pd.DataFrame({"X": [0.5], "Y": [-0.5]}, index=["k4"]),
        pd.DataFrame({"hX": [0.2]}, index=["k4"]),
        pd.DataFrame(
            {"bus0": ["C"], "bus1": ["X"], "hub0": [None], "hub1": ["hX"]},
            index=["CX"],
        ),
        pd.Series([20.0], index=["k4"]),
    ),
}
_VH_LOAD = {"A": 300, "B": 500, "C": 200, "X": 250, "Y": 150}
_VH_COST = {"A": 10, "B": 50, "C": 30, "E": 5, "X": 60, "Y": 20}


def _vh_network(eta):
    n = pypsa.Network()
    n.add("Bus", [*_VH_COST])
    n.add("Load", list(_VH_LOAD), bus=list(_VH_LOAD), p_set=list(_VH_LOAD.values()))
    n.add(
        "Generator",
        list(_VH_COST),
        bus=list(_VH_COST),
        p_nom=1000,
        marginal_cost=list(_VH_COST.values()),
    )
    corridors = _VH_DOMAINS["D1"][2]
    n.add(
        "Link",
        corridors.index,
        bus0=corridors.bus0,
        bus1=corridors.bus1,
        p_nom=2000,
        efficiency=eta,
    )
    return n


def _solve_virtual_hub(eta):
    """Reference: net positions are generation minus load; hubs carry the link flows."""
    n = _vh_network(eta)
    zones = list(_VH_LOAD)
    n.add("Generator", [f"np {z}" for z in zones], bus=zones, p_nom=1e4, p_min_pu=-1)
    m = n.optimize.create_model(include_objective_constant=False)
    g, f = m.variables["Generator-p"], m.variables["Link-p"]
    for name, (ptdf, hubs, corridors, ram) in _VH_DOMAINS.items():
        np_ = {z: g.sel(name=z) - _VH_LOAD[z] for z in ptdf}
        inj = {}
        for link, row in corridors.iterrows():
            for hub, a in ((row.hub0, -1), (row.hub1, eta)):
                if pd.notna(hub):
                    inj[hub] = a * f.sel(name=link)
        for k in ptdf.index:
            lhs = sum(ptdf.at[k, z] * np_[z] for z in np_)
            lhs += sum(hubs.at[k, h] * inj[h] for h in inj)
            m.add_constraints(lhs <= ram[k], name=f"vh-{k}")
        balance = sum(np_.values()) + sum(inj.values())
        m.add_constraints(balance == 0, name=f"vh-balance-{name}")
    n.optimize.solve_model(solver_name="highs")
    return n


@pytest.mark.parametrize("eta", [1.0, 0.9])
def test_from_hubs_matches_virtual_hub_formulation(eta):
    """Link columns from hubs give the same optimum as the published virtual hubs."""
    n = _vh_network(eta)
    for name, (ptdf, hubs, corridors, ram) in _VH_DOMAINS.items():
        n.c.global_constraints.add_flow_based_from_hubs(
            ptdf, hubs, corridors, ram, domain=name
        )
    n.optimize(solver_name="highs", include_objective_constant=False)
    ref = _solve_virtual_hub(eta)

    assert n.objective == pytest.approx(ref.objective, rel=1e-6)
    pd.testing.assert_frame_equal(n.links_t.p0, ref.links_t.p0, atol=1e-4)
    binding = n.global_constraints_t.mu_ram.iloc[0].abs() > 1e-6
    assert binding[["k1", "k4"]].all()  # binding CNECs in both domains
    assert (n.links_t.p0.iloc[0] < 2000).all()  # no link at its capacity


def test_from_hubs_lossy_link_needs_both_hubs():
    """A lossy link whose zone end has no hub cannot be converted exactly."""
    n = _vh_network(0.9)
    ptdf, hubs, corridors, ram = _VH_DOMAINS["D1"]
    corridors = corridors.assign(hub0=None)  # e.g. ERAA's EvFB hub difference
    with pytest.raises(ValueError, match="needs a hub at each zone end"):
        n.c.global_constraints.add_flow_based_from_hubs(ptdf, hubs, corridors, ram)


_DATA = Path(__file__).parent / "data" / "flow_based"
_REAL_TSO = Path(__file__).parent / "data" / "fbmc" / "tso_domain.csv"


@pytest.mark.parametrize(
    ("importer", "fixture", "kwargs"),
    [
        (
            "flow_based_from_eraa",
            "eraa_workbook",
            {"year": "2030", "fb_id": "winter1"},
        ),
        ("flow_based_from_jao", "jao_csv", {}),
        ("flow_based_from_tso", "tso_domain", {}),
    ],
)
def test_importers_name_the_domain(request, importer, fixture, kwargs):
    """Every importer passes `domain` on to the rows it adds."""
    path = request.getfixturevalue(fixture)
    path = path(".") if callable(path) else path
    n = _network()
    getattr(n.c.global_constraints, importer)(path, domain="Core", **kwargs)
    assert n.c.global_constraints.static.domain.eq("Core").all()


@pytest.mark.skipif(not _REAL_TSO.exists(), reason="TSO example data not available")
def test_from_tso_real_data():
    """Parse the real (scrambled) TSO domain: 164 CNECs x 12 zones."""
    zones = [
        "CZ",
        "NL",
        "AT",
        "PL",
        "HR",
        "FR",
        "BE",
        "SI",
        "SK",
        "RO",
        "HU",
        "DE",
    ]
    n = pypsa.Network()
    n.add("Bus", zones)
    n.c.global_constraints.flow_based_from_tso(str(_REAL_TSO))
    c = n.c.global_constraints
    assert c.zonal_ptdf.shape == (164, 12)
    assert set(c.zonal_ptdf.columns) == set(zones)
    assert not c.static.ram.isna().any()


def _eraa_zones(path):
    head = pd.read_excel(path, sheet_name="PTDF 2030", header=None, nrows=2)
    return head.iloc[1][head.iloc[0] == "PTDF_SZ"].tolist()


def test_from_eraa_core_and_nordic_by_weather_scenario():
    """Trimmed ERAA 2025 Core and Nordic: time-varying, two domains, one shared link."""
    pytest.importorskip("openpyxl")
    core, nordic = (
        _DATA / "FB-Domain-CORE_2030.xlsx",
        _DATA / "FB-Domain-NORDIC_2030.xlsx",
    )
    zones = {"Core": _eraa_zones(core), "Nordic": _eraa_zones(nordic)}
    buses = zones["Core"] + zones["Nordic"]
    rng = np.random.default_rng(0)
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=48, freq="h"))
    n.add("Bus", buses)
    n.add("Load", buses, bus=buses, p_set=rng.uniform(2e3, 3e4, len(buses)))
    n.add(
        "Generator",
        buses,
        bus=buses,
        p_nom=1e5,
        marginal_cost=rng.uniform(10, 100, len(buses)),
    )
    for link in ["DKE1-DE00", "BE00-DE00", "FI00-SE03", "SE03-SE04"]:
        bus0, bus1 = link.split("-")
        n.add("Link", link, bus0=bus0, bus1=bus1, p_nom=1000, p_min_pu=-1)
    c = n.c.global_constraints
    for domain, path, links in [
        ("Core", core, None),
        ("Nordic", nordic, {"DE00-DKE1": "DKE1-DE00"}),
    ]:
        c.flow_based_from_eraa(
            path, year=2030, weather_scenario="WS01", domain=domain, links=links
        )
    status, _ = n.optimize(solver_name="highs", include_objective_constant=False)

    assert status == "ok"
    domain = c.static.domain
    assert domain.value_counts().to_dict() == {"Core": 12, "Nordic": 12}
    ram = c.dynamic["ram"]
    assert ram.nunique().gt(1).groupby(domain).any().all()  # RAM changes over time
    p = n.buses_t.p
    for zs in zones.values():
        assert p[zs].sum(axis=1).abs().max() < 1e-6  # zero-sum per domain
    ptdf = c.zonal_ptdf
    flows = pd.concat([p, n.links_t.p0], axis=1).reindex(
        index=ptdf.index.get_level_values("snapshot"), columns=ptdf.columns
    )
    loading = pd.Series((ptdf.to_numpy() * flows.to_numpy()).sum(axis=1), ptdf.index)
    assert (loading.unstack() - ram).max().max() < 1e-6
    assert (n.global_constraints_t.mu_ram.abs() > 1e-6).any().any()


def test_from_jao_trimmed_data():
    """Trimmed JAO day-ahead domain: presolved rows only, ALEGrO as one link."""
    path = _DATA / "jao_final_computation.csv"
    raw = pd.read_csv(path, sep=";")
    hubs = [c.removeprefix("Ptdf_") for c in raw.columns if c.startswith("Ptdf_")]
    zones = [h for h in hubs if "_" not in h and h not in ("ALBE", "ALDE", "CH")]
    n = pypsa.Network()
    n.add("Bus", zones)
    n.add("Link", "ALBE-ALDE", bus0="BE", bus1="DE", p_nom=1000)
    n.c.global_constraints.flow_based_from_jao(str(path))
    c = n.c.global_constraints
    assert c.zonal_ptdf.shape == (9, 13)
    assert list(c.zonal_ptdf.columns) == [*zones, "ALBE-ALDE"]
    rows = raw[raw["Presolved"]].set_index(raw[raw["Presolved"]]["Id"].astype(str))
    expected = (rows["Ptdf_ALDE"] - rows["Ptdf_ALBE"]) - (
        rows["Ptdf_DE"] - rows["Ptdf_BE"]
    )
    pd.testing.assert_series_equal(
        c.zonal_ptdf["ALBE-ALDE"], expected, check_names=False, check_index_type=False
    )
