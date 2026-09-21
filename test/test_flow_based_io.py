# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the flow-based domain importers (from_eraa, ...)."""

from pathlib import Path

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
    path = tmp_path / "eraa.xlsx"
    with pd.ExcelWriter(path) as xl:
        ptdf.to_excel(xl, sheet_name="PTDF 2030", header=False, index=False)
        ram.to_excel(xl, sheet_name="RAM 2030", index=False)
    return str(path)


def _network(buses=ZONES):
    n = pypsa.Network()
    n.add("Bus", buses)
    return n


@pytest.mark.parametrize("fmt", ["nc", "csv"])
def test_round_trip_preserves_domain(tmp_path, fmt):
    """Export/import preserves the static frame and the zonal PTDF matrix (incl. links)."""
    n = _network([*ZONES, "X"])
    n.add("Link", "ev", bus0="Z1", bus1="Z2", p_nom=500)
    ptdf = pd.DataFrame(
        {"Z1": [0.4, 0.1], "Z2": [-0.2, 0.3], "Z3": [0.0, 0.0], "ev": [0.2, -0.1]},
        index=["c1", "c2"],
    )
    n.c.global_constraints.add_flow_based(ptdf, pd.Series([1000.0, 800.0], ptdf.index))

    path = tmp_path / ("net.nc" if fmt == "nc" else "csv")
    (n.export_to_netcdf if fmt == "nc" else n.export_to_csv_folder)(str(path))
    m = pypsa.Network(str(path))

    c, cm = n.c.global_constraints, m.c.global_constraints
    pd.testing.assert_frame_equal(cm.zonal_ptdf[c.zonal_ptdf.columns], c.zonal_ptdf)
    pd.testing.assert_series_equal(cm.static["constant"], c.static["constant"])


def test_from_eraa_parses_zones_season_and_ram(eraa_workbook):
    """Only PTDF_SZ columns and the selected season are read; NaN-RAM CNECs are dropped."""
    n = _network()
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", season="winter1"
    )
    c = n.c.global_constraints

    assert list(c.static.index) == ["c1", "c2"]  # c3 dropped (NaN RAM), summer ignored
    assert list(c.zonal_ptdf.columns) == ZONES  # AHC column X-Z1 ignored
    assert c.zonal_ptdf.loc["c1", "Z1"] == pytest.approx(0.4)
    assert c.static.constant.to_dict() == {"c1": 1000.0, "c2": 800.0}


def test_from_eraa_bus_mapping(eraa_workbook):
    """An explicit buses mapping renames zone columns to network bus names."""
    n = _network(["ZoneOne", "Z2", "Z3"])
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", season="winter1", buses={"Z1": "ZoneOne"}
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
            eraa_workbook, year="2030", season="winter1"
        )


def test_from_eraa_unknown_season_raises(eraa_workbook):
    """Selecting a season absent from the RAM sheet fails rather than adding an empty domain."""
    n = _network()
    with pytest.raises(KeyError):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year="2030", season="autumn9"
        )


def test_from_eraa_maps_corridors_to_links(eraa_workbook):
    """Corridors go to links of the same name; EvFB columns get the zone PTDFs subtracted."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "EXT-Z1", bus0="EXT", bus1="Z1", p_nom=100)
    n.add("Link", "EXT-Z2_1", bus0="EXT", bus1="Z2", p_nom=100)  # numbered border
    n.add("Link", "Z1-Z2", bus0="Z1", bus1="Z2", p_nom=100)
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", season="winter1"
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
        eraa_workbook, year="2030", season="winter1", links={border: "rev"}
    )
    z = n.c.global_constraints.zonal_ptdf
    assert z.loc["c1", "rev"] == pytest.approx(expected)


def test_from_eraa_corridor_endpoint_mismatch_raises(eraa_workbook):
    """A mapped link must connect the border's flow-based zone."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "bad", bus0="Z2", bus1="Z3", p_nom=100)  # not EXT<->Z1
    with pytest.raises(ValueError, match="does not fit"):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year="2030", season="winter1", links={"EXT-Z1": "bad"}
        )


def test_from_eraa_unknown_corridor_raises(eraa_workbook):
    """A links key that is not a corridor fails fast."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "ahc", bus0="EXT", bus1="Z1", p_nom=100)
    with pytest.raises(ValueError, match="not corridors"):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year="2030", season="winter1", links={"NOPE": "ahc"}
        )


@pytest.mark.parametrize(
    ("period", "names"), [(None, ["c1", "c2"]), (2030, ["c1 2030", "c2 2030"])]
)
def test_from_eraa_investment_period(eraa_workbook, period, names):
    """By default the domain applies in all periods; a period ties (and suffixes) its rows."""
    n = _network()
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", season="winter1", investment_period=period
    )
    static = n.c.global_constraints.static
    assert list(static.index) == names
    assert list(n.c.global_constraints.zonal_ptdf.index) == names
    assert static["investment_period"].isna().all() == (period is None)


def test_from_eraa_time_varying_by_season(eraa_workbook):
    """A snapshot->season Series builds a time-varying domain; unmatched CNECs go inert."""
    n = _network()
    n.set_snapshots([0, 1])
    n.c.global_constraints.flow_based_from_eraa(
        eraa_workbook, year="2030", season=pd.Series({0: "winter1", 1: "summer1"})
    )
    c = n.c.global_constraints
    z = c.zonal_ptdf
    assert isinstance(z.index, pd.MultiIndex)
    assert sorted(c.static.index) == ["c1", "c2"]  # union of both seasons
    assert z.loc[(0, "c1"), "Z1"] == pytest.approx(0.4)  # winter1 PTDF
    assert z.loc[(1, "c1"), "Z1"] == pytest.approx(0.9)  # summer1 PTDF
    assert z.loc[(1, "c2")].eq(0.0).all()  # c2 absent in summer1 -> zero row
    ram = c.dynamic["constant"]
    assert ram.loc[0, "c2"] == pytest.approx(800.0)
    assert ram.loc[1, "c2"] == float("inf")  # inert that hour


def test_from_eraa_time_varying_incomplete_mapping_raises(eraa_workbook):
    """A season Series that misses a snapshot fails fast."""
    n = _network()
    n.set_snapshots([0, 1])
    with pytest.raises(ValueError, match="every network snapshot"):
        n.c.global_constraints.flow_based_from_eraa(
            eraa_workbook, year="2030", season=pd.Series({0: "winter1"})
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
    assert c.static.constant.to_dict() == {"1": 1000.0, "2": 900.0}
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
    assert c.static.constant.to_dict() == {"c1": 1000.0, "c2": 800.0}
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


_REAL_ERAA = Path(__file__).parent / "data" / "fbmc" / "FB-Domain-CORE_simplified.xlsx"
_REAL_JAO = Path(__file__).parent / "data" / "fbmc" / "finalComputation.csv"
_REAL_TSO = Path(__file__).parent / "data" / "fbmc" / "tso_domain.csv"


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
    assert not c.static.constant.isna().any()


@pytest.mark.skipif(not _REAL_ERAA.exists(), reason="ERAA example data not available")
def test_from_eraa_real_data():
    """Reproduce the ERAA Core 2030 winter1 domain (134 CNECs x 13 zones)."""
    pytest.importorskip("openpyxl")
    zones = [
        "AT00",
        "BE00",
        "CZ00",
        "DE00",
        "FR00",
        "HR00",
        "HU00",
        "ITN1",
        "NL00",
        "PL00",
        "RO00",
        "SI00",
        "SK00",
    ]
    n = pypsa.Network()
    n.add("Bus", zones)
    n.c.global_constraints.flow_based_from_eraa(
        str(_REAL_ERAA), year="2030", season="winter1"
    )
    c = n.c.global_constraints
    assert c.zonal_ptdf.shape == (134, 13)
    assert list(c.zonal_ptdf.columns) == zones
    assert not c.static.constant.isna().any()
    assert c.static.constant.min() > 100
    assert c.static.constant.max() < 5000


@pytest.mark.skipif(not _REAL_JAO.exists(), reason="JAO example data not available")
def test_from_jao_real_data():
    """Reproduce the JAO presolved domain (181 CNECs x 12 zones) with ALEGrO as one link."""
    raw = pd.read_csv(str(_REAL_JAO), sep=";", low_memory=False)
    hubs = [c.removeprefix("Ptdf_") for c in raw.columns if c.startswith("Ptdf_")]
    zones = [h for h in hubs if "_" not in h and h not in ("ALBE", "ALDE", "CH")]
    n = pypsa.Network()
    n.add("Bus", zones)
    n.add("Link", "ALBE-ALDE", bus0="BE", bus1="DE", p_nom=1000)
    n.c.global_constraints.flow_based_from_jao(str(_REAL_JAO))
    c = n.c.global_constraints
    assert c.zonal_ptdf.shape == (181, 13)
    assert list(c.zonal_ptdf.columns) == [*zones, "ALBE-ALDE"]
    assert c.static.index.is_unique
    rows = raw[raw["Presolved"]].set_index(raw[raw["Presolved"]]["Id"].astype(str))
    expected = (rows["Ptdf_ALDE"] - rows["Ptdf_ALBE"]) - (
        rows["Ptdf_DE"] - rows["Ptdf_BE"]
    )
    pd.testing.assert_series_equal(
        c.zonal_ptdf["ALBE-ALDE"], expected, check_names=False, check_index_type=False
    )
