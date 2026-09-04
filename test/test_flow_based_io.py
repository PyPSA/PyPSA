# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the flow-based domain importers (from_eraa, ...)."""

from pathlib import Path

import pandas as pd
import pytest

import pypsa

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
    ]
    header_label = ["FB_ID", "CNEC_ID", *ZONES, "EXT-Z1", "Z1-Z2"]
    data = [
        ["winter1", "c1", 0.4, -0.2, 0.1, 0.9, 0.5],
        ["winter1", "c2", -0.3, 0.5, 0.2, 0.1, 0.3],
        ["winter1", "c3", 0.1, 0.1, 0.1, 0.0, 0.0],  # NaN RAM below -> dropped
        ["summer1", "c1", 0.9, 0.9, 0.9, 0.0, 0.0],  # other season -> ignored
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
    n.add("FlowBasedConstraint", ptdf.index, zonal_ptdf=ptdf, ram=[1000.0, 800.0])
    n.c.flow_based_constraints.static.loc["c2", "active"] = False

    path = tmp_path / ("net.nc" if fmt == "nc" else "csv")
    (n.export_to_netcdf if fmt == "nc" else n.export_to_csv_folder)(str(path))
    m = pypsa.Network(str(path))

    c, cm = n.c.flow_based_constraints, m.c.flow_based_constraints
    pd.testing.assert_frame_equal(cm.zonal_ptdf[c.zonal_ptdf.columns], c.zonal_ptdf)
    pd.testing.assert_series_equal(cm.static["ram"], c.static["ram"])
    pd.testing.assert_series_equal(cm.static["active"], c.static["active"])
    assert not any(col.startswith("zonal_ptdf") for col in cm.static.columns)


def test_from_eraa_parses_zones_season_and_ram(eraa_workbook):
    """Only PTDF_SZ columns and the selected season are read; NaN-RAM CNECs are dropped."""
    n = _network()
    n.c.flow_based_constraints.from_eraa(eraa_workbook, year="2030", season="winter1")
    c = n.c.flow_based_constraints

    assert list(c.static.index) == ["c1", "c2"]  # c3 dropped (NaN RAM), summer ignored
    assert list(c.zonal_ptdf.columns) == ZONES  # AHC column X-Z1 ignored
    assert c.zonal_ptdf.loc["c1", "Z1"] == pytest.approx(0.4)
    assert c.static.ram.to_dict() == {"c1": 1000.0, "c2": 800.0}


def test_from_eraa_bus_mapping(eraa_workbook):
    """An explicit buses mapping renames zone columns to network bus names."""
    n = _network(["ZoneOne", "Z2", "Z3"])
    n.c.flow_based_constraints.from_eraa(
        eraa_workbook, year="2030", season="winter1", buses={"Z1": "ZoneOne"}
    )
    assert list(n.c.flow_based_constraints.zonal_ptdf.columns) == [
        "ZoneOne",
        "Z2",
        "Z3",
    ]


def test_from_eraa_fails_fast_on_unknown_zone(eraa_workbook):
    """A zone that is not a network bus raises instead of being silently mapped."""
    n = _network(["Z1", "Z2"])  # missing Z3
    with pytest.raises(ValueError, match="not network buses"):
        n.c.flow_based_constraints.from_eraa(
            eraa_workbook, year="2030", season="winter1"
        )


def test_from_eraa_unknown_season_raises(eraa_workbook):
    """Selecting a season absent from the RAM sheet fails rather than adding an empty domain."""
    n = _network()
    with pytest.raises(KeyError):
        n.c.flow_based_constraints.from_eraa(
            eraa_workbook, year="2030", season="autumn9"
        )


def test_from_eraa_domain_solves(eraa_workbook):
    """The imported domain feeds straight into the optimisation."""
    n = _network()
    n.add("Load", ZONES, bus=ZONES, p_set=[300.0, 500.0, 200.0])
    n.add("Generator", ZONES, bus=ZONES, p_nom=2000, marginal_cost=[10.0, 50.0, 30.0])
    n.c.flow_based_constraints.from_eraa(eraa_workbook, year="2030", season="winter1")
    n.optimize(log_to_console=False)
    assert n.buses_t.p.iloc[0][ZONES].sum() == pytest.approx(
        0.0
    )  # net positions balance


def test_from_eraa_maps_corridors_to_links(eraa_workbook):
    """AHC/EvFB columns are included as link terms when mapped; others are dropped."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "ahc", bus0="EXT", bus1="Z1", p_nom=100)  # EXT->Z1 matches "EXT-Z1"
    n.add("Link", "ev", bus0="Z1", bus1="Z2", p_nom=100)  # Z1->Z2 matches "Z1-Z2"
    n.c.flow_based_constraints.from_eraa(
        eraa_workbook,
        year="2030",
        season="winter1",
        links={"EXT-Z1": "ahc", "Z1-Z2": "ev"},
    )
    z = n.c.flow_based_constraints.zonal_ptdf
    assert set(z.columns) == {*ZONES, "ahc", "ev"}
    assert z.loc["c1", "ahc"] == pytest.approx(0.9)  # same orientation -> +
    assert z.loc["c1", "ev"] == pytest.approx(0.5)


def test_from_eraa_corridor_orientation_flips_sign(eraa_workbook):
    """A link oriented opposite to the border label flips the PTDF sign."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "rev", bus0="Z1", bus1="EXT", p_nom=100)  # reversed vs "EXT-Z1"
    n.c.flow_based_constraints.from_eraa(
        eraa_workbook, year="2030", season="winter1", links={"EXT-Z1": "rev"}
    )
    assert n.c.flow_based_constraints.zonal_ptdf.loc["c1", "rev"] == pytest.approx(-0.9)


def test_from_eraa_corridor_endpoint_mismatch_raises(eraa_workbook):
    """A mapped link must connect the border's endpoints."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "bad", bus0="Z2", bus1="Z3", p_nom=100)  # not EXT<->Z1
    with pytest.raises(ValueError, match="does not connect"):
        n.c.flow_based_constraints.from_eraa(
            eraa_workbook, year="2030", season="winter1", links={"EXT-Z1": "bad"}
        )


def test_from_eraa_warns_on_dropped_corridors(eraa_workbook, caplog):
    """Unmapped AHC/EvFB corridors are dropped with a warning."""
    n = _network()
    with caplog.at_level("WARNING"):
        n.c.flow_based_constraints.from_eraa(
            eraa_workbook, year="2030", season="winter1"
        )
    assert "unmapped" in caplog.text.lower()


def test_from_eraa_unknown_corridor_raises(eraa_workbook):
    """A links key that is not an AHC/EvFB column fails fast."""
    n = _network([*ZONES, "EXT"])
    n.add("Link", "ahc", bus0="EXT", bus1="Z1", p_nom=100)
    with pytest.raises(ValueError, match="AHC/EvFB column"):
        n.c.flow_based_constraints.from_eraa(
            eraa_workbook, year="2030", season="winter1", links={"NOPE": "ahc"}
        )


def test_from_eraa_time_varying_by_season(eraa_workbook):
    """A snapshot->season Series builds a time-varying domain; unmatched CNECs go inert."""
    n = _network()
    n.set_snapshots([0, 1])
    n.c.flow_based_constraints.from_eraa(
        eraa_workbook, year="2030", season=pd.Series({0: "winter1", 1: "summer1"})
    )
    c = n.c.flow_based_constraints
    z = c.zonal_ptdf
    assert isinstance(z.index, pd.MultiIndex)
    assert sorted(c.static.index) == ["c1", "c2"]  # union of both seasons
    assert z.loc[(0, "c1"), "Z1"] == pytest.approx(0.4)  # winter1 PTDF
    assert z.loc[(1, "c1"), "Z1"] == pytest.approx(0.9)  # summer1 PTDF
    assert z.loc[(1, "c2")].eq(0.0).all()  # c2 absent in summer1 -> zero row
    ram = c.dynamic["ram"]
    assert ram.loc[0, "c2"] == pytest.approx(800.0)
    assert ram.loc[1, "c2"] == float("inf")  # inert that hour


def test_from_eraa_time_varying_solves(eraa_workbook):
    """The time-varying ERAA domain feeds straight into the optimisation."""
    n = _network()
    n.set_snapshots([0, 1])
    n.add("Load", ZONES, bus=ZONES, p_set=[300.0, 500.0, 200.0])
    n.add("Generator", ZONES, bus=ZONES, p_nom=2000, marginal_cost=[10.0, 50.0, 30.0])
    n.c.flow_based_constraints.from_eraa(
        eraa_workbook, year="2030", season=pd.Series({0: "winter1", 1: "summer1"})
    )
    n.optimize(log_to_console=False)
    assert (n.buses_t.p[ZONES].sum(axis=1).abs() < 1e-6).all()  # balances each hour


def test_from_eraa_time_varying_incomplete_mapping_raises(eraa_workbook):
    """A season Series that misses a snapshot fails fast."""
    n = _network()
    n.set_snapshots([0, 1])
    with pytest.raises(ValueError, match="every network snapshot"):
        n.c.flow_based_constraints.from_eraa(
            eraa_workbook, year="2030", season=pd.Series({0: "winter1"})
        )


@pytest.fixture
def jao_csv(tmp_path):
    """A minimal JAO finalComputation CSV: non-unique CneName, one non-presolved row."""
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "CneName": ["line_x", "line_x", "line_y"],  # not unique across directions
            "Direction": ["DIRECT", "OPPOSITE", "DIRECT"],
            "Presolved": [True, True, False],  # row 3 is filtered out by default
            "Ram": [1000.0, 900.0, 500.0],
            "Ptdf_Z1": [0.4, -0.4, 0.1],
            "Ptdf_Z2": [-0.2, 0.2, 0.1],
            "Ptdf_Z3": [0.1, -0.1, 0.1],
        }
    )
    path = tmp_path / "jao.csv"
    df.to_csv(path, sep=";", index=False)
    return str(path)


def test_from_jao_strips_prefix_and_filters_presolved(jao_csv):
    """Ptdf_ is stripped to hub names; only presolved rows are kept; Id is the name."""
    n = _network()
    n.c.flow_based_constraints.from_jao(jao_csv)
    c = n.c.flow_based_constraints

    assert list(c.static.index) == ["1", "2"]  # non-presolved row 3 dropped; Id as name
    assert list(c.zonal_ptdf.columns) == ZONES  # Ptdf_ prefix removed
    assert c.zonal_ptdf.loc["1", "Z1"] == pytest.approx(0.4)
    assert c.static.ram.to_dict() == {"1": 1000.0, "2": 900.0}


def test_from_jao_presolved_false_reads_all(jao_csv):
    """presolved=False keeps every row."""
    n = _network()
    n.c.flow_based_constraints.from_jao(jao_csv, presolved=False)
    assert len(n.c.flow_based_constraints.static) == 3


def test_from_jao_non_unique_name_col_raises(jao_csv):
    """CneName is not unique, so it fails fast rather than silently collapsing rows."""
    n = _network()
    with pytest.raises(ValueError, match="unique"):
        n.c.flow_based_constraints.from_jao(jao_csv, name_col="CneName")


def test_from_jao_bus_mapping(jao_csv):
    """An explicit buses mapping renames hub columns to network bus names."""
    n = _network(["ZoneOne", "Z2", "Z3"])
    n.c.flow_based_constraints.from_jao(jao_csv, buses={"Z1": "ZoneOne"})
    assert list(n.c.flow_based_constraints.zonal_ptdf.columns) == [
        "ZoneOne",
        "Z2",
        "Z3",
    ]


def test_from_jao_links_mapping(jao_csv):
    """A hub can be mapped to a link (external virtual hub); the value is not re-signed."""
    n = pypsa.Network()
    n.add("Bus", ["Z1", "Z2", "X"])
    n.add("Link", "cobra", bus0="Z1", bus1="X", p_nom=100)
    n.c.flow_based_constraints.from_jao(jao_csv, links={"Z3": "cobra"})
    z = n.c.flow_based_constraints.zonal_ptdf
    assert set(z.columns) == {"Z1", "Z2", "cobra"}
    assert z.loc["1", "cobra"] == pytest.approx(0.1)  # renamed, sign unchanged


@pytest.fixture
def tso_domain(tmp_path):
    """A minimal TSO MS_FBMC domain CSV factory (English or German decimals)."""

    def build(decimal="."):
        def n(x):
            return str(x).replace(".", decimal)

        lines = [
            "!DATEITYP;MS_FBMC_Domain_TS*",
            "!!FORMAT_NAME;FORMAT_FLOW_BASED_DOMAIN",
            "!!OBJEKTTYP;FB_RAM;FB_DOMAIN;FB_DOMAIN;FB_DOMAIN_AHC;HGUE",
            "CNEC_ID;RAM_MW;Z1;Z2;EXT;KONV_X",
            f"c1;{n(1000.0)};{n(0.4)};{n(-0.2)};{n(0.1)};{n(0.3)}",
            f"c2;{n(800.0)};{n(0.1)};{n(0.5)};{n(0.2)};{n(-0.1)}",
        ]
        path = tmp_path / f"tso_{decimal!r}.csv"
        path.write_text("\n".join(lines), encoding="latin-1")
        return str(path)

    return build


TSO_ZONES = ["Z1", "Z2", "EXT"]


@pytest.mark.parametrize("decimal", [".", ","])
def test_from_tso_parses_domain(tso_domain, decimal, caplog):
    """OBJEKTTYP types the columns; the decimal locale is auto-detected; converters drop."""
    n = pypsa.Network()
    n.add("Bus", TSO_ZONES)
    with caplog.at_level("WARNING"):
        n.c.flow_based_constraints.from_tso(tso_domain(decimal))
    c = n.c.flow_based_constraints
    assert list(c.static.index) == ["c1", "c2"]
    assert (
        list(c.zonal_ptdf.columns) == TSO_ZONES
    )  # FB_DOMAIN + FB_DOMAIN_AHC; HGUE dropped
    assert c.zonal_ptdf.loc["c1", "Z1"] == pytest.approx(0.4)
    assert c.static.ram.to_dict() == {"c1": 1000.0, "c2": 800.0}
    assert "unmapped" in caplog.text.lower()  # KONV_X dropped with a warning


def test_from_tso_maps_converter(tso_domain):
    """A converter column is included as a link term when mapped."""
    n = pypsa.Network()
    n.add("Bus", TSO_ZONES)
    n.add("Link", "dc", bus0="Z1", bus1="EXT", p_nom=100)
    n.c.flow_based_constraints.from_tso(tso_domain(), links={"KONV_X": "dc"})
    z = n.c.flow_based_constraints.zonal_ptdf
    assert set(z.columns) == {*TSO_ZONES, "dc"}
    assert z.loc["c1", "dc"] == pytest.approx(0.3)


def test_from_tso_unknown_converter_raises(tso_domain):
    """A links key that is not a converter column fails fast."""
    n = pypsa.Network()
    n.add("Bus", TSO_ZONES)
    n.add("Link", "dc", bus0="Z1", bus1="EXT", p_nom=100)
    with pytest.raises(ValueError, match="converter column"):
        n.c.flow_based_constraints.from_tso(tso_domain(), links={"Z1": "dc"})


_REAL_ERAA = Path(__file__).parent / "data" / "fbmc" / "FB-Domain-CORE_simplified.xlsx"
_REAL_JAO = Path(__file__).parent / "data" / "fbmc" / "finalComputation.csv"
_REAL_TSO = Path(__file__).parent / "data" / "fbmc" / "tso_domain.csv"


@pytest.mark.skipif(not _REAL_TSO.exists(), reason="TSO example data not available")
def test_from_tso_real_data():
    """Parse the real (scrambled) TSO domain: 164 CNECs x 15 zones."""
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
        "DKW",
        "LT",
        "BG",
    ]
    n = pypsa.Network()
    n.add("Bus", zones)
    n.c.flow_based_constraints.from_tso(str(_REAL_TSO))
    c = n.c.flow_based_constraints
    assert c.zonal_ptdf.shape == (164, 15)
    assert set(c.zonal_ptdf.columns) == set(zones)
    assert not c.static.ram.isna().any()


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
    n.c.flow_based_constraints.from_eraa(str(_REAL_ERAA), year="2030", season="winter1")
    c = n.c.flow_based_constraints
    assert c.zonal_ptdf.shape == (134, 13)
    assert list(c.zonal_ptdf.columns) == zones
    assert not c.static.ram.isna().any()
    assert c.static.ram.min() > 100
    assert c.static.ram.max() < 5000


@pytest.mark.skipif(not _REAL_JAO.exists(), reason="JAO example data not available")
def test_from_jao_real_data():
    """Reproduce the JAO presolved domain (181 CNECs x 24 hubs)."""
    hubs = [
        c.removeprefix("Ptdf_")
        for c in pd.read_csv(str(_REAL_JAO), sep=";", nrows=1).columns
        if c.startswith("Ptdf_")
    ]
    n = pypsa.Network()
    n.add("Bus", hubs)
    n.c.flow_based_constraints.from_jao(str(_REAL_JAO))
    c = n.c.flow_based_constraints
    assert c.zonal_ptdf.shape == (181, 24)
    assert list(c.zonal_ptdf.columns) == hubs
    assert c.static.index.is_unique
