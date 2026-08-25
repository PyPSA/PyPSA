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
    header_kind = [None, None, "PTDF_SZ", "PTDF_SZ", "PTDF_SZ", "PTDF*_AHC,SZ"]
    header_label = ["FB_ID", "CNEC_ID", *ZONES, "X-Z1"]
    data = [
        ["winter1", "c1", 0.4, -0.2, 0.1, 0.9],
        ["winter1", "c2", -0.3, 0.5, 0.2, 0.1],
        ["winter1", "c3", 0.1, 0.1, 0.1, 0.0],  # NaN RAM below -> dropped
        ["summer1", "c1", 0.9, 0.9, 0.9, 0.0],  # other season -> ignored
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


def test_from_eraa_parses_zones_season_and_ram(eraa_workbook):
    """Only PTDF_SZ columns and the selected season are read; NaN-RAM CNECs are dropped."""
    n = _network()
    n.c.flow_based_domains.from_eraa(eraa_workbook, year="2030", season="winter1")
    c = n.c.flow_based_domains

    assert list(c.static.index) == ["c1", "c2"]  # c3 dropped (NaN RAM), summer ignored
    assert list(c.zonal_ptdf.columns) == ZONES  # AHC column X-Z1 ignored
    assert c.zonal_ptdf.loc["c1", "Z1"] == pytest.approx(0.4)
    assert c.static.ram.to_dict() == {"c1": 1000.0, "c2": 800.0}


def test_from_eraa_bus_mapping(eraa_workbook):
    """An explicit buses mapping renames zone columns to network bus names."""
    n = _network(["ZoneOne", "Z2", "Z3"])
    n.c.flow_based_domains.from_eraa(
        eraa_workbook, year="2030", season="winter1", buses={"Z1": "ZoneOne"}
    )
    assert list(n.c.flow_based_domains.zonal_ptdf.columns) == ["ZoneOne", "Z2", "Z3"]


def test_from_eraa_fails_fast_on_unknown_zone(eraa_workbook):
    """A zone that is not a network bus raises instead of being silently mapped."""
    n = _network(["Z1", "Z2"])  # missing Z3
    with pytest.raises(ValueError, match="not network buses"):
        n.c.flow_based_domains.from_eraa(eraa_workbook, year="2030", season="winter1")


def test_from_eraa_unknown_season_raises(eraa_workbook):
    """Selecting a season absent from the RAM sheet fails rather than adding an empty domain."""
    n = _network()
    with pytest.raises(KeyError):
        n.c.flow_based_domains.from_eraa(eraa_workbook, year="2030", season="autumn9")


def test_from_eraa_domain_solves(eraa_workbook):
    """The imported domain feeds straight into the optimisation."""
    n = _network()
    n.add("Load", ZONES, bus=ZONES, p_set=[300.0, 500.0, 200.0])
    n.add("Generator", ZONES, bus=ZONES, p_nom=2000, marginal_cost=[10.0, 50.0, 30.0])
    n.c.flow_based_domains.from_eraa(eraa_workbook, year="2030", season="winter1")
    n.optimize(log_to_console=False)
    assert n.buses_t.p.iloc[0][ZONES].sum() == pytest.approx(0.0)  # net positions balance


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
    n.c.flow_based_domains.from_jao(jao_csv)
    c = n.c.flow_based_domains

    assert list(c.static.index) == ["1", "2"]  # non-presolved row 3 dropped; Id as name
    assert list(c.zonal_ptdf.columns) == ZONES  # Ptdf_ prefix removed
    assert c.zonal_ptdf.loc["1", "Z1"] == pytest.approx(0.4)
    assert c.static.ram.to_dict() == {"1": 1000.0, "2": 900.0}


def test_from_jao_presolved_false_reads_all(jao_csv):
    """presolved=False keeps every row."""
    n = _network()
    n.c.flow_based_domains.from_jao(jao_csv, presolved=False)
    assert len(n.c.flow_based_domains.static) == 3


def test_from_jao_non_unique_name_col_raises(jao_csv):
    """CneName is not unique, so it fails fast rather than silently collapsing rows."""
    n = _network()
    with pytest.raises(ValueError, match="unique"):
        n.c.flow_based_domains.from_jao(jao_csv, name_col="CneName")


def test_from_jao_bus_mapping(jao_csv):
    """An explicit buses mapping renames hub columns to network bus names."""
    n = _network(["ZoneOne", "Z2", "Z3"])
    n.c.flow_based_domains.from_jao(jao_csv, buses={"Z1": "ZoneOne"})
    assert list(n.c.flow_based_domains.zonal_ptdf.columns) == ["ZoneOne", "Z2", "Z3"]


_REAL_ERAA = Path(__file__).parent / "data" / "fbmc" / "FB-Domain-CORE_simplified.xlsx"
_REAL_JAO = Path(__file__).parent / "data" / "fbmc" / "finalComputation.csv"


@pytest.mark.skipif(not _REAL_ERAA.exists(), reason="ERAA example data not available")
def test_from_eraa_real_data():
    """Reproduce the ERAA Core 2030 winter1 domain (134 CNECs x 13 zones)."""
    pytest.importorskip("openpyxl")
    zones = [
        "AT00", "BE00", "CZ00", "DE00", "FR00", "HR00", "HU00",
        "ITN1", "NL00", "PL00", "RO00", "SI00", "SK00",
    ]
    n = pypsa.Network()
    n.add("Bus", zones)
    n.c.flow_based_domains.from_eraa(str(_REAL_ERAA), year="2030", season="winter1")
    c = n.c.flow_based_domains
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
    n.c.flow_based_domains.from_jao(str(_REAL_JAO))
    c = n.c.flow_based_domains
    assert c.zonal_ptdf.shape == (181, 24)
    assert list(c.zonal_ptdf.columns) == hubs
    assert c.static.index.is_unique
