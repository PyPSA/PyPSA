# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the flow-based domain importers (from_eraa, ...)."""

from pathlib import Path

import pandas as pd
import pytest

import pypsa

pytest.importorskip("openpyxl")

ZONES = ["Z1", "Z2", "Z3"]


@pytest.fixture
def eraa_workbook(tmp_path):
    """A minimal ERAA-shaped workbook: two header rows, two seasons, one NaN RAM."""
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


def test_from_eraa_domain_solves(eraa_workbook):
    """The imported domain feeds straight into the optimisation."""
    n = _network()
    n.add("Load", ZONES, bus=ZONES, p_set=[300.0, 500.0, 200.0])
    n.add("Generator", ZONES, bus=ZONES, p_nom=2000, marginal_cost=[10.0, 50.0, 30.0])
    n.c.flow_based_domains.from_eraa(eraa_workbook, year="2030", season="winter1")
    n.optimize(log_to_console=False)
    assert n.buses_t.p.iloc[0][ZONES].sum() == pytest.approx(0.0)  # net positions balance


_REAL_ERAA = Path(__file__).parent / "data" / "fbmc" / "FB-Domain-CORE_simplified.xlsx"


@pytest.mark.skipif(not _REAL_ERAA.exists(), reason="ERAA example data not available")
def test_from_eraa_real_data():
    """Reproduce the ERAA Core 2030 winter1 domain (134 CNECs x 13 zones)."""
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
