import os

import numpy as np
import pytest

import pypsa
import pypsa.consistency


def assert_log_or_error_in_consistency(n, caplog, strict=False):
    if strict:
        with pytest.raises(pypsa.consistency.ConsistencyError):
            n.consistency_check(strict=strict)
    else:
        n.consistency_check(strict=strict)
        assert caplog.records[-1].levelname == "WARNING"


@pytest.fixture
def consistent_n():
    n = pypsa.Network()
    n.add("Bus", "one")
    n.add("Bus", "two")
    n.add("Generator", "gen_one", bus="one", p_nom_max=10)
    n.add("Line", "line_one", bus0="one", bus1="two", x=0.01, r=0.01)
    n.add("Carrier", "AC")
    return n


@pytest.mark.parametrize("strict", [[], ["unknown_buses"]])
@pytest.mark.skipif(os.name == "nt", reason="dtype confusing on Windows")
def test_consistency(consistent_n, caplog, strict):
    if strict:
        consistent_n.consistency_check(strict=strict)
    else:
        consistent_n.consistency_check()
        assert not caplog.records


@pytest.mark.parametrize("strict", [[], ["disconnected_buses"]])
def test_missing_bus(consistent_n, caplog, strict):
    consistent_n.add("Bus", "three")
    assert_log_or_error_in_consistency(consistent_n, caplog, strict=strict)


@pytest.mark.parametrize("strict", [[], ["assets"]])
def test_infeasible_capacity_limits(consistent_n, caplog, strict):
    consistent_n.generators.loc["gen_one", ["p_nom_extendable", "committable"]] = (
        True,
        True,
    )
    assert_log_or_error_in_consistency(consistent_n, caplog, strict=strict)


@pytest.mark.parametrize("strict", [[], ["static_power_attrs"]])
def test_nans_in_capacity_limits(consistent_n, caplog, strict):
    consistent_n.generators.loc["gen_one", "p_nom_extendable"] = True
    consistent_n.generators.loc["gen_one", "p_nom_max"] = np.nan
    assert_log_or_error_in_consistency(consistent_n, caplog, strict=strict)


@pytest.mark.parametrize("strict", [[], ["shapes"]])
def test_shapes_with_missing_idx(ac_dc_network_shapes, caplog, strict):
    n = ac_dc_network_shapes
    n.add(
        "Shape",
        "missing_idx",
        geometry=n.shapes.geometry.iloc[0],
        component="Bus",
        idx="missing_idx",
    )
    assert_log_or_error_in_consistency(ac_dc_network_shapes, caplog, strict=strict)


@pytest.mark.parametrize("strict", [[], ["unknown_buses"]])
def test_unknown_carriers(consistent_n, caplog, strict):
    consistent_n.add("Generator", "wind", bus="hub", carrier="wind")
    assert_log_or_error_in_consistency(consistent_n, caplog, strict=strict)


@pytest.mark.parametrize("strict", [[], ["generators"]])
def test_inconsistent_e_sum_values(consistent_n, caplog, strict):
    """
    Test that the consistency check raises a warning if the e_sum_min is greater than e_sum_max.
    """
    consistent_n.add(
        "Generator", "gen_two", bus="one", p_nom_max=10, e_sum_min=10, e_sum_max=5
    )
    assert_log_or_error_in_consistency(consistent_n, caplog, strict=strict)


def test_unknown_check():
    n = pypsa.Network()
    with pytest.raises(ValueError):
        n.consistency_check(strict=["some_check"])
