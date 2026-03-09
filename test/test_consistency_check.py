# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import os

import numpy as np
import pandas as pd
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


def test_committable_extendable_allowed(consistent_n, caplog):
    consistent_n.c.generators.static.loc[
        "gen_one", ["p_nom_extendable", "committable"]
    ] = (
        True,
        True,
    )
    consistent_n.consistency_check()
    assert not any(
        "only be committable or extendable" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize("strict", [[], ["static_power_attrs"]])
def test_nans_in_capacity_limits(consistent_n, caplog, strict):
    consistent_n.c.generators.static.loc["gen_one", "p_nom_extendable"] = True
    consistent_n.c.generators.static.loc["gen_one", "p_nom_max"] = np.nan
    assert_log_or_error_in_consistency(consistent_n, caplog, strict=strict)


@pytest.mark.parametrize("strict", [[], ["shapes"]])
def test_shapes_with_missing_idx(ac_dc_shapes, caplog, strict):
    n = ac_dc_shapes
    with pytest.warns(UserWarning, match="CRS not set"):  # Userwarning from pyproj
        n.add(
            "Shape",
            "missing_idx",
            geometry=n.c.shapes.static.geometry.iloc[0],
            component="Bus",
            idx="missing_idx",
        )
    assert_log_or_error_in_consistency(ac_dc_shapes, caplog, strict=strict)
    if not strict:
        assert any(
            "have idx values that are not included" in r.message for r in caplog.records
        )


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


@pytest.mark.parametrize("strict", [[], ["scenarios_sum"]])
def test_scenarios_sum_to_one(consistent_n, caplog, strict):
    """
    Test that the consistency check raises a warning if scenarios don't sum to 1.
    """
    # Set up scenarios that sum to 1 (should pass)
    consistent_n.set_scenarios({"low": 0.4, "high": 0.6})

    # Manually modify scenarios to break sum=1 constraint
    consistent_n._scenarios_data.iloc[0, 0] = 0.2  # Sum becomes 0.8

    assert_log_or_error_in_consistency(consistent_n, caplog, strict=strict)


@pytest.mark.parametrize("strict", [[], ["generators"]])
def test_committable_down_with_p_init(consistent_n, caplog, strict):
    consistent_n.add(
        "Generator",
        "gen_uc",
        bus="one",
        committable=True,
        up_time_before=0,
        p_init=50,
    )
    assert_log_or_error_in_consistency(consistent_n, caplog, strict=strict)


def test_unknown_check():
    n = pypsa.Network()
    with pytest.raises(ValueError):
        n.consistency_check(strict=["some_check"])


@pytest.mark.parametrize("strict", [[], ["scenario_invariant_attrs"]])
def test_scenario_invariant_attributes(consistent_n, caplog, strict):
    """
    Test that the consistency check raises an error if invariant attributes vary across scenarios.
    """
    # Set up scenarios
    consistent_n.set_scenarios({"s1": 0.5, "s2": 0.5})

    # Modify an invariant attribute (carrier) across scenarios - this should always fail
    # regardless of strict mode
    consistent_n.c.generators.static.loc[("s1", "gen_one"), "carrier"] = (
        "different_carrier"
    )

    # This check always raises an error
    with pytest.raises(pypsa.consistency.ConsistencyError):
        consistent_n.consistency_check(strict=strict)


def test_scenario_invariant_attributes_comprehensive():
    """
    Comprehensive test covering all invariant attributes and edge cases.

    This test verifies that the following attributes are not changed across scenarios
    of a stochastic network by user modifications. Applies *exclusively* to stochastic
    networks.

    Invariant attributes (must be identical across all scenarios):
    - name
    - bus
    - type
    - p_nom_extendable
    - committable
    - sign
    - carrier
    - weight
    - p_nom_opt
    - build_year
    - lifetime
    - active

    Note: "control" is not included as an invariant attribute because different
    scenarios can have different control types (PQ, PV, Slack). However, slack bus
    consistency is enforced separately via check_stochastic_slack_bus_consistency.
    """
    n = pypsa.Network()

    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Carrier", "gas")
    n.add("Carrier", "wind")
    n.add("Carrier", "AC")

    # Let's add multiple components to test different cases with invariant attributes
    n.add(
        "Generator",
        "gen1",
        bus="bus1",
        carrier="gas",
        p_nom_extendable=True,
        committable=True,
    )
    n.add("Line", "line1", bus0="bus1", bus1="bus2", x=0.1, r=0.01, carrier="AC")
    n.add("Load", "load1", bus="bus1", p_set=50, carrier="AC")
    n.add("Link", "link1", bus0="bus1", bus1="bus2", p_nom=75, carrier="AC")

    # Set up scenarios
    n.set_scenarios({"scenario1": 0.4, "scenario2": 0.6})

    # Test 1: Network with consistent invariant attributes should pass
    n.consistency_check()

    # Test 2: Test invariant attributes for generators (only test attributes that exist)
    # Note: "control" is no longer tested as invariant because different scenarios
    # can have different control types (PQ, PV, Slack) but slack bus consistency
    # is enforced separately
    generator_invariant_tests = [
        ("generators", "gen1", "carrier", "wind"),
        ("generators", "gen1", "bus", "bus2"),
        ("generators", "gen1", "p_nom_extendable", False),
        ("generators", "gen1", "committable", False),
        ("generators", "gen1", "sign", -1.0),
        ("generators", "gen1", "weight", 5.0),
        ("generators", "gen1", "type", "solar"),
        ("generators", "gen1", "active", False),
    ]

    for component_name, element_name, attr, new_value in generator_invariant_tests:
        n_test = n.copy()

        # Modify the invariant attribute in one scenario
        component = getattr(n_test.c, component_name)
        component.static.loc[("scenario1", element_name), attr] = new_value

        # Should always raise an error regardless of strict mode
        with pytest.raises(
            pypsa.consistency.ConsistencyError,
            match=f"Component '{element_name}' .* has attribute '{attr}' that varies across scenarios",
        ):
            n_test.consistency_check()

    # Test 3: Test invariant attributes for lines
    line_invariant_tests = [
        ("lines", "line1", "carrier", "gas"),
        ("lines", "line1", "active", False),
    ]

    for component_name, element_name, attr, new_value in line_invariant_tests:
        n_test = n.copy()

        # Modify the invariant attribute in one scenario
        component = getattr(n_test.c, component_name)
        component.static.loc[("scenario1", element_name), attr] = new_value

        # Should always raise an error
        with pytest.raises(
            pypsa.consistency.ConsistencyError,
            match=f"Component '{element_name}' .* has attribute '{attr}' that varies across scenarios",
        ):
            n_test.consistency_check()

    # Test 4: Test invariant attributes for links
    link_invariant_tests = [
        ("links", "link1", "carrier", "gas"),
        ("links", "link1", "active", False),
        ("links", "link1", "committable", True),
        ("links", "link1", "p_nom_extendable", True),
        ("links", "link1", "type", "HVDC"),
    ]

    for component_name, element_name, attr, new_value in link_invariant_tests:
        n_test = n.copy()

        # Modify the invariant attribute in one scenario
        component = getattr(n_test.c, component_name)
        component.static.loc[("scenario1", element_name), attr] = new_value

        # Should always raise an error
        with pytest.raises(
            pypsa.consistency.ConsistencyError,
            match=f"Component '{element_name}' .* has attribute '{attr}' that varies across scenarios",
        ):
            n_test.consistency_check()

    # Test 5: Test with NaN values - should not raise error if all scenarios have NaN
    n_nan = n.copy()
    n_nan.c.generators.static["build_year"] = n_nan.c.generators.static[
        "build_year"
    ].astype(float)
    n_nan.c.generators.static.loc[:, "build_year"] = np.nan
    n_nan.consistency_check()  # Should pass

    # Test 6: Test with mixed NaN and non-NaN values - should raise error (strict behavior)
    n_mixed_nan = n.copy()
    n_mixed_nan.c.generators.static["lifetime"] = n_mixed_nan.c.generators.static[
        "lifetime"
    ].astype(float)
    n_mixed_nan.c.generators.static.loc[("scenario1", "gen1"), "lifetime"] = 25.0
    n_mixed_nan.c.generators.static.loc[("scenario2", "gen1"), "lifetime"] = np.nan

    with pytest.raises(
        pypsa.consistency.ConsistencyError,
        match="Component 'gen1' .* has attribute 'lifetime' that varies across scenarios",
    ):
        n_mixed_nan.consistency_check()  # Should raise error (any difference is not allowed)

    # Test 7: Test that non-invariant attributes can vary (should not raise error)
    n_varying = n.copy()
    n_varying.c.generators.static.loc[("scenario1", "gen1"), "p_nom"] = (
        150  # p_nom is not invariant
    )
    n_varying.c.generators.static.loc[("scenario1", "gen1"), "p_set"] = (
        80  # p_set is not invariant
    )
    n_varying.c.lines.static.loc[("scenario1", "line1"), "s_nom"] = (
        200  # s_nom is not invariant
    )
    n_varying.c.links.static.loc[("scenario1", "link1"), "p_nom"] = (
        100  # p_nom is not invariant
    )
    n_varying.consistency_check()  # Should pass

    # Test 8: Test with non-stochastic network (should skip check)
    n_non_stoch = pypsa.Network()
    n_non_stoch.add("Bus", "bus")
    n_non_stoch.add("Generator", "gen", bus="bus", carrier="test")
    n_non_stoch.consistency_check()  # Should pass


def test_p_nom_mod_invariant():
    """Test that p_nom_mod is enforced as invariant across scenarios."""
    n = pypsa.Network()
    n.add("Bus", "bus1")
    n.add(
        "Generator",
        "gen1",
        bus="bus1",
        p_nom_extendable=True,
        p_nom_mod=0.7,
    )

    n.set_scenarios({"s1": 0.5, "s2": 0.5})

    # Should pass with same p_nom_mod across scenarios
    n.consistency_check()

    n.c.generators.static.loc[("s1", "gen1"), "p_nom_mod"] = 0.7
    n.c.generators.static.loc[("s2", "gen1"), "p_nom_mod"] = 1.0

    # Should raise error
    with pytest.raises(
        pypsa.consistency.ConsistencyError,
        match="Component 'gen1' .* has attribute 'p_nom_mod' that varies across scenarios",
    ):
        n.consistency_check()


@pytest.mark.parametrize("strict", [[], ["line_types"]])
def test_line_types_consistency(caplog, strict):
    """
    Test that the consistency check raises an error if line_types vary across scenarios.
    """
    n = pypsa.Network()
    n.add("Bus", "bus1", v_nom=20)
    n.add("Bus", "bus2", v_nom=20)

    n.set_scenarios({"s1": 0.5, "s2": 0.5})

    # Create line_types with MultiIndex (different across scenarios)
    import pandas as pd

    line_types_data = {
        ("s1", "type1"): {"r": 0.1, "x": 0.2, "c": 0.0, "i_nom": 100},
        ("s1", "type2"): {"r": 0.15, "x": 0.25, "c": 0.0, "i_nom": 150},
        ("s2", "type1"): {"r": 0.1, "x": 0.2, "c": 0.0, "i_nom": 100},  # Same as s1
        ("s2", "type2"): {
            "r": 0.2,
            "x": 0.3,
            "c": 0.0,
            "i_nom": 200,
        },  # Different from s1
    }

    line_types_df = pd.DataFrame.from_dict(line_types_data, orient="index")
    line_types_df.index = pd.MultiIndex.from_tuples(
        line_types_df.index, names=["scenario", "type"]
    )

    # Manually set line_types to simulate stochastic network
    n.c.line_types.static = line_types_df

    # Test only the line_types consistency check directly to avoid calculate_dependent_values
    if strict and "line_types" in strict:
        with pytest.raises(pypsa.consistency.ConsistencyError):
            pypsa.consistency.check_line_types_consistency(n, strict=True)
    else:
        # For non-strict mode, check that it logs a warning
        pypsa.consistency.check_line_types_consistency(n, strict=False)
        assert caplog.records[-1].levelname == "WARNING"


def test_line_types_consistency_pass():
    """
    Test that the consistency check passes when line_types are identical across scenarios.
    """
    n = pypsa.Network()
    n.add("Bus", "bus1", v_nom=20)
    n.add("Bus", "bus2", v_nom=20)

    n.set_scenarios({"s1": 0.5, "s2": 0.5})

    # Create identical line_types across scenarios
    import pandas as pd

    line_types_data = {
        ("s1", "type1"): {"r": 0.1, "x": 0.2, "c": 0.0, "i_nom": 100},
        ("s1", "type2"): {"r": 0.15, "x": 0.25, "c": 0.0, "i_nom": 150},
        ("s2", "type1"): {"r": 0.1, "x": 0.2, "c": 0.0, "i_nom": 100},  # Same as s1
        ("s2", "type2"): {"r": 0.15, "x": 0.25, "c": 0.0, "i_nom": 150},  # Same as s1
    }

    line_types_df = pd.DataFrame.from_dict(line_types_data, orient="index")
    line_types_df.index = pd.MultiIndex.from_tuples(
        line_types_df.index, names=["scenario", "type"]
    )

    # Manually set line_types to simulate stochastic network
    n.c.line_types.static = line_types_df

    # This should pass because line_types are identical across scenarios
    pypsa.consistency.check_line_types_consistency(n, strict=True)


def test_line_types_consistency_non_stochastic():
    """
    Test that the consistency check is skipped for non-stochastic networks.
    """
    n = pypsa.Network()
    n.add("Bus", "bus1", v_nom=20)
    n.add("Bus", "bus2", v_nom=20)

    # Add line_types without scenarios (normal operation)
    n.add("LineType", "type1", r=0.1, x=0.2, c=0.0, i_nom=100)
    n.add("LineType", "type2", r=0.15, x=0.25, c=0.0, i_nom=150)

    # This should pass because it's not a stochastic network
    pypsa.consistency.check_line_types_consistency(n, strict=True)


@pytest.mark.parametrize("strict", [[], ["transformer_types"]])
def test_transformer_types_consistency_fail(caplog, strict):
    """
    Test that the consistency check fails when transformer_types differ across scenarios.
    """
    n = pypsa.Network()
    n.add("Bus", "bus1", v_nom=220)
    n.add("Bus", "bus2", v_nom=110)

    n.set_scenarios({"s1": 0.5, "s2": 0.5})

    transformer_types_data = {
        ("s1", "type1"): {"s_nom": 100, "vsc": 12.0, "vscr": 0.26, "pfe": 55},
        ("s2", "type1"): {"s_nom": 150, "vsc": 15.0, "vscr": 0.30, "pfe": 65},
    }

    transformer_types_df = pd.DataFrame.from_dict(
        transformer_types_data, orient="index"
    )
    transformer_types_df.index = pd.MultiIndex.from_tuples(
        transformer_types_df.index, names=["scenario", "type"]
    )

    n.c.transformer_types.static = transformer_types_df

    if strict and "transformer_types" in strict:
        with pytest.raises(pypsa.consistency.ConsistencyError):
            pypsa.consistency.check_transformer_types_consistency(n, strict=True)
    else:
        pypsa.consistency.check_transformer_types_consistency(n, strict=False)
        assert caplog.records[-1].levelname == "WARNING"


def test_transformer_types_consistency_pass():
    """
    Test that the consistency check passes when transformer_types are identical across scenarios.
    """
    n = pypsa.Network()
    n.add("Bus", "bus1", v_nom=220)
    n.add("Bus", "bus2", v_nom=110)

    n.set_scenarios({"s1": 0.5, "s2": 0.5})

    transformer_types_data = {
        ("s1", "type1"): {"s_nom": 100, "vsc": 12.0, "vscr": 0.26, "pfe": 55},
        ("s2", "type1"): {"s_nom": 100, "vsc": 12.0, "vscr": 0.26, "pfe": 55},
    }

    transformer_types_df = pd.DataFrame.from_dict(
        transformer_types_data, orient="index"
    )
    transformer_types_df.index = pd.MultiIndex.from_tuples(
        transformer_types_df.index, names=["scenario", "type"]
    )

    n.c.transformer_types.static = transformer_types_df

    pypsa.consistency.check_transformer_types_consistency(n, strict=True)


def test_transformer_types_consistency_non_stochastic():
    """
    Test that the consistency check is skipped for non-stochastic networks.
    """
    n = pypsa.Network()
    n.add("Bus", "bus1", v_nom=220)
    n.add("Bus", "bus2", v_nom=110)

    n.add("TransformerType", "type1", s_nom=100, vsc=12.0, vscr=0.26, pfe=55)
    n.add("TransformerType", "type2", s_nom=160, vsc=12.2, vscr=0.25, pfe=60)

    pypsa.consistency.check_transformer_types_consistency(n, strict=True)


@pytest.mark.parametrize("strict", [[], ["unknown_buses"]])
def test_check_for_unknown_buses(caplog, strict):
    """Test check_for_unknown_buses via consistency_check(): GlobalConstraint/Link empty buses OK, invalid warns."""
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")

    # Add components with empty buses (should be OK)
    n.add("GlobalConstraint", "gc1")
    n.add("Link", "link1", bus0="bus0", bus1="bus1")
    caplog.clear()
    n.consistency_check(strict=strict)
    assert not any("buses which are not defined" in r.message for r in caplog.records)

    # Add component with invalid bus (should warn/error)
    n.add("Generator", "gen1", bus="invalid_bus")
    assert_log_or_error_in_consistency(n, caplog, strict=strict)


def test_check_for_unknown_buses_when_adding(caplog):
    """Test check_for_unknown_buses: empty buses in GlobalConstraint/Links OK, invalid buses warn."""
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")

    # GlobalConstraint with empty bus - no warning
    caplog.clear()
    n.add("GlobalConstraint", "gc1")
    assert not any("buses which are not defined" in r.message for r in caplog.records)

    # Link with empty bus2/bus3 - no warning
    caplog.clear()
    n.add("Link", "link1", bus0="bus0", bus1="bus1")
    assert not any("buses which are not defined" in r.message for r in caplog.records)

    # Invalid bus - should warn
    caplog.clear()
    n.add("Generator", "gen1", bus="invalid")
    assert any("buses which are not defined" in r.message for r in caplog.records)
