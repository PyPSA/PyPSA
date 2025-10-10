# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest


def test_operational_limit_n_ac_dc_meshed(ac_dc_network):
    n = ac_dc_network

    limit = 30_000

    n.c.global_constraints.static.drop(
        n.c.global_constraints.static.index, inplace=True
    )

    n.add(
        "GlobalConstraint",
        "gas_limit",
        type="operational_limit",
        carrier_attribute="gas",
        sense="<=",
        constant=limit,
    )

    n.optimize()
    assert n.statistics.energy_balance().loc[:, "gas"].sum().round(3) == limit


def test_operational_limit_storage_hvdc(storage_hvdc_network):
    n = storage_hvdc_network

    limit = 5_000

    n.c.global_constraints.static.drop(
        n.c.global_constraints.static.index, inplace=True
    )

    n.add(
        "GlobalConstraint",
        "battery_limit",
        type="operational_limit",
        carrier_attribute="battery",
        sense="<=",
        constant=limit,
    )

    n.c.storage_units.static["state_of_charge_initial"] = 1_000
    n.c.storage_units.static.p_nom_extendable = True
    n.c.storage_units.static.cyclic_state_of_charge = False

    n.optimize()

    soc_diff = (
        n.c.storage_units.static.state_of_charge_initial.sum()
        - n.c.storage_units.dynamic.state_of_charge.sum(1).iloc[-1]
    )
    assert soc_diff.round(3) == limit


@pytest.mark.parametrize("assign", [True, False])
def test_assign_all_duals(ac_dc_network, assign):
    n = ac_dc_network

    limit = 30_000

    m = n.optimize.create_model()

    transmission = m.variables["Link-p"]
    m.add_constraints(
        transmission.sum() <= limit, name="GlobalConstraint-generation_limit"
    )
    m.add_constraints(
        transmission.sum(dim="name") <= limit,
        name="GlobalConstraint-generation_limit_dynamic",
    )

    n.optimize.solve_model(assign_all_duals=assign)

    assert ("generation_limit" in n.c.global_constraints.static.index) == assign
    assert ("mu_generation_limit_dynamic" in n.c.global_constraints.dynamic) == assign


def test_assign_duals_noname(ac_dc_network):
    """Test that dual values are correctly assigned back to network,
    also for a special case of constraints without component dimension."""
    n = ac_dc_network

    limit = 10000
    m = n.optimize.create_model()
    investment = m.variables["Generator-p_nom"]
    m.add_constraints(
        investment.sum() == limit, name="GlobalConstraint-investment_limit"
    )
    n.optimize.solve_model(assign_all_duals=True)

    dual_model_investment = float(
        n.model.constraints["GlobalConstraint-investment_limit"].dual
    )
    dual_network_investment = float(
        n.c.global_constraints.static.mu.loc["investment_limit"]
    )
    assert dual_model_investment == pytest.approx(
        dual_network_investment, rel=1e-8, abs=1e-10
    )

    dual_model_co2 = float(n.model.constraints["GlobalConstraint-co2_limit"].dual)
    dual_network_co2 = float(n.c.global_constraints.static.mu.loc["co2_limit"])
    assert dual_model_co2 == pytest.approx(dual_network_co2, rel=1e-8, abs=1e-10)


def test_assign_custom_variable(ac_dc_network):
    """Test that adding custom variables does not raise an error when optimizing."""
    n = ac_dc_network
    m = n.optimize.create_model()

    m.add_variables(name="custom_var")

    n.optimize.solve_model()


def test_custom_variable_warnings(ac_dc_network, caplog):
    """Test that appropriate warnings are logged for custom variables."""
    import logging

    caplog.set_level(logging.INFO)

    m = ac_dc_network.optimize.create_model()
    m.add_variables(name="custom_var_no_dash")
    m.add_variables(name="NonExistentComponent-attr")
    ac_dc_network.optimize.solve_model()

    assert all(
        text in caplog.text
        for text in [
            "could not be mapped to the network component because it does not include the symbol '-'",
            "custom_var_no_dash",
            "could not be mapped to the network component because the component 'NonExistentComponent' does not exist",
        ]
    )
