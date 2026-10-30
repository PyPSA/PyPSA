# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import pypsa


def test_operational_limit_n_ac_dc_meshed(ac_dc_network):
    n = ac_dc_network

    limit = 30_000

    n.remove("GlobalConstraint", n.c.global_constraints.static.index)

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

    n.remove("GlobalConstraint", n.c.global_constraints.static.index)

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

    weights = n.snapshot_weightings.stores

    weighted_inflow = n.storage_units_t.inflow.mul(weights, axis=0).sum().sum()

    weighted_spill = n.storage_units_t.spill.mul(weights, axis=0).sum().sum()

    soc_diff = (
        n.c.storage_units.static.state_of_charge_initial.sum()
        - n.c.storage_units.dynamic.state_of_charge.sum(axis=1).iloc[-1]
    )

    assert (soc_diff + weighted_inflow - weighted_spill).round(3) == limit


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


def test_1449():
    """
    Inactive components with global carrier constraint should not break.
    See https://github.com/PyPSA/PyPSA/issues/1449.
    """
    snapshots = pd.date_range("2025-01-01", freq="h", periods=5)

    n = pypsa.Network(snapshots=snapshots)

    n.add("Carrier", name="gird_with_emissions", co2_emissions=0.5)

    n.add("Bus", name="b_electricity", unit="MW")

    n.add(
        "Generator",
        name="grid",
        bus="b_electricity",
        p_nom=10,
        marginal_cost=5,
    )

    n.add(
        "Generator",
        name="grid_2",
        bus="b_electricity",
        carrier="gird_with_emissions",
        p_nom_extendable=True,
        marginal_cost=5,
        active=False,
    )

    n.add(
        "Load",
        name="demand",
        bus="b_electricity",
        p_set=pd.Series([0, 2, 0, 0, 0], index=snapshots),
    )

    n.add(
        "GlobalConstraint",
        name="CO2_limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=50,
    )

    status, _ = n.optimize()
    assert status == "ok"


def test_1884():
    """
    A tech_capacity_expansion_limit with no investment_period must still apply.
    See https://github.com/PyPSA/PyPSA/issues/1884.
    """
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "b")
    n.add("Carrier", ["wind", "gas"])
    n.add("Load", "l", bus="b", p_set=100)
    # cheap wind is capped at 50, so expensive gas must cover the rest
    n.add(
        "Generator",
        ["wind", "gas"],
        bus="b",
        carrier=["wind", "gas"],
        p_nom_extendable=True,
        capital_cost=1,
        marginal_cost=[1, 100],
    )
    n.add(
        "GlobalConstraint",
        "wind_limit",
        type="tech_capacity_expansion_limit",
        carrier_attribute="wind",
        sense="<=",
        constant=50,
    )

    n.optimize()
    assert n.c.generators.static.p_nom_opt["wind"] == pytest.approx(50)


@pytest.mark.parametrize("component", ["Store", "StorageUnit"])
@pytest.mark.parametrize("cyclic", [True, False])
@pytest.mark.parametrize(
    ("glc_type", "carrier_attribute"),
    [
        ("primary_energy", "co2_emissions"),
        ("operational_limit", "fuel"),
    ],
)
def test_2000(component, cyclic, glc_type, carrier_attribute):
    """Global constraints must account for storage inflow and spill."""

    n = pypsa.Network(snapshots=range(4))

    n.add("Carrier", "fuel", co2_emissions=1.0)
    n.add("Bus", "b")
    n.add("Load", "l", bus="b", p_set=10)

    n.add(
        "Generator",
        "backup",
        bus="b",
        p_nom=100,
        marginal_cost=100,
    )

    # Add storage
    if component == "Store":
        n.add(
            "Store",
            "s",
            bus="b",
            carrier="fuel",
            e_nom=100,
            max_hours=10,
            e_cyclic=cyclic,
            inflow=10,
        )
    else:
        n.add(
            "StorageUnit",
            "s",
            bus="b",
            carrier="fuel",
            p_nom=100,
            max_hours=10,
            cyclic_state_of_charge=cyclic,
            inflow=10,
        )

    # Add global constraint
    n.add(
        "GlobalConstraint",
        "limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=0,
    )

    n.optimize()

    if component == "Store":
        spill = n.c.stores.dynamic.spill.sum().item()
        initial_energy = n.stores.e_initial.sum()
        final_energy = n.stores_t.e.iloc[-1].sum()
    else:
        spill = n.c.storage_units.dynamic.spill.sum().item()
        initial_energy = n.storage_units.state_of_charge_initial.sum()
        final_energy = n.storage_units_t.state_of_charge.iloc[-1].sum()

    if cyclic:
        assert spill == pytest.approx(40, abs=1e-6)
    else:
        assert spill + final_energy - initial_energy == pytest.approx(40, abs=1e-6)


@pytest.mark.parametrize(
    ("glc_type", "carrier_attribute"),
    [("primary_energy", "co2_emissions"), ("operational_limit", "gas")],
)
def test_investment_period_without_multi_invest_raises(glc_type, carrier_attribute):
    """
    An investment_period raises if the network has no investment periods.
    See https://github.com/PyPSA/PyPSA/issues/1954.
    """
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", "b")
    n.add("Carrier", ["gas", "wind"], co2_emissions=[1, 0])
    n.add("Load", "l", bus="b", p_set=50)
    n.add(
        "Generator",
        ["gas", "wind"],
        bus="b",
        carrier=["gas", "wind"],
        p_nom=100,
        marginal_cost=[1, 100],
    )
    n.add(
        "GlobalConstraint",
        "gas_limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=100,
        investment_period=0,
    )

    with pytest.raises(ValueError, match="has no investment periods"):
        n.optimize()


def test_transmission_cost_limit_overnight_cost():
    n = pypsa.Network()
    n.snapshot_weightings.loc[:, :] = 8760.0
    n.add("Bus", ["a", "b"])
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.01,
        s_nom_extendable=True,
        overnight_cost=1000,
        discount_rate=0,
        lifetime=10,
    )
    n.add(
        "GlobalConstraint",
        "cost_limit",
        type="transmission_expansion_cost_limit",
        constant=10000,
        sense="<=",
        carrier_attribute="AC",
    )
    m = n.optimize.create_model()
    con = m.constraints["GlobalConstraint-cost_limit"]
    assert (con.coeffs == 100.0).all()
