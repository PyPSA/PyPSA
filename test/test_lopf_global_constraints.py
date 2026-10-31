# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
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

    # The limit applies to the energy drawn from the storage units: their
    # discharge before and charge after conversion losses, plus standing losses.
    # It is computed from the solved flows, independent of inflow and spill.
    static = n.c.storage_units.static
    dynamic = n.c.storage_units.dynamic
    w = n.snapshot_weightings.stores
    assert (dynamic.inflow.sum() > 0).any()

    conversion = (
        dynamic.p_dispatch / static.efficiency_dispatch
        - dynamic.p_store * static.efficiency_store
    ).mul(w, axis=0)
    eff_stand = pd.DataFrame(
        {name: (1 - loss) ** w for name, loss in static.standing_loss.items()}
    )
    # The energy balance applies standing losses from the second snapshot on,
    # not to `state_of_charge_initial`.
    standing = (1 - eff_stand) * dynamic.state_of_charge.shift()

    assert conversion.sum().sum() + standing.sum().sum() == pytest.approx(limit)


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


GLC_TYPES = [("primary_energy", "co2_emissions"), ("operational_limit", "fuel")]


def _fuel_network(snapshots=range(4), investment_periods=None, co2_emissions=1.0):
    """Network with a load of 10 MW, an expensive backup generator and fuel carrier."""
    n = pypsa.Network(snapshots=snapshots)
    if investment_periods is not None:
        n.investment_periods = investment_periods
    n.add("Carrier", "fuel", co2_emissions=co2_emissions)
    n.add("Bus", "b")
    n.add("Load", "l", bus="b", p_set=10)
    n.add("Generator", "backup", bus="b", p_nom=100, marginal_cost=100)
    return n


def _add_fuel_storage(n, component, cyclic=False, per_period=False, **kwargs):
    """Add a lossless fuel storage `s` with 10 MW and 100 MWh."""
    if component == "Store":
        n.add(
            "Store",
            "s",
            bus="b",
            carrier="fuel",
            e_nom=100,
            max_hours=10,
            e_cyclic=cyclic,
            e_cyclic_per_period=per_period,
            **kwargs,
        )
    else:
        n.add(
            "StorageUnit",
            "s",
            bus="b",
            carrier="fuel",
            p_nom=10,
            max_hours=10,
            cyclic_state_of_charge=cyclic,
            cyclic_state_of_charge_per_period=per_period,
            **kwargs,
        )


@pytest.mark.parametrize("component", ["Store", "StorageUnit"])
@pytest.mark.parametrize("cyclic", [True, False])
@pytest.mark.parametrize("inflow", [10, [0, 20, 5, 15]])
@pytest.mark.parametrize(("glc_type", "carrier_attribute"), GLC_TYPES)
def test_2000(component, cyclic, inflow, glc_type, carrier_attribute):
    """A limit of zero forbids dispatching storage inflow, so the backup runs."""
    n = _fuel_network()
    _add_fuel_storage(n, component, cyclic=cyclic, inflow=inflow)
    n.add(
        "GlobalConstraint",
        "limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=0,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.dynamic.p["backup"].sum() == pytest.approx(40)
    assert n.c[component].dynamic.p["s"].sum() == pytest.approx(0, abs=1e-6)
    assert n.objective == pytest.approx(4000)


@pytest.mark.parametrize("component", ["Store", "StorageUnit"])
def test_2000_non_unit_emission_factor(component):
    """Inflow and spill are both scaled by the emission factor exactly once.

    The cyclic storage receives 4 x 10 MWh of inflow at 2 t/MWh, so the limit of
    40 t allows 20 MWh of dispatch and the remaining 20 MWh must be spilled. The
    backup covers the other 20 MWh of the 40 MWh load. Omitting the factor would
    allow 40 MWh of dispatch, applying it twice only 10 MWh.
    """
    n = _fuel_network(co2_emissions=2.0)
    _add_fuel_storage(n, component, cyclic=True, inflow=10)
    n.add(
        "GlobalConstraint",
        "limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=40,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c[component].dynamic.p["s"].sum() == pytest.approx(20)
    assert n.c[component].dynamic.spill["s"].sum() == pytest.approx(20)
    assert n.c.generators.dynamic.p["backup"].sum() == pytest.approx(20)
    assert n.objective == pytest.approx(2000)
    # 1 t more allows 0.5 MWh more storage dispatch, saving 0.5 MWh of backup at 100
    assert n.c.global_constraints.static.mu["limit"] == pytest.approx(-50)


@pytest.mark.parametrize("component", ["Store", "StorageUnit"])
@pytest.mark.parametrize("investment_period", [np.nan, 2030])
@pytest.mark.parametrize(("glc_type", "carrier_attribute"), GLC_TYPES)
def test_2000_inflow_counted_only_while_active(
    component, investment_period, glc_type, carrier_attribute
):
    """Inflow counts only in snapshots in which the storage is active.

    The storage is built in 2030 and cycles per period, so its dispatch equals its
    net inflow. The limit counts net inflow weighted by the stores snapshot weighting
    and the period weighting, 2 h x 5 a, so 100 allows 10 MWh of dispatch in 2030.
    The backup covers the remaining 20 + 10 MWh at 100 with objective weighting 3.
    Weighting by the generators snapshot weighting instead would allow only 5 MWh.
    """
    n = _fuel_network(range(2), [2020, 2030])
    n.investment_period_weightings["years"] = [10, 5]
    n.snapshot_weightings["objective"] = 3.0
    n.snapshot_weightings["stores"] = 2.0
    n.snapshot_weightings["generators"] = 4.0
    _add_fuel_storage(
        n, component, cyclic=True, per_period=True, inflow=10, build_year=2030
    )
    n.add(
        "GlobalConstraint",
        "limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        investment_period=investment_period,
        sense="<=",
        constant=100,
    )

    status, _ = n.optimize(multi_investment_periods=True)

    assert status == "ok"
    p = n.c[component].dynamic.p["s"].groupby(level="period").sum()
    assert p.loc[2020] == pytest.approx(0, abs=1e-6)
    assert p.loc[2030] == pytest.approx(10)
    backup = n.c.generators.dynamic.p["backup"].groupby(level="period").sum()
    assert backup.loc[2020] == pytest.approx(20)
    assert backup.loc[2030] == pytest.approx(10)
    assert n.objective == pytest.approx(9000)


@pytest.mark.parametrize("component", ["Store", "StorageUnit"])
@pytest.mark.parametrize(("glc_type", "carrier_attribute"), GLC_TYPES)
def test_2000_non_cyclic_per_period_inflow(component, glc_type, carrier_attribute):
    """Non-cyclic storage counts its level change and net inflow in each period.

    Each period starts at 10 MWh and ends at the pinned 4 and 6 MWh. The limit
    weights the level change by the period years and the net inflow additionally
    by the stores snapshot weighting of 2 h, so it counts 20 per unit of dispatch
    in 2020 and 10 in 2030. Without spill, 2030 dispatches (10 - 6) / 2 + 2 x 5 =
    12, leaving (200 - 120) / 20 = 4 for 2020, which spills 3 + 10 - 4 = 9. The
    backup covers the remaining 16 + 8 at 100.
    """
    n = _fuel_network(range(2), [2020, 2030])
    n.investment_period_weightings["years"] = [10, 5]
    n.snapshot_weightings["stores"] = 2.0
    end = pd.Series(np.nan, n.snapshots)
    end.loc[(2020, 1)] = 4
    end.loc[(2030, 1)] = 6
    if component == "Store":
        kwargs = {
            "e_initial": 10,
            "e_initial_per_period": True,
            # relative to e_nom of 100 MWh
            "e_min_pu": (end / 100).fillna(0),
            "e_max_pu": (end / 100).fillna(1),
        }
    else:
        kwargs = {
            "state_of_charge_initial": 10,
            "state_of_charge_initial_per_period": True,
            "state_of_charge_set": end,
        }
    _add_fuel_storage(n, component, inflow=5, **kwargs)
    n.add(
        "GlobalConstraint",
        "limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=200,
    )

    status, _ = n.optimize(multi_investment_periods=True)

    assert status == "ok"
    dynamic = n.c[component].dynamic
    p = dynamic.p["s"].groupby(level="period").sum()
    spill = dynamic.spill["s"].groupby(level="period").sum()
    backup = n.c.generators.dynamic.p["backup"].groupby(level="period").sum()
    assert p.tolist() == pytest.approx([4, 12])
    assert spill.tolist() == pytest.approx([9, 0], abs=1e-6)
    assert backup.tolist() == pytest.approx([16, 8])
    assert n.objective == pytest.approx(2400)
    # 2030 dispatch is already capped by zero spill, so more allowance goes to
    # 2020 dispatch at 20 per unit, saving 100 / 20.
    assert n.c.global_constraints.static.mu["limit"] == pytest.approx(-5)


@pytest.mark.parametrize("component", ["Store", "StorageUnit"])
@pytest.mark.parametrize(("glc_type", "carrier_attribute"), GLC_TYPES)
def test_2000_storage_without_inflow(component, glc_type, carrier_attribute):
    """Storages without inflow have no spill variable and count their depletion."""
    n = _fuel_network()
    initial = "e_initial" if component == "Store" else "state_of_charge_initial"
    _add_fuel_storage(n, component, **{initial: 30})
    n.add(
        "GlobalConstraint",
        "limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=20,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert f"{component}-spill" not in n.model.variables
    assert n.c[component].dynamic.p["s"].sum() == pytest.approx(20)


@pytest.mark.parametrize(("glc_type", "carrier_attribute"), GLC_TYPES)
def test_2000_spill_of_other_storage(glc_type, carrier_attribute):
    """A spill variable of a storage with another carrier is not counted.

    The cyclic fuel store without inflow does not contribute, so no constraint
    is added, as without the spill variable.
    """
    n = _fuel_network()
    n.add("Carrier", "water")
    _add_fuel_storage(n, "Store", cyclic=True)
    n.add("Store", "hydro", bus="b", carrier="water", e_nom=100, inflow=5)
    n.add(
        "GlobalConstraint",
        "limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=0,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert "Store-spill" in n.model.variables
    assert "GlobalConstraint-limit" not in n.model.constraints


@pytest.mark.parametrize("component", ["Store", "StorageUnit"])
@pytest.mark.parametrize(("glc_type", "carrier_attribute"), GLC_TYPES)
def test_2000_negative_inflow(component, glc_type, carrier_attribute):
    """Negative inflow withdraws fuel from the storage outside the network.

    The cyclic storage has to be charged with 20 MWh from the bus, which offsets
    20 MWh of fuel generation within the limit of zero.
    """
    n = _fuel_network()
    n.add("Generator", "fuel", bus="b", carrier="fuel", p_nom=100, marginal_cost=1)
    _add_fuel_storage(n, component, cyclic=True, inflow=-5)
    n.add(
        "GlobalConstraint",
        "limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=0,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert f"{component}-spill" not in n.model.variables
    assert n.c[component].dynamic.p["s"].sum() == pytest.approx(-20)
    assert n.c.generators.dynamic.p["fuel"].sum() == pytest.approx(20)


@pytest.mark.parametrize("component", ["Store", "StorageUnit"])
@pytest.mark.parametrize(("glc_type", "carrier_attribute"), GLC_TYPES)
def test_2000_mixed_sign_inflow(component, glc_type, carrier_attribute):
    """Spill exists only with positive inflow, but negative inflow still counts.

    The net inflow of 10 MWh must be spilled under a limit of zero. Ignoring the
    negative inflow would require spilling 20 MWh and charging the storage.
    """
    n = _fuel_network()
    _add_fuel_storage(n, component, cyclic=True, inflow=[10, -5, 10, -5])
    n.add(
        "GlobalConstraint",
        "limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=0,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c[component].dynamic.spill["s"].sum() == pytest.approx(10)
    assert n.c[component].dynamic.p["s"].sum() == pytest.approx(0, abs=1e-6)
    assert n.objective == pytest.approx(4000)


@pytest.mark.parametrize(("glc_type", "carrier_attribute"), GLC_TYPES)
def test_2000_only_constant_contribution_is_skipped(glc_type, carrier_attribute):
    """Without decision variables, the constraint is skipped.

    The net inflow of the cyclic store without spill variable is a constant.
    """
    n = _fuel_network()
    _add_fuel_storage(n, "Store", cyclic=True, inflow=-5)
    n.add(
        "GlobalConstraint",
        "limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=0,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert "GlobalConstraint-limit" not in n.model.constraints


@pytest.mark.parametrize(("glc_type", "carrier_attribute"), GLC_TYPES)
def test_2000_scenario_specific_inflow(glc_type, carrier_attribute):
    """Each scenario counts its own inflow and spill."""
    n = _fuel_network()
    _add_fuel_storage(n, "Store", cyclic=True)
    n.add(
        "GlobalConstraint",
        "limit",
        type=glc_type,
        carrier_attribute=carrier_attribute,
        sense="<=",
        constant=10,
    )
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.c.stores.dynamic.inflow[("low", "s")] = 5.0
    n.c.stores.dynamic.inflow[("high", "s")] = 10.0

    status, _ = n.optimize()

    assert status == "ok"
    p = n.c.stores.dynamic.p
    spill = n.c.stores.dynamic.spill
    assert p[("low", "s")].sum() == pytest.approx(10)
    assert p[("high", "s")].sum() == pytest.approx(10)
    assert spill[("low", "s")].sum() == pytest.approx(10)
    assert spill[("high", "s")].sum() == pytest.approx(30)


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
