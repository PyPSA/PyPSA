# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest
from numpy.testing import assert_almost_equal as almost_equal
from numpy.testing import assert_array_almost_equal as equal
from pandas import IndexSlice as idx

import pypsa

kwargs = {"multi_investment_periods": True}


@pytest.fixture
def n():
    n = pypsa.Network(snapshots=range(10))
    n.investment_periods = [2020, 2030, 2040, 2050]
    n.add("Carrier", "gencarrier")
    n.add("Bus", [1, 2])

    for i, period in enumerate(n.investment_periods):
        factor = (10 + i) / 10
        n.add(
            "Generator",
            [f"gen1-{period}", f"gen2-{period}"],
            bus=[1, 2],
            lifetime=30,
            build_year=period,
            capital_cost=[100 / factor, 100 * factor],
            marginal_cost=[i + 2, i + 1],
            p_nom_extendable=True,
            carrier="gencarrier",
        )

    for i, period in enumerate(n.investment_periods):
        n.add(
            "Line",
            f"line-{period}",
            bus0=1,
            bus1=2,
            length=1,
            build_year=period,
            lifetime=40,
            capital_cost=30 + i,
            x=0.0001,
            s_nom_extendable=True,
        )

    load = range(100, 100 + len(n.snapshots))
    load = pd.DataFrame({"load1": load, "load2": load}, index=n.snapshots)
    n.add(
        "Load",
        ["load1", "load2"],
        bus=[1, 2],
        p_set=load,
    )

    return n


@pytest.fixture
def n_sus(n):
    # only keep generators which are getting more expensiv and push generator
    # capital cost, so that sus are activated
    n.remove("Generator", n.c.generators.static.query('bus == "1"').index)
    n.c.generators.static.capital_cost *= 5

    for i, period in enumerate(n.investment_periods):
        factor = (10 + i) / 10
        n.add(
            "StorageUnit",
            f"sto1-{period}",
            bus=1,
            lifetime=30,
            build_year=period,
            capital_cost=10 / factor,
            marginal_cost=i,
            p_nom_extendable=True,
        )
    return n


@pytest.fixture
def n_sts(n):
    # only keep generators which are getting more expensiv and push generator
    # capital cost, so that sus are activated
    n.remove("Generator", n.c.generators.static.query('bus == "1"').index)
    n.c.generators.static.capital_cost *= 5

    n.add("Bus", "1 battery")

    n.add(
        "Store",
        "sto1-2020",
        bus="1 battery",
        e_nom_extendable=True,
        e_initial=20,
        build_year=2020,
        lifetime=30,
        capital_cost=0.1,
    )

    n.add(
        "Link", "bus2 battery charger", bus0=1, bus1="1 battery", p_nom_extendable=True
    )

    n.add(
        "Link",
        "My bus2 battery discharger",
        bus0="1 battery",
        bus1=1,
        p_nom_extendable=True,
    )

    return n


def test_single_to_multi_level_snapshots():
    n = pypsa.Network(snapshots=range(2))
    years = [2030, 2040]
    n.investment_periods = years
    assert isinstance(n.snapshots, pd.MultiIndex)
    equal(n.snapshots.unique(level="period"), years)


def test_investment_period_values():
    sns = pd.MultiIndex.from_product([[2020, 2030, 2040], [1, 2, 3]])
    n = pypsa.Network(snapshots=sns)

    with pytest.raises(ValueError):
        n.investment_periods = [2040, 2030, 2020]

    with pytest.raises(ValueError):
        n.investment_periods = ["2020", "2030", "2040"]

    with pytest.raises(NotImplementedError):
        n.investment_periods = [2020]

    n = pypsa.Network(snapshots=range(2))
    with pytest.raises(ValueError):
        n.investment_periods = ["2020", "2030", "2040"]


def test_active_assets(n):
    active_gens = n.c.generators.get_active_assets(2030)[lambda ds: ds].index
    assert (active_gens == ["gen1-2020", "gen2-2020", "gen1-2030", "gen2-2030"]).all()

    active_gens = n.c.generators.get_active_assets(2050)[lambda ds: ds].index
    assert (
        active_gens
        == [
            "gen1-2030",
            "gen2-2030",
            "gen1-2040",
            "gen2-2040",
            "gen1-2050",
            "gen2-2050",
        ]
    ).all()


def test_tiny_with_default():
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030]
    n.add("Bus", 1)
    n.add("Generator", 1, bus=1, p_nom_extendable=True, capital_cost=10)
    n.add("Load", 1, bus=1, p_set=100)
    status, _ = n.optimize(**kwargs)
    assert status == "ok"
    assert n.c.generators.static.p_nom_opt.item() == 100


def test_tiny_with_build_year():
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030]
    n.add("Bus", 1)
    n.add(
        "Generator", 1, bus=1, p_nom_extendable=True, capital_cost=10, build_year=2020
    )
    n.add("Load", 1, bus=1, p_set=100)
    status, _ = n.optimize(**kwargs)
    assert status == "ok"
    assert n.c.generators.static.p_nom_opt.item() == 100


def test_tiny_infeasible():
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030]
    n.add("Bus", 1)
    n.add(
        "Generator", 1, bus=1, p_nom_extendable=True, capital_cost=10, build_year=2030
    )
    n.add("Load", 1, bus=1, p_set=100)
    with pytest.raises(ValueError):
        status, cond = n.optimize(**kwargs)


def test_simple_network(n):
    status, cond = n.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n.c.generators.dynamic.p.loc[[2020, 2030, 2040], "gen1-2050"] == 0).all()
    assert (n.c.generators.dynamic.p.loc[[2050], "gen1-2020"] == 0).all()

    assert (n.c.lines.dynamic.p0.loc[[2020, 2030, 2040], "line-2050"] == 0).all()


def test_simple_network_snapshot_subset(n):
    status, cond = n.optimize(n.snapshots[:20], **kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n.c.generators.dynamic.p.loc[[2020, 2030, 2040], "gen1-2050"] == 0).all()
    assert (n.c.generators.dynamic.p.loc[[2050], "gen1-2020"] == 0).all()

    assert (n.c.lines.dynamic.p0.loc[[2020, 2030, 2040], "line-2050"] == 0).all()


def test_simple_network_storage_noncyclic(n_sus):
    n_sus.c.storage_units.static["state_of_charge_initial"] = 200
    n_sus.c.storage_units.static["cyclic_state_of_charge"] = False
    n_sus.c.storage_units.static["state_of_charge_initial_per_period"] = False

    status, cond = n_sus.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    soc = n_sus.c.storage_units.dynamic.state_of_charge
    p = n_sus.c.storage_units.dynamic.p
    assert round((soc + p).loc[idx[2020, 0], "sto1-2020"], 4) == 200
    assert soc.loc[idx[2040, 9], "sto1-2020"] == 0


def test_simple_network_storage_noncyclic_per_period(n_sus):
    n_sus.c.storage_units.static["state_of_charge_initial"] = 200
    n_sus.c.storage_units.static["cyclic_state_of_charge"] = False
    n_sus.c.storage_units.static["state_of_charge_initial_per_period"] = True

    status, cond = n_sus.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (
        n_sus.c.storage_units.dynamic.p.loc[[2020, 2030, 2040], "sto1-2050"] == 0
    ).all()
    assert (n_sus.c.storage_units.dynamic.p.loc[[2050], "sto1-2020"] == 0).all()

    soc_initial = (
        n_sus.c.storage_units.dynamic.state_of_charge + n_sus.c.storage_units.dynamic.p
    ).loc[idx[:, 0], :]
    soc_initial = soc_initial.droplevel("timestep")
    assert soc_initial.loc[2020, "sto1-2020"] == 200
    assert soc_initial.loc[2030, "sto1-2020"] == 200
    assert soc_initial.loc[2040, "sto1-2040"] == 200


def test_simple_network_storage_cyclic(n_sus):
    n_sus.c.storage_units.static["cyclic_state_of_charge"] = True
    n_sus.c.storage_units.static["cyclic_state_of_charge_per_period"] = False

    status, cond = n_sus.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    soc = n_sus.c.storage_units.dynamic.state_of_charge
    p = n_sus.c.storage_units.dynamic.p
    assert (
        soc.loc[idx[2040, 9], "sto1-2020"] == (soc + p).loc[idx[2020, 0], "sto1-2020"]
    )
    assert (
        soc.loc[idx[2050, 9], "sto1-2030"] == (soc + p).loc[idx[2030, 0], "sto1-2030"]
    )


def test_simple_network_storage_cyclic_per_period(n_sus):
    # Watch out breaks with xarray version 2022.06.00 !
    n_sus.c.storage_units.static["cyclic_state_of_charge"] = True
    n_sus.c.storage_units.static["cyclic_state_of_charge_per_period"] = True

    status, cond = n_sus.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    soc = n_sus.c.storage_units.dynamic.state_of_charge
    p = n_sus.c.storage_units.dynamic.p
    assert (
        soc.loc[idx[2020, 9], "sto1-2020"] == (soc + p).loc[idx[2020, 0], "sto1-2020"]
    )


def test_simple_network_store_noncyclic(n_sts):
    n_sts.c.stores.static["e_cyclic"] = False
    n_sts.c.stores.static["e_initial_per_period"] = False

    status, cond = n_sts.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.c.stores.dynamic.p.loc[[2050], "sto1-2020"] == 0).all()

    e_initial = (n_sts.c.stores.dynamic.e + n_sts.c.stores.dynamic.p).loc[idx[:, 0], :]
    e_initial = e_initial.droplevel("timestep")
    assert e_initial.loc[2020, "sto1-2020"] == 20


def test_simple_network_store_noncyclic_per_period(n_sts):
    n_sts.c.stores.static["e_cyclic"] = False
    n_sts.c.stores.static["e_initial_per_period"] = True

    status, cond = n_sts.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.c.stores.dynamic.p.loc[[2050], "sto1-2020"] == 0).all()

    e_initial = (n_sts.c.stores.dynamic.e + n_sts.c.stores.dynamic.p).loc[idx[:, 0], :]
    e_initial = e_initial.droplevel("timestep")
    assert e_initial.loc[2020, "sto1-2020"] == 20
    assert e_initial.loc[2030, "sto1-2020"] == 20

    # lifetime is over here
    assert e_initial.loc[2050, "sto1-2020"] == 0


def test_simple_network_store_cyclic(n_sts):
    n_sts.c.stores.static["e_cyclic"] = True
    n_sts.c.stores.static["e_cyclic_per_period"] = False

    status, cond = n_sts.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.c.stores.dynamic.p.loc[[2050], "sto1-2020"] == 0).all()

    e = n_sts.c.stores.dynamic.e
    p = n_sts.c.stores.dynamic.p
    assert e.loc[idx[2040, 9], "sto1-2020"] == (e + p).loc[idx[2020, 0], "sto1-2020"]


def test_simple_network_store_cyclic_per_period(n_sts):
    # Watch out breaks with xarray version 2022.06.00 !
    n_sts.c.stores.static["e_cyclic"] = True
    n_sts.c.stores.static["e_cyclic_per_period"] = True

    status, cond = n_sts.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.c.stores.dynamic.p.loc[[2050], "sto1-2020"] == 0).all()

    e = n_sts.c.stores.dynamic.e
    p = n_sts.c.stores.dynamic.p
    assert e.loc[idx[2020, 9], "sto1-2020"] == (e + p).loc[idx[2020, 0], "sto1-2020"]


def test_global_constraint_primary_energy_storage(n_sus):
    c = n_sus.components["StorageUnit"]
    n_sus.add("Carrier", "emitting_carrier", co2_emissions=100)
    c.static["state_of_charge_initial"] = 200
    c.static["cyclic_state_of_charge"] = False
    c.static["state_of_charge_initial_per_period"] = False
    c.static["carrier"] = "emitting_carrier"

    n_sus.add("GlobalConstraint", name="co2limit", type="primary_energy", constant=3000)

    status, cond = n_sus.optimize(**kwargs)

    active = c.get_activity_mask()
    soc_end = c.dynamic.state_of_charge.where(active).ffill().iloc[-1]
    soc_diff = c.static.state_of_charge_initial - soc_end
    emissions = c.static.carrier.map(n_sus.c.carriers.static.co2_emissions)
    assert round(soc_diff @ emissions, 0) == 3000


def test_global_constraint_primary_energy_store(n_sts):
    c = n_sts.components["Store"]
    n_sts.add("Carrier", "emitting_carrier", co2_emissions=100)
    c.static["e_initial"] = 200
    c.static["e_cyclic"] = False
    c.static["e_initial_per_period"] = False

    n_sts.c.buses.static.loc["1 battery", "carrier"] = "emitting_carrier"

    n_sts.add("GlobalConstraint", name="co2limit", type="primary_energy", constant=3000)

    status, cond = n_sts.optimize(**kwargs)

    active = c.get_activity_mask()
    soc_end = c.dynamic.e.where(active).ffill().iloc[-1]
    soc_diff = c.static.e_initial - soc_end
    emissions = c.static.carrier.map(n_sts.c.carriers.static.co2_emissions)
    assert round(soc_diff @ emissions, 0) == 3000


def test_global_constraint_primary_energy_storage_stochastic(n_sus):
    """
    Test global constraints with primary energy for storage in stochastic networks.

    This test ensures that multi-period optimization with storage units and
    global constraints work correctly when scenarios are present.
    """

    c = "StorageUnit"

    n_sus.add("Carrier", "emitting_carrier", co2_emissions=100)
    n_sus.c[c].static["state_of_charge_initial"] = 200
    n_sus.c[c].static["cyclic_state_of_charge"] = False
    n_sus.c[c].static["state_of_charge_initial_per_period"] = False
    n_sus.c[c].static["carrier"] = "emitting_carrier"

    n_sus.add("GlobalConstraint", name="co2limit", type="primary_energy", constant=3000)
    n_sus.set_scenarios({"s1": 0.5, "s2": 0.5})

    status, cond = n_sus.optimize(multi_investment_periods=True)
    assert status == "ok"
    assert n_sus.model.constraints["GlobalConstraint-co2limit"].rhs[0] == -77000.0


def test_global_constraint_transmission_expansion_limit(n):
    n.add(
        "GlobalConstraint",
        "expansion_limit",
        type="transmission_volume_expansion_limit",
        constant=100,
        sense="==",
        carrier_attribute="AC",
    )

    status, cond = n.optimize(**kwargs)
    assert n.c.lines.static.s_nom_opt.sum() == 100

    # when only optimizing the first 10 snapshots the contraint must hold for
    # the 2020 period
    status, cond = n.optimize(n.snapshots[:10], **kwargs)
    assert n.c.lines.static.loc["line-2020", "s_nom_opt"] == 100

    n.c.global_constraints.static["investment_period"] = 2030
    status, cond = n.optimize(**kwargs)
    assert n.c.lines.static.s_nom_opt[["line-2020", "line-2030"]].sum() == 100


def test_global_constraint_transmission_cost_limit(n):
    n.add(
        "GlobalConstraint",
        "expansion_limit",
        type="transmission_expansion_cost_limit",
        constant=1000,
        sense="==",
        carrier_attribute="AC",
    )

    active = pd.concat(
        {
            period: n.c.lines.get_active_assets(period)
            for period in n.investment_periods
        },
        axis=1,
    )
    weight = active @ n.investment_period_weightings.objective

    status, cond = n.optimize(**kwargs)
    assert (
        round((weight * n.c.lines.static.eval("s_nom_opt * capital_cost")).sum(), 2)
        == 1000
    )

    # when only optimizing the first 10 snapshots the contraint must hold for
    # the 2020 period
    status, cond = n.optimize(n.snapshots[:10], **kwargs)
    assert (
        round(n.c.lines.static.eval("s_nom_opt * capital_cost")["line-2020"].sum(), 2)
        == 1000
    )

    n.c.global_constraints.static["investment_period"] = 2030
    status, cond = n.optimize(**kwargs)
    lines = n.c.lines.static.loc[["line-2020", "line-2030"]]
    assert round(lines.eval("s_nom_opt * capital_cost").sum(), 2) == 1000


def test_global_constraint_bus_tech_limit(n):
    n.add(
        "GlobalConstraint",
        "expansion_limit",
        type="tech_capacity_expansion_limit",
        constant=300,
        sense="==",
        carrier_attribute="gencarrier",
        investment_period=2020,
    )

    status, cond = n.optimize(**kwargs)
    assert (
        round(n.c.generators.static.p_nom_opt[["gen1-2020", "gen2-2020"]], 1).sum()
        == 300
    )

    n.c.global_constraints.static["bus"] = 1
    status, cond = n.optimize(**kwargs)
    assert n.c.generators.static.at["gen1-2020", "p_nom_opt"] == 300

    # make the constraint non-binding and check that the shadow price is zero
    n.c.global_constraints.static.sense = "<="
    status, cond = n.optimize(**kwargs)
    assert n.c.global_constraints.static.at["expansion_limit", "mu"] == 0


def test_nominal_constraint_bus_carrier_expansion_limit(n):
    n.c.buses.static.at["1", "nom_max_gencarrier"] = 100
    with pytest.warns(
        # DeprecationWarning, match="Nominal constraints per bus carrier are deprecated"
        DeprecationWarning,
        match=".+",
    ):
        status, cond = n.optimize(**kwargs)
    gen1s = [f"gen1-{period}" for period in n.investment_periods]
    assert round(n.c.generators.static.p_nom_opt[gen1s], 0).sum() == 100
    n.c.buses.static.drop(["nom_max_gencarrier"], inplace=True, axis=1)

    n.c.buses.static.at["1", "nom_max_gencarrier_2020"] = 100
    with pytest.warns(
        # DeprecationWarning, match="Nominal constraints per bus carrier are deprecated"
        DeprecationWarning,
        match=".+",
    ):
        status, cond = n.optimize(**kwargs)
    assert n.c.generators.static.at["gen1-2020", "p_nom_opt"] == 100
    n.c.buses.static.drop(["nom_max_gencarrier_2020"], inplace=True, axis=1)

    # make the constraint non-binding and check that the shadow price is zero
    n.c.buses.static.at["1", "nom_min_gencarrier_2020"] = 100
    with pytest.warns(
        # DeprecationWarning, match="Nominal constraints per bus carrier are deprecated"
        DeprecationWarning,
        match=".+",
    ):
        status, cond = n.optimize(**kwargs)
    assert (n.model.constraints["Bus-nom_min_gencarrier_2020"].dual).item() == 0


def test_max_growth_constraint(n):
    # test generator grow limit
    gen_carrier = n.c.generators.static.carrier.unique()[0]
    n.c.carriers.static.at[gen_carrier, "max_growth"] = 218
    status, cond = n.optimize(**kwargs)
    assert all(
        n.c.generators.static.p_nom_opt.groupby(n.c.generators.static.build_year).sum()
        <= 218
    )


def test_max_relative_growth_constraint(n):
    # test generator relative grow limit
    gen_carrier = n.c.generators.static.carrier.unique()[0]
    n.c.carriers.static.at[gen_carrier, "max_growth"] = 218
    n.c.carriers.static.at[gen_carrier, "max_relative_growth"] = 1.5
    status, cond = n.optimize(**kwargs)
    built_per_period = n.c.generators.static.p_nom_opt.groupby(
        n.c.generators.static.build_year
    ).sum()
    assert all(built_per_period - built_per_period.shift(fill_value=0) * 1.5 <= 218)


def test_store_primary_energy_and_operational_limit_constraint_without_per_period():
    """Test that Store with primary energy constraint raises NotImplementedError without e_initial_per_period."""

    n = pypsa.Network()
    years = [2030, 2040]
    timesteps = [1, 2]

    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years
    n.investment_period_weightings.loc[:, :] = 2

    n.add("Bus", "bus")
    n.add("Carrier", "gas", co2_emissions=0.2)
    n.add(
        "Store",
        "store",
        bus="bus",
        carrier="gas",
        marginal_cost=1,
        e_nom=10,
        e_initial=10,
    )
    n.add("Generator", "gen", bus="bus", marginal_cost=10, p_nom=2)
    n.add("Load", "load", bus="bus", p_set=pd.Series(1, index=n.snapshots))

    n.add(
        "GlobalConstraint",
        "co2",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=0.8,
    )

    with pytest.raises(NotImplementedError):
        n.optimize(multi_investment_periods=True)

    n.remove("GlobalConstraint", n.c.global_constraints.static.index)
    n.add(
        "GlobalConstraint",
        "co2",
        type="operational_limit",
        carrier_attribute="gas",
        sense="<=",
        constant=6,
    )

    with pytest.raises(NotImplementedError):
        n.optimize(multi_investment_periods=True)


def test_store_primary_energy_and_operational_limit_constraint_with_per_period():
    """Test that Store with primary energy constraint works with e_initial_per_period=True."""

    n = pypsa.Network()
    years = [2030, 2040]
    timesteps = [1, 2]

    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years
    n.investment_period_weightings.loc[:, :] = 2

    n.add("Bus", "bus")
    n.add("Carrier", "gas", co2_emissions=0.2)
    n.add(
        "Store",
        "store",
        bus="bus",
        carrier="gas",
        marginal_cost=1,
        e_nom=10,
        e_initial=10,
    )
    n.add("Generator", "gen", bus="bus", marginal_cost=10, p_nom=2)
    n.add("Load", "load", bus="bus", p_set=pd.Series(1, index=n.snapshots))

    n.c.stores.static.e_initial_per_period = True
    n.add(
        "GlobalConstraint",
        "co2",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=0.8,
    )

    n.optimize(multi_investment_periods=True)
    almost_equal(n.objective, 10 * 4 + 1 * 4)

    # Test operational limit constraint as well
    n.remove("GlobalConstraint", n.c.global_constraints.names)
    n.add(
        "GlobalConstraint",
        "dispatch",
        carrier_attribute="gas",
        type="operational_limit",
        sense="<=",
        constant=6,
    )

    n.optimize(multi_investment_periods=True)
    almost_equal(n.objective, 10 * 2 + 1 * 6)


def test_storage_unit_primary_energy_and_operational_limit_constraint_without_per_period():
    """Test that StorageUnit with primary energy constraint raises NotImplementedError without state_of_charge_initial_per_period."""

    n = pypsa.Network()
    years = [2030, 2040]
    timesteps = [1, 2]

    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years
    n.investment_period_weightings.loc[:, :] = 2

    n.add("Bus", "bus")
    n.add("Carrier", "gas", co2_emissions=0.2)
    n.add(
        "StorageUnit",
        "su",
        bus="bus",
        carrier="gas",
        marginal_cost=1,
        p_nom=10,
        max_hours=1,
        state_of_charge_initial=10,
    )
    n.add("Generator", "gen", bus="bus", marginal_cost=10, p_nom=2)
    n.add("Load", "load", bus="bus", p_set=pd.Series(1, index=n.snapshots))

    n.add(
        "GlobalConstraint",
        "co2",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=0.8,
    )

    with pytest.raises(NotImplementedError):
        n.optimize(multi_investment_periods=True)

    n.remove("GlobalConstraint", n.c.global_constraints.names)
    n.add(
        "GlobalConstraint",
        "co2",
        type="operational_limit",
        carrier_attribute="gas",
        sense="<=",
        constant=6,
    )

    with pytest.raises(NotImplementedError):
        n.optimize(multi_investment_periods=True)


def test_storage_unit_primary_energy_and_operational_limit_constraint_with_per_period():
    """Test that StorageUnit with primary energy constraint works with state_of_charge_initial_per_period=True."""

    n = pypsa.Network()
    years = [2030, 2040]
    timesteps = [1, 2]

    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years
    n.investment_period_weightings.loc[:, :] = 2

    n.add("Bus", "bus")
    n.add("Carrier", "gas", co2_emissions=0.2)
    n.add(
        "StorageUnit",
        "su",
        bus="bus",
        carrier="gas",
        marginal_cost=1,
        p_nom=10,
        max_hours=1,
        state_of_charge_initial=10,
    )
    n.add("Generator", "gen", bus="bus", marginal_cost=10, p_nom=2)
    n.add("Load", "load", bus="bus", p_set=pd.Series(1, index=n.snapshots))

    n.c.storage_units.static.state_of_charge_initial_per_period = True
    n.add(
        "GlobalConstraint",
        "co2",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=0.8,
    )

    n.optimize(multi_investment_periods=True)
    almost_equal(n.objective, 10 * 4 + 1 * 4)

    # Test operational limit constraint as well
    n.remove("GlobalConstraint", n.c.global_constraints.names)
    n.add(
        "GlobalConstraint",
        "dispatch",
        carrier_attribute="gas",
        type="operational_limit",
        sense="<=",
        constant=6,
    )

    n.optimize(multi_investment_periods=True)
    almost_equal(n.objective, 10 * 2 + 1 * 6)


def test_bug_1360_stores():
    """
    Storage state of charge should behave correctly with various snapshot configurations.
    See https://github.com/PyPSA/PyPSA/issues/1360.
    """
    # Case 1: Simple snapshots without multi_investment_periods
    # Expected: stores_t.e [9, 8, 7, 6] - continuous discharge
    n = pypsa.Network()
    n.snapshots = [1, 2, 3, 4]

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[1, 1, 1, 1])
    n.add("Store", "store", bus="bus", e_nom=10, e_initial=10, marginal_cost=1)

    n.optimize(multi_investment_periods=False)
    equal(n.stores_t.e["store"].values, [9, 8, 7, 6], decimal=5)

    # Case 2: Multi-indexed snapshots without multi_investment_periods
    # Expected: stores_t.e [9, 8, 7, 6] - continuous discharge
    n = pypsa.Network()
    years = [2030, 2040]
    timesteps = [1, 2]

    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[1, 1, 1, 1])
    n.add(
        "Store",
        "store",
        bus="bus",
        e_nom=10,
        e_initial=10,
        marginal_cost=1,
        e_initial_per_period=False,
    )

    n.optimize(multi_investment_periods=False)
    equal(n.stores_t.e["store"].values, [9, 8, 7, 6], decimal=5)

    # Case 3: Multi_investment_periods with e_initial_per_period=True
    # Expected: stores_t.e [9, 8, 9, 8] - reset at each period
    n = pypsa.Network()
    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[1, 1, 1, 1])
    n.add(
        "Store",
        "store",
        bus="bus",
        e_nom=10,
        e_initial=10,
        marginal_cost=1,
        e_initial_per_period=True,
    )

    n.optimize(multi_investment_periods=True)
    equal(n.stores_t.e["store"].values, [9, 8, 9, 8], decimal=5)

    # Case 4: Multi_investment_periods with e_initial_per_period=False
    # Expected: stores_t.e [9, 8, 7, 6] - continuous discharge across periods
    n = pypsa.Network()
    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[1, 1, 1, 1])
    n.add(
        "Store",
        "store",
        bus="bus",
        e_nom=10,
        e_initial=10,
        marginal_cost=1,
        e_initial_per_period=False,
    )

    n.optimize(multi_investment_periods=True)
    equal(n.stores_t.e["store"].values, [9, 8, 7, 6], decimal=5)


def test_bug_1360_storage_units():
    """
    Storage units state of charge should behave correctly with various snapshot configurations.
    See https://github.com/PyPSA/PyPSA/issues/1360.
    """
    # Case 1: Simple snapshots without multi_investment_periods
    # Expected: storage_units_t.state_of_charge [0.9, 0.8, 0.7, 0.6] - continuous discharge
    n = pypsa.Network()
    n.snapshots = [1, 2, 3, 4]

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[0.1, 0.1, 0.1, 0.1])
    n.add(
        "StorageUnit",
        "storage_unit",
        bus="bus",
        p_nom=1,
        max_hours=1,
        state_of_charge_initial=1,
        marginal_cost=1,
    )

    n.optimize(multi_investment_periods=False)
    equal(
        n.c.storage_units.dynamic.state_of_charge["storage_unit"].values,
        [0.9, 0.8, 0.7, 0.6],
        decimal=5,
    )

    # Case 2: Multi-indexed snapshots without multi_investment_periods
    # Expected: storage_units_t.state_of_charge [0.9, 0.8, 0.7, 0.6] - continuous discharge
    n = pypsa.Network()
    years = [2030, 2040]
    timesteps = [1, 2]

    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[0.1, 0.1, 0.1, 0.1])
    n.add(
        "StorageUnit",
        "storage_unit",
        bus="bus",
        p_nom=1,
        max_hours=1,
        state_of_charge_initial=1,
        marginal_cost=1,
        state_of_charge_initial_per_period=False,
    )

    n.optimize(multi_investment_periods=False)
    equal(
        n.c.storage_units.dynamic.state_of_charge["storage_unit"].values,
        [0.9, 0.8, 0.7, 0.6],
        decimal=5,
    )

    # Case 3: Multi_investment_periods with state_of_charge_initial_per_period=True
    # Expected: storage_units_t.state_of_charge [0.9, 0.8, 0.9, 0.8] - reset at each period
    n = pypsa.Network()
    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[0.1, 0.1, 0.1, 0.1])
    n.add(
        "StorageUnit",
        "storage_unit",
        bus="bus",
        p_nom=1,
        max_hours=1,
        state_of_charge_initial=1,
        marginal_cost=1,
        state_of_charge_initial_per_period=True,
    )

    n.optimize(multi_investment_periods=True)
    equal(
        n.c.storage_units.dynamic.state_of_charge["storage_unit"].values,
        [0.9, 0.8, 0.9, 0.8],
        decimal=5,
    )

    # Case 4: Multi_investment_periods with state_of_charge_initial_per_period=False
    # Expected: storage_units_t.state_of_charge [0.9, 0.8, 0.7, 0.6] - continuous discharge across periods
    n = pypsa.Network()
    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[0.1, 0.1, 0.1, 0.1])
    n.add(
        "StorageUnit",
        "storage_unit",
        bus="bus",
        p_nom=1,
        max_hours=1,
        state_of_charge_initial=1,
        marginal_cost=1,
        state_of_charge_initial_per_period=False,
    )

    n.optimize(multi_investment_periods=True)
    equal(
        n.c.storage_units.dynamic.state_of_charge["storage_unit"].values,
        [0.9, 0.8, 0.7, 0.6],
        decimal=5,
    )


def test_storageunit_cp_only_wraps_per_period():
    """cp=True, ip=False, c=False: per-period wrap is enforced.

    Verifies that with cyclic per period enabled (and no per-period initial reset),
    the model links the first snapshot of each period to the last snapshot of the
    same period (wrap) and thus avoids a purely continuous discharge pattern.
    """
    n = pypsa.Network()
    years = [2030, 2040]
    timesteps = [1, 2]
    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[0.1, 0.1, 0.1, 0.1])
    # Add a generator that is more expensive than discharging the storage so the
    # optimizer prefers to discharge if cyclic-per-period isn't enforced. This
    # keeps the test sensitive: with correct semantics it would need to
    # recharge within each period, producing level pattern like [0.9,1.0,0.9,1.0].
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=1.0,
        marginal_cost=1.0,
    )
    n.add(
        "StorageUnit",
        "su",
        bus="bus",
        p_nom=1,
        max_hours=1,
        state_of_charge_initial=1,
        cyclic_state_of_charge=False,
        cyclic_state_of_charge_per_period=True,
        state_of_charge_initial_per_period=False,
        marginal_cost=0.0,  # cheaper than generator
    )

    status, _ = n.optimize(multi_investment_periods=True)
    assert status == "ok"
    soc = n.c.storage_units.dynamic.state_of_charge["su"].values
    # Detect monotonic discharge signature.
    expected_continuous = [0.9, 0.8, 0.7, 0.6]
    if all(abs(soc[i] - expected_continuous[i]) < 1e-6 for i in range(4)):
        raise AssertionError(
            "Observed continuous discharge pattern [0.9,0.8,0.7,0.6]"
            "cyclic_state_of_charge_per_period=True was ignored"
        )


def test_storageunit_ip_only_resets_per_period():
    """ip=True, cp=False, c=False: per-period resets to initial are enforced.

    Verifies that with per-period initial enabled (and no cyclic-per-period wrap),
    the model resets state of charge to the user-provided initial at each period
    start, avoiding a continuous discharge pattern across period boundaries.
    """
    n = pypsa.Network()
    years = [2030, 2040]
    timesteps = [1, 2]
    n.snapshots = pd.MultiIndex.from_product([years, timesteps])
    n.investment_periods = years

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[0.1, 0.1, 0.1, 0.1])
    n.add(
        "StorageUnit",
        "su_ip_only",
        bus="bus",
        p_nom=1,
        max_hours=1,
        state_of_charge_initial=1,
        cyclic_state_of_charge=False,
        cyclic_state_of_charge_per_period=False,
        state_of_charge_initial_per_period=True,  # target behavior
        marginal_cost=1.0,  # non-zero cost for objective contribution
    )

    status, _ = n.optimize(multi_investment_periods=True)
    assert status == "ok"
    soc = n.c.storage_units.dynamic.state_of_charge["su_ip_only"].values
    continuous = [0.9, 0.8, 0.7, 0.6]
    # If we observe the continuous pattern, storage level reset did not occur.
    if all(abs(soc[i] - continuous[i]) < 1e-6 for i in range(4)):
        raise AssertionError(
            "Observed continuous discharge [0.9,0.8,0.7,0.6]"
            "state_of_charge_initial_per_period=True was ignored"
        )
