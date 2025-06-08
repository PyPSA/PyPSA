"""
Created on Fri Jul  2 10:21:16 2021.

@author: fabian
"""

import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal
from pandas import IndexSlice as idx

import pypsa
from pypsa.descriptors import get_activity_mask

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
    n.remove("Generator", n.generators.query('bus == "1"').index)
    n.generators.capital_cost *= 5

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
    n.remove("Generator", n.generators.query('bus == "1"').index)
    n.generators.capital_cost *= 5

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
    active_gens = n.get_active_assets("Generator", 2030)[lambda ds: ds].index
    assert (active_gens == ["gen1-2020", "gen2-2020", "gen1-2030", "gen2-2030"]).all()

    active_gens = n.get_active_assets("Generator", 2050)[lambda ds: ds].index
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
    assert n.generators.p_nom_opt.item() == 100


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
    assert n.generators.p_nom_opt.item() == 100


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

    assert (n.generators_t.p.loc[[2020, 2030, 2040], "gen1-2050"] == 0).all()
    assert (n.generators_t.p.loc[[2050], "gen1-2020"] == 0).all()

    assert (n.lines_t.p0.loc[[2020, 2030, 2040], "line-2050"] == 0).all()


def test_simple_network_snapshot_subset(n):
    status, cond = n.optimize(n.snapshots[:20], **kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n.generators_t.p.loc[[2020, 2030, 2040], "gen1-2050"] == 0).all()
    assert (n.generators_t.p.loc[[2050], "gen1-2020"] == 0).all()

    assert (n.lines_t.p0.loc[[2020, 2030, 2040], "line-2050"] == 0).all()


def test_simple_network_storage_noncyclic(n_sus):
    n_sus.storage_units["state_of_charge_initial"] = 200
    n_sus.storage_units["cyclic_state_of_charge"] = False
    n_sus.storage_units["state_of_charge_initial_per_period"] = False

    status, cond = n_sus.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    soc = n_sus.storage_units_t.state_of_charge
    p = n_sus.storage_units_t.p
    assert round((soc + p).loc[idx[2020, 0], "sto1-2020"], 4) == 200
    assert soc.loc[idx[2040, 9], "sto1-2020"] == 0


def test_simple_network_storage_noncyclic_per_period(n_sus):
    n_sus.storage_units["state_of_charge_initial"] = 200
    n_sus.storage_units["cyclic_state_of_charge"] = False
    n_sus.storage_units["state_of_charge_initial_per_period"] = True

    status, cond = n_sus.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sus.storage_units_t.p.loc[[2020, 2030, 2040], "sto1-2050"] == 0).all()
    assert (n_sus.storage_units_t.p.loc[[2050], "sto1-2020"] == 0).all()

    soc_initial = (n_sus.storage_units_t.state_of_charge + n_sus.storage_units_t.p).loc[
        idx[:, 0], :
    ]
    soc_initial = soc_initial.droplevel("timestep")
    assert soc_initial.loc[2020, "sto1-2020"] == 200
    assert soc_initial.loc[2030, "sto1-2020"] == 200
    assert soc_initial.loc[2040, "sto1-2040"] == 200


def test_simple_network_storage_cyclic(n_sus):
    n_sus.storage_units["cyclic_state_of_charge"] = True
    n_sus.storage_units["cyclic_state_of_charge_per_period"] = False

    status, cond = n_sus.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    soc = n_sus.storage_units_t.state_of_charge
    p = n_sus.storage_units_t.p
    assert (
        soc.loc[idx[2040, 9], "sto1-2020"] == (soc + p).loc[idx[2020, 0], "sto1-2020"]
    )
    assert (
        soc.loc[idx[2050, 9], "sto1-2030"] == (soc + p).loc[idx[2030, 0], "sto1-2030"]
    )


def test_simple_network_storage_cyclic_per_period(n_sus):
    # Watch out breaks with xarray version 2022.06.00 !
    n_sus.storage_units["cyclic_state_of_charge"] = True
    n_sus.storage_units["cyclic_state_of_charge_per_period"] = True

    status, cond = n_sus.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    soc = n_sus.storage_units_t.state_of_charge
    p = n_sus.storage_units_t.p
    assert (
        soc.loc[idx[2020, 9], "sto1-2020"] == (soc + p).loc[idx[2020, 0], "sto1-2020"]
    )


def test_simple_network_store_noncyclic(n_sts):
    n_sts.stores["e_cyclic"] = False
    n_sts.stores["e_initial_per_period"] = False

    status, cond = n_sts.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.stores_t.p.loc[[2050], "sto1-2020"] == 0).all()

    e_initial = (n_sts.stores_t.e + n_sts.stores_t.p).loc[idx[:, 0], :]
    e_initial = e_initial.droplevel("timestep")
    assert e_initial.loc[2020, "sto1-2020"] == 20


def test_simple_network_store_noncyclic_per_period(n_sts):
    n_sts.stores["e_cyclic"] = False
    n_sts.stores["e_initial_per_period"] = True

    status, cond = n_sts.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.stores_t.p.loc[[2050], "sto1-2020"] == 0).all()

    e_initial = (n_sts.stores_t.e + n_sts.stores_t.p).loc[idx[:, 0], :]
    e_initial = e_initial.droplevel("timestep")
    assert e_initial.loc[2020, "sto1-2020"] == 20
    assert e_initial.loc[2030, "sto1-2020"] == 20

    # lifetime is over here
    assert e_initial.loc[2050, "sto1-2020"] == 0


def test_simple_network_store_cyclic(n_sts):
    n_sts.stores["e_cyclic"] = True
    n_sts.stores["e_cyclic_per_period"] = False

    status, cond = n_sts.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.stores_t.p.loc[[2050], "sto1-2020"] == 0).all()

    e = n_sts.stores_t.e
    p = n_sts.stores_t.p
    assert e.loc[idx[2040, 9], "sto1-2020"] == (e + p).loc[idx[2020, 0], "sto1-2020"]


def test_simple_network_store_cyclic_per_period(n_sts):
    # Watch out breaks with xarray version 2022.06.00 !
    n_sts.stores["e_cyclic"] = True
    n_sts.stores["e_cyclic_per_period"] = True

    status, cond = n_sts.optimize(**kwargs)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.stores_t.p.loc[[2050], "sto1-2020"] == 0).all()

    e = n_sts.stores_t.e
    p = n_sts.stores_t.p
    assert e.loc[idx[2020, 9], "sto1-2020"] == (e + p).loc[idx[2020, 0], "sto1-2020"]


def test_global_constraint_primary_energy_storage(n_sus):
    c = "StorageUnit"
    n_sus.add("Carrier", "emitting_carrier", co2_emissions=100)
    n_sus.static(c)["state_of_charge_initial"] = 200
    n_sus.static(c)["cyclic_state_of_charge"] = False
    n_sus.static(c)["state_of_charge_initial_per_period"] = False
    n_sus.static(c)["carrier"] = "emitting_carrier"

    n_sus.add("GlobalConstraint", name="co2limit", type="primary_energy", constant=3000)

    status, cond = n_sus.optimize(**kwargs)

    active = get_activity_mask(n_sus, c)
    soc_end = n_sus.dynamic(c).state_of_charge.where(active).ffill().iloc[-1]
    soc_diff = n_sus.static(c).state_of_charge_initial - soc_end
    emissions = n_sus.static(c).carrier.map(n_sus.carriers.co2_emissions)
    assert round(soc_diff @ emissions, 0) == 3000


def test_global_constraint_primary_energy_store(n_sts):
    c = "Store"
    n_sts.add("Carrier", "emitting_carrier", co2_emissions=100)
    n_sts.static(c)["e_initial"] = 200
    n_sts.static(c)["e_cyclic"] = False
    n_sts.static(c)["e_initial_per_period"] = False

    n_sts.buses.loc["1 battery", "carrier"] = "emitting_carrier"

    n_sts.add("GlobalConstraint", name="co2limit", type="primary_energy", constant=3000)

    status, cond = n_sts.optimize(**kwargs)

    active = get_activity_mask(n_sts, c)
    soc_end = n_sts.dynamic(c).e.where(active).ffill().iloc[-1]
    soc_diff = n_sts.static(c).e_initial - soc_end
    emissions = n_sts.static(c).carrier.map(n_sts.carriers.co2_emissions)
    assert round(soc_diff @ emissions, 0) == 3000


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
    assert n.lines.s_nom_opt.sum() == 100

    # when only optimizing the first 10 snapshots the contraint must hold for
    # the 2020 period
    status, cond = n.optimize(n.snapshots[:10], **kwargs)
    assert n.lines.loc["line-2020", "s_nom_opt"] == 100

    n.global_constraints["investment_period"] = 2030
    status, cond = n.optimize(**kwargs)
    assert n.lines.s_nom_opt[["line-2020", "line-2030"]].sum() == 100


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
            period: n.get_active_assets("Line", period)
            for period in n.investment_periods
        },
        axis=1,
    )
    weight = active @ n.investment_period_weightings.objective

    status, cond = n.optimize(**kwargs)
    assert round((weight * n.lines.eval("s_nom_opt * capital_cost")).sum(), 2) == 1000

    # when only optimizing the first 10 snapshots the contraint must hold for
    # the 2020 period
    status, cond = n.optimize(n.snapshots[:10], **kwargs)
    assert round(n.lines.eval("s_nom_opt * capital_cost")["line-2020"].sum(), 2) == 1000

    n.global_constraints["investment_period"] = 2030
    status, cond = n.optimize(**kwargs)
    lines = n.lines.loc[["line-2020", "line-2030"]]
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
    assert round(n.generators.p_nom_opt[["gen1-2020", "gen2-2020"]], 1).sum() == 300

    n.global_constraints["bus"] = 1
    status, cond = n.optimize(**kwargs)
    assert n.generators.at["gen1-2020", "p_nom_opt"] == 300

    # make the constraint non-binding and check that the shadow price is zero
    n.global_constraints.sense = "<="
    status, cond = n.optimize(**kwargs)
    assert n.global_constraints.at["expansion_limit", "mu"] == 0


def test_nominal_constraint_bus_carrier_expansion_limit(n):
    n.buses.at["1", "nom_max_gencarrier"] = 100
    status, cond = n.optimize(**kwargs)
    gen1s = [f"gen1-{period}" for period in n.investment_periods]
    assert round(n.generators.p_nom_opt[gen1s], 0).sum() == 100
    n.buses.drop(["nom_max_gencarrier"], inplace=True, axis=1)

    n.buses.at["1", "nom_max_gencarrier_2020"] = 100
    status, cond = n.optimize(**kwargs)
    assert n.generators.at["gen1-2020", "p_nom_opt"] == 100
    n.buses.drop(["nom_max_gencarrier_2020"], inplace=True, axis=1)

    # make the constraint non-binding and check that the shadow price is zero
    n.buses.at["1", "nom_min_gencarrier_2020"] = 100
    status, cond = n.optimize(**kwargs)
    assert (n.model.constraints["Bus-nom_min_gencarrier_2020"].dual).item() == 0


def test_max_growth_constraint(n):
    # test generator grow limit
    gen_carrier = n.generators.carrier.unique()[0]
    n.carriers.at[gen_carrier, "max_growth"] = 218
    status, cond = n.optimize(**kwargs)
    assert all(n.generators.p_nom_opt.groupby(n.generators.build_year).sum() <= 218)


def test_max_relative_growth_constraint(n):
    # test generator relative grow limit
    gen_carrier = n.generators.carrier.unique()[0]
    n.carriers.at[gen_carrier, "max_growth"] = 218
    n.carriers.at[gen_carrier, "max_relative_growth"] = 1.5
    status, cond = n.optimize(**kwargs)
    built_per_period = n.generators.p_nom_opt.groupby(n.generators.build_year).sum()
    assert all(built_per_period - built_per_period.shift(fill_value=0) * 1.5 <= 218)
