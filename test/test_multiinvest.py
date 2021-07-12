#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:21:16 2021

@author: fabian
"""

import pypsa
import pytest
import os
import pandas as pd
from numpy.testing import assert_array_almost_equal as equal
from pandas import IndexSlice as idx


@pytest.fixture
def n():
    n = pypsa.Network(snapshots=range(10))
    n.investment_periods = [2020, 2030, 2040, 2050]
    n.madd("Bus", [1, 2])

    for i, period in enumerate(n.investment_periods):
        factor = (10 + i) / 10
        n.madd(
            "Generator",
            [f"gen1-{period}", f"gen2-{period}"],
            bus=[1, 2],
            lifetime=30,
            build_year=period,
            capital_cost=[100 / factor, 100 * factor],
            marginal_cost=[i + 2, i + 1],
            p_nom_extendable=True,
        )

    for i, period in enumerate(n.investment_periods):
        n.add(
            "Line",
            f"line-{period}",
            bus0=1,
            bus1=2,
            build_year=period,
            lifetime=40,
            capital_cost=30 + i,
            x=0.0001,
            s_nom_extendable=True,
        )

    load = range(100, 100 + len(n.snapshots))
    load = pd.DataFrame({"load1": load, "load2": load}, index=n.snapshots)
    n.madd(
        "Load",
        ["load1", "load2"],
        bus=[1, 2],
        p_set=load,
    )

    return n


@pytest.fixture
def n_sus(n):
    for i, period in enumerate(n.investment_periods):
        factor = (10 + i) / 10
        n.add(
            "StorageUnit",
            f"sto1-{period}",
            bus=1,
            lifetime=30,
            build_year=period,
            capital_cost=50 / factor,
            marginal_cost=i,
            p_nom_extendable=True,
        )
    return n


@pytest.fixture
def n_sts(n):
    n.add("Bus", "1 battery")

    n.add(
        "Store",
        "sto1-2020",
        bus="1 battery",
        e_nom_extendable=True,
        e_initial=20,
        build_year=2020,
        lifetime=20,
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
    equal(n.snapshots.levels[0], years)


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
    n.lopf(pyomo=False, multi_investment_periods=True)
    assert n.generators.p_nom_opt.item() == 100


def test_tiny_with_build_year():
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030]
    n.add("Bus", 1)
    n.add(
        "Generator", 1, bus=1, p_nom_extendable=True, capital_cost=10, build_year=2020
    )
    n.add("Load", 1, bus=1, p_set=100)
    n.lopf(pyomo=False, multi_investment_periods=True)
    assert n.generators.p_nom_opt.item() == 100


def test_tiny_infeasible():
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030]
    n.add("Bus", 1)
    n.add(
        "Generator", 1, bus=1, p_nom_extendable=True, capital_cost=10, build_year=2030
    )
    n.add("Load", 1, bus=1, p_set=100)
    status, cond = n.lopf(pyomo=False, multi_investment_periods=True)
    assert status == "warning"
    assert cond == "infeasible"


def test_simple_network(n):
    status, cond = n.lopf(pyomo=False, multi_investment_periods=True)
    assert status == "ok"
    assert cond == "optimal"

    assert (n.generators_t.p.loc[[2020, 2030, 2040], "gen1-2050"] == 0).all()
    assert (n.generators_t.p.loc[[2050], "gen1-2020"] == 0).all()

    assert (n.lines_t.p0.loc[[2020, 2030, 2040], "line-2050"] == 0).all()


def test_simple_network_storage_periodically(n_sus):

    n_sus.storage_units["state_of_charge_initial"] = 200
    n_sus.storage_units["cyclic_state_of_charge"] = False
    n_sus.storage_units["state_of_charge_initial_per_period"] = True

    status, cond = n_sus.lopf(pyomo=False, multi_investment_periods=True)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sus.storage_units_t.p.loc[[2020, 2030, 2040], "sto1-2050"] == 0).all()
    assert (n_sus.storage_units_t.p.loc[[2050], "sto1-2020"] == 0).all()

    soc_initial = (n_sus.storage_units_t.state_of_charge + n_sus.storage_units_t.p).loc[
        idx[:, 0], :
    ]
    soc_initial = soc_initial.droplevel("snapshot")
    assert soc_initial.loc[2020, "sto1-2020"] == 200
    assert soc_initial.loc[2030, "sto1-2020"] == 200
    assert soc_initial.loc[2040, "sto1-2040"] == 200


def test_simple_network_storage_cyclic(n_sus):

    n_sus.storage_units["cyclic_state_of_charge"] = True

    status, cond = n_sus.lopf(pyomo=False, multi_investment_periods=True)
    assert status == "ok"
    assert cond == "optimal"

    soc = n_sus.storage_units_t.state_of_charge
    p = n_sus.storage_units_t.p
    assert (
        soc.loc[idx[2020, 9], "sto1-2020"] == (soc + p).loc[idx[2020, 0], "sto1-2020"]
    )


def test_simple_network_storage_noncyclic(n_sus):

    n_sus.storage_units["state_of_charge_initial"] = 200
    n_sus.storage_units["cyclic_state_of_charge"] = False
    n_sus.storage_units["state_of_charge_initial_per_period"] = False

    status, cond = n_sus.lopf(pyomo=False, multi_investment_periods=True)
    assert status == "ok"
    assert cond == "optimal"

    soc = n_sus.storage_units_t.state_of_charge
    p = n_sus.storage_units_t.p
    assert (soc + p).loc[idx[2020, 0], "sto1-2020"] == 200
    assert soc.loc[idx[2040, 9], "sto1-2020"] == 0


def test_simple_network_store_periodically(n_sts):

    n_sts.stores["e_initial_per_period"] = True

    status, cond = n_sts.lopf(pyomo=False, multi_investment_periods=True)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.stores_t.p.loc[[2050], "sto1-2020"] == 0).all()

    e_initial = (n_sts.stores_t.e + n_sts.stores_t.p).loc[idx[:, 0], :]
    e_initial = e_initial.droplevel("snapshot")
    assert e_initial.loc[2020, "sto1-2020"] == 20
    assert e_initial.loc[2030, "sto1-2020"] == 20

    # lifetime is over here
    assert e_initial.loc[2040, "sto1-2020"] == 0


def test_simple_network_store_cyclic(n_sts):

    n_sts.stores["e_cyclic"] = True
    n_sts.stores["e_cyclic_per_period"] = True

    status, cond = n_sts.lopf(pyomo=False, multi_investment_periods=True)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.stores_t.p.loc[[2050], "sto1-2020"] == 0).all()

    e = n_sts.stores_t.e
    p = n_sts.stores_t.p
    assert e.loc[idx[2020, 9], "sto1-2020"] == (e + p).loc[idx[2020, 0], "sto1-2020"]


def test_simple_network_store_noncyclic(n_sts):

    status, cond = n_sts.lopf(pyomo=False, multi_investment_periods=True)
    assert status == "ok"
    assert cond == "optimal"

    assert (n_sts.stores_t.p.loc[[2050], "sto1-2020"] == 0).all()

    e_initial = (n_sts.stores_t.e + n_sts.stores_t.p).loc[idx[:, 0], :]
    e_initial = e_initial.droplevel("snapshot")
    assert e_initial.loc[2020, "sto1-2020"] == 20
