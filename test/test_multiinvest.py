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



def test_single_to_multi_level_snapshots():
    n = pypsa.Network(snapshots=range(2))
    years = [2030, 2040]
    n.investment_periods = years
    assert isinstance(n.snapshots, pd.MultiIndex)
    equal(n.snapshots.levels[0], years)


# TODO: test one different styles of investment_periods, depends on the
# implementation:
# what kind of IPs are allowed?
# do they always have to match the first snapshot level?


def test_active_assets(n):
    active_gens = n.get_active_assets('Generator', 2030)[lambda ds: ds].index
    assert (active_gens == ['gen1-2020', 'gen2-2020', 'gen1-2030', 'gen2-2030']).all()

    active_gens = n.get_active_assets('Generator', 2050)[lambda ds: ds].index
    assert (active_gens == ['gen1-2030', 'gen2-2030', 'gen1-2040', 'gen2-2040',
                            'gen1-2050', 'gen2-2050']).all()



def test_tiny_with_default():
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030,]
    n.add("Bus", 1,)
    n.add("Generator", 1, bus=1, p_nom_extendable=True, capital_cost=10)
    n.add("Load", 1, bus=1, p_set=100)
    n.lopf(pyomo=False, multi_investment_periods=True)
    assert n.generators.p_nom_opt.item() == 100


def test_tiny_with_build_year():
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030,]
    n.add("Bus", 1,)
    n.add("Generator", 1, bus=1, p_nom_extendable=True, capital_cost=10,
          build_year=2020)
    n.add("Load", 1, bus=1, p_set=100)
    n.lopf(pyomo=False, multi_investment_periods=True)
    assert n.generators.p_nom_opt.item() == 100


def test_tiny_infeasible():
    n = pypsa.Network(snapshots=range(2))
    n.investment_periods = [2020, 2030,]
    n.add("Bus", 1,)
    n.add("Generator", 1, bus=1, p_nom_extendable=True, capital_cost=10, build_year=2030)
    n.add("Load", 1, bus=1, p_set=100)
    status, condition = n.lopf(pyomo=False, multi_investment_periods=True)
    assert status, condition == ('warning', 'infeasible')


def test_simple_network(n):
    status, cond = n.lopf(pyomo=False, multi_investment_periods=True)
    assert status, cond == ('ok', 'optimal')
