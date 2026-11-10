# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import numpy as np
from numpy.testing import assert_array_almost_equal as equal

import pypsa


def test_modular_components():
    """
    This test is based on https://docs.pypsa.org/latest/examples/unit-commitment/
    and is not very comprehensive.
    """
    n = pypsa.Network()

    snapshots = range(4)

    n.set_snapshots(snapshots)

    n.add("Bus", "bus")

    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom_extendable=True,
        marginal_cost=10,
        p_nom_max=5000,
        p_nom_mod=1000,
    )

    n.add(
        "Store",
        "Store_unit",
        bus="bus",
        e_nom_extendable=True,
        e_nom_max=2000,
        e_nom_mod=100,
        capital_cost=100,
    )

    n.add("Load", "load", bus="bus", p_set=[4000, 5000, 6000, 800])

    n.optimize()

    expected_p_nom_opt_gen = np.array([5000], dtype=float).T
    expected_e_nom_opt_store = np.array([1000], dtype=float).T

    equal(n.c.generators.static.p_nom_opt, expected_p_nom_opt_gen)
    equal(n.c.stores.static.e_nom_opt, expected_e_nom_opt_store)


def test_modular_committable_with_ramp_limits():
    n = pypsa.Network(snapshots=range(6))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[200, 500, 300, 600, 250, 550])

    n.add(
        "Generator",
        "modgen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        p_nom_max=800,
        p_nom_mod=200,
        p_min_pu=0.3,
        marginal_cost=30,
        capital_cost=100,
        ramp_limit_up=0.5,
        ramp_limit_down=0.5,
        status=0,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    p_nom_opt = n.c["Generator"].static.loc["modgen", "p_nom_opt"]
    p_nom_mod = n.c["Generator"].static.loc["modgen", "p_nom_mod"]
    assert p_nom_opt > 0
    assert p_nom_opt % p_nom_mod == 0

    p = n.c["Generator"].dynamic["p"]["modgen"]
    u = n.c["Generator"].dynamic["status"]["modgen"]
    ru = n.c["Generator"].static.loc["modgen", "ramp_limit_up"]
    rd = n.c["Generator"].static.loc["modgen", "ramp_limit_down"]

    constraints = list(n.model.constraints)
    assert "Generator-p-ramp_limit_up" in constraints
    assert "Generator-p-ramp_limit_down" in constraints

    for t in range(1, len(p)):
        delta = p.iloc[t] - p.iloc[t - 1]
        u_prev, u_cur = u.iloc[t - 1], u.iloc[t]
        startup = max(0, u_cur - u_prev)
        shutdown = max(0, u_prev - u_cur)
        assert delta <= ru * p_nom_mod * u_prev + 1.0 * p_nom_mod * startup + 1e-6
        assert delta >= -rd * p_nom_mod * u_cur - 1.0 * p_nom_mod * shutdown - 1e-6


def test_modular_fixed_committable_dispatch():
    """Regression test for https://github.com/PyPSA/PyPSA/issues/1899."""
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=150)
    n.add(
        "Generator",
        "modgen",
        bus="bus",
        p_nom=300,
        p_nom_mod=100,
        committable=True,
        p_min_pu=0.4,
        marginal_cost=1,
    )

    status, _ = n.optimize(solver_name="highs")

    assert status == "ok"
    assert (n.c.generators.dynamic.p["modgen"] == 150).all()
    assert "Generator-com-p-lower" not in n.model.constraints


def test_modular_fixed_committable_ramp_limits():
    """Regression test for https://github.com/PyPSA/PyPSA/issues/1985."""
    dispatch = {}
    for extendable in (False, True):
        n = pypsa.Network(snapshots=range(2))
        n.add("Bus", "bus")
        n.add("Load", "load", bus="bus", p_set=[50, 100])
        n.add("Generator", "peak", bus="bus", p_nom=1000, marginal_cost=100)
        n.add(
            "Generator",
            "modgen",
            bus="bus",
            committable=True,
            p_nom=0 if extendable else 100,
            p_nom_extendable=extendable,
            p_nom_min=100,
            p_nom_max=100,
            p_nom_mod=10,
            ramp_limit_up=0.1,
            ramp_limit_start_up=0.1,
            marginal_cost=1,
        )
        status, _ = n.optimize(solver_name="highs")
        assert status == "ok"
        dispatch[extendable] = n.c.generators.dynamic.p["modgen"]

    # 10 modules of 10 MW may ramp by 0.1 * 10 * 10 = 10 MW in total
    equal(dispatch[False], [50, 60])
    equal(dispatch[True], [50, 60])


def test_modular_fixed_committable_ramp_limits_first_snapshot():
    """A fixed modular unit that was on ramps from ``p_init`` with all modules."""
    n = pypsa.Network(snapshots=range(2))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=100)
    n.add("Generator", "cheap", bus="bus", p_nom=1000, marginal_cost=0)
    n.add(
        "Generator",
        "modgen",
        bus="bus",
        committable=True,
        p_nom=100,
        p_nom_mod=10,
        p_init=100,
        up_time_before=1,
        ramp_limit_down=0.1,
        ramp_limit_shut_down=0.1,
        marginal_cost=1,
    )
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    # 10 modules of 10 MW may ramp down by 0.1 * 10 * 10 = 10 MW per snapshot
    equal(n.c.generators.dynamic.p["modgen"], [90, 80])


def test_modular_fixed_committable_start_up_ramp_limits():
    """Modules starting from zero ramp up one module allowance at a time."""
    dispatch = {}
    for extendable in (False, True):
        n = pypsa.Network(snapshots=range(3))
        n.add("Bus", "bus")
        n.add("Load", "load", bus="bus", p_set=[0, 50, 50])
        n.add("Generator", "backup", bus="bus", p_nom=1000, marginal_cost=1000)
        n.add(
            "Generator",
            "modgen",
            bus="bus",
            committable=True,
            p_nom=0 if extendable else 50,
            p_nom_extendable=extendable,
            p_nom_min=50,
            p_nom_max=50,
            p_nom_mod=10,
            ramp_limit_up=0.2,
            ramp_limit_start_up=0.2,
            marginal_cost=1,
        )
        status, _ = n.optimize(solver_name="highs")
        assert status == "ok"
        dispatch[extendable] = n.c.generators.dynamic.p["modgen"]

    # 5 modules of 10 MW may ramp by 0.2 * 10 * 5 = 10 MW in total
    equal(dispatch[False], [0, 10, 20])
    equal(dispatch[True], [0, 10, 20])
