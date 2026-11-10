# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT
"""Test the committable+extendable functionality using big-M formulation."""

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.consistency import check_big_m_exceeded


@pytest.fixture
def base_network():
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 200, 150, 120])
    return n


@pytest.fixture
def two_bus_network():
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "gen0", bus="bus0", p_nom=1000, marginal_cost=10)
    n.add("Load", "load", bus="bus1", p_set=[300, 500, 400, 350])
    return n


def add_com_ext_generator(n, name="gen", **overrides):
    defaults = {
        "bus": "bus",
        "p_nom_extendable": True,
        "committable": True,
        "marginal_cost": 50,
        "capital_cost": 50000,
        "p_nom_max": 500,
        "p_min_pu": 0.3,
    }
    defaults.update(overrides)
    n.add("Generator", name, **defaults)


def add_com_ext_link(n, name="link", **overrides):
    defaults = {
        "bus0": "bus0",
        "bus1": "bus1",
        "p_nom_extendable": True,
        "committable": True,
        "marginal_cost": 5,
        "capital_cost": 30000,
        "p_nom_max": 600,
        "p_min_pu": 0.2,
    }
    defaults.update(overrides)
    n.add("Link", name, **defaults)


def assert_power_balance(n, tol=1e-3):
    total_load = n.loads_t.p_set.sum().sum()
    total_gen = n.c["Generator"].dynamic["p"].sum().sum()
    assert abs(total_load - total_gen) < tol


def test_committable_extendable_modular_generator(base_network):
    n = base_network
    n.loads_t.p_set["load"] = [400, 600, 800, 500]

    n.add(
        "Generator",
        "modular_gas",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        p_nom_mod=200,
        p_nom_max=1000,
        p_min_pu=0.3,
        marginal_cost=50,
        capital_cost=50000,
        start_up_cost=100,
        shut_down_cost=50,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    p_nom_opt = n.c["Generator"].static.loc["modular_gas", "p_nom_opt"]
    p_nom_mod = n.c["Generator"].static.loc["modular_gas", "p_nom_mod"]

    assert p_nom_opt > 0
    n_modules = p_nom_opt / p_nom_mod
    assert abs(n_modules - round(n_modules)) < 1e-3

    assert "Generator-n_mod" in n.model.variables
    n_mod_solution = n.model.variables["Generator-n_mod"].solution.loc["modular_gas"]
    assert n_mod_solution == round(n_modules)


def test_committable_extendable_modular_link(two_bus_network):
    n = two_bus_network

    n.add(
        "Link",
        "modular_link",
        bus0="bus0",
        bus1="bus1",
        p_nom_extendable=True,
        committable=True,
        p_nom_mod=150,
        p_nom_max=600,
        p_min_pu=0.2,
        marginal_cost=5,
        capital_cost=30000,
        start_up_cost=50,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    p_nom_opt = n.c["Link"].static.loc["modular_link", "p_nom_opt"]
    p_nom_mod = n.c["Link"].static.loc["modular_link", "p_nom_mod"]
    n_modules = p_nom_opt / p_nom_mod

    assert p_nom_opt > 0
    assert abs(n_modules - round(n_modules)) < 1e-3
    assert "Link-n_mod" in n.model.variables


def test_committable_extendable_generator(base_network):
    n = base_network
    n.loads_t.p_set["load"] = [300, 500, 400, 350]

    add_com_ext_generator(
        n, "gas_gen", p_nom_max=600, start_up_cost=100, shut_down_cost=50
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    assert n.c["Generator"].static.loc["gas_gen", "p_nom_opt"] > 0
    assert "Generator-status" in n.model.variables
    assert "Generator-start_up" in n.model.variables
    assert "Generator-shut_down" in n.model.variables

    status_values = n.model.variables["Generator-status"].solution
    dispatch_values = n.c["Generator"].dynamic["p"]["gas_gen"]
    p_nom_opt = n.c["Generator"].static.loc["gas_gen", "p_nom_opt"]
    p_min_pu = n.c["Generator"].static.loc["gas_gen", "p_min_pu"]
    min_power = p_min_pu * p_nom_opt

    for t in range(4):
        if status_values.loc[t, "gas_gen"] == 0:
            assert abs(dispatch_values.iloc[t]) < 1e-6
        else:
            assert dispatch_values.iloc[t] >= min_power - 1e-6


def test_committable_extendable_multiple_generators(base_network):
    n = base_network
    n.loads_t.p_set["load"] = [400, 600, 500, 450]

    n.add(
        "Generator",
        "coal_fixed",
        bus="bus",
        p_nom=300,
        committable=True,
        marginal_cost=30,
        p_min_pu=0.5,
        start_up_cost=500,
    )
    n.add(
        "Generator",
        "gas_ext",
        bus="bus",
        p_nom_extendable=True,
        marginal_cost=60,
        capital_cost=50000,
        p_nom_max=500,
    )
    add_com_ext_generator(
        n,
        "gas_com_ext",
        marginal_cost=40,
        capital_cost=80000,
        p_nom_max=400,
        start_up_cost=800,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    assert n.c["Generator"].static.loc["gas_ext", "p_nom_opt"] >= 0
    assert n.c["Generator"].static.loc["gas_com_ext", "p_nom_opt"] >= 0

    if "Generator-status" in n.model.variables:
        status_vars = n.model.variables["Generator-status"]
        committable_gens = (
            n.c["Generator"].static.loc[n.c["Generator"].static["committable"]].index
        )
        for gen in committable_gens:
            assert gen in status_vars.coords["name"].values


def test_big_m_formulation_constraints(base_network):
    n = base_network
    add_com_ext_generator(n, "test_gen", p_nom_max=1000, p_min_pu=0.4, p_max_pu=0.8)

    n.optimize.create_model()

    constraint_names = list(n.model.constraints)
    assert any("Generator-com-ext-p-lower" in name for name in constraint_names)
    assert any("Generator-com-ext-p-upper-bigM" in name for name in constraint_names)
    assert any("Generator-com-ext-p-upper-cap" in name for name in constraint_names)


def test_big_m_scale_infers_peak_load():
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 250, 180])

    assert n.c.generators._infer_committable_big_m_scale() == pytest.approx(2500)


def test_big_m_scale_fallback_without_load():
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Bus", "bus")

    assert n.c.generators._infer_committable_big_m_scale() == pytest.approx(1e6)


def test_component_api_get_committable_big_m_values():
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 200])
    add_com_ext_generator(n, "gen", p_nom_max=np.inf)

    c = n.c.generators
    _, max_pu = c.get_bounds_pu(attr="p")
    M = c.get_committable_big_m_values(
        names=c.static.index.intersection(["gen"]),
        max_pu=max_pu,
        committable_big_m=123.0,
    )
    assert M.sel(name="gen") == pytest.approx(123.0)


def test_component_api_get_committable_big_m_values_without_max_pu():
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 200])
    add_com_ext_generator(n, "gen", p_nom_max=np.inf)

    M = n.c.generators.get_committable_big_m_values(
        names=pd.Index(["gen"]),
        committable_big_m=321.0,
    )
    assert M.sel(name="gen") == pytest.approx(321.0)


def test_big_m_validation(base_network):
    n = base_network
    add_com_ext_generator(n, "uc_gen")

    n.optimize.create_model(committable_big_m=1000)

    with pytest.raises(ValueError, match="must be finite"):
        n.optimize.create_model(committable_big_m=np.inf)

    with pytest.raises(ValueError, match="must be finite"):
        n.optimize.create_model(committable_big_m=np.nan)

    with pytest.raises(ValueError, match="must be positive"):
        n.optimize.create_model(committable_big_m=0)

    with pytest.raises(ValueError, match="must be positive"):
        n.optimize.create_model(committable_big_m=-100)


@pytest.mark.parametrize(
    ("p_min_pu", "expect_nonneg"),
    [
        (0.0, True),
        (-0.1, False),
    ],
)
def test_non_negative_constraint(base_network, p_min_pu, expect_nonneg):
    n = base_network
    add_com_ext_generator(n, "uc_gen", p_min_pu=p_min_pu, p_max_pu=0.8)

    n.optimize.create_model()

    constraint_names = list(n.model.constraints)
    has_nonneg = any(
        "Generator-com-ext-p-lower-nonneg" in name for name in constraint_names
    )
    assert has_nonneg == expect_nonneg


def test_committable_extendable_can_switch_off():
    n = pypsa.Network()
    n.set_snapshots(range(2))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 0])

    add_com_ext_generator(
        n, "uc_gen", p_nom_max=200, p_min_pu=0.4, start_up_cost=200, shut_down_cost=100
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"
    assert n.c["Generator"].static.loc["uc_gen", "p_nom_opt"] > 0

    status_values = n.c["Generator"].dynamic["status"]["uc_gen"]
    dispatch_values = n.c["Generator"].dynamic["p"]["uc_gen"]

    assert status_values.iloc[0] > 0.5
    assert dispatch_values.iloc[0] == pytest.approx(100, rel=1e-6, abs=1e-6)
    assert status_values.iloc[1] < 0.5
    assert abs(dispatch_values.iloc[1]) < 1e-6


def test_big_m_warning_emitted(caplog):
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "uc_gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        p_nom_max=100,
    )

    n.c["Generator"].static.loc["uc_gen", "p_nom_opt"] = 60
    n.c["Generator"].static.loc["uc_gen", "p_max_pu"] = 0.2
    n.c["Generator"].dynamic["p_max_pu"]["uc_gen"] = 0.2

    caplog.set_level("WARNING", logger="pypsa.consistency")
    check_big_m_exceeded(n)

    assert any("big-M bounds" in record.message for record in caplog.records)


def test_many_generators_performance():
    n = pypsa.Network()
    n.set_snapshots(range(8))
    n.add("Bus", "bus")

    base_load = 500
    load_pattern = [
        base_load * (0.6 + 0.4 * np.sin(2 * np.pi * t / 8)) for t in range(8)
    ]
    n.add("Load", "load", bus="bus", p_set=load_pattern)

    n_generators = 10
    for i in range(n_generators):
        is_committable = i % 3 != 0
        is_extendable = i % 2 == 0
        gen_name = f"gen_{i}"

        if is_committable and is_extendable:
            n.add(
                "Generator",
                gen_name,
                bus="bus",
                p_nom_extendable=True,
                committable=True,
                marginal_cost=30 + i * 5,
                capital_cost=50000,
                p_nom_max=200,
                p_min_pu=0.2,
                start_up_cost=500,
            )
        elif is_committable:
            n.add(
                "Generator",
                gen_name,
                bus="bus",
                p_nom=150,
                committable=True,
                marginal_cost=30 + i * 5,
                p_min_pu=0.3,
                start_up_cost=500,
            )
        elif is_extendable:
            n.add(
                "Generator",
                gen_name,
                bus="bus",
                p_nom_extendable=True,
                marginal_cost=40 + i * 5,
                capital_cost=60000,
                p_nom_max=300,
            )
        else:
            n.add("Generator", gen_name, bus="bus", p_nom=100, marginal_cost=35 + i * 5)

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"
    assert_power_balance(n)


@pytest.mark.parametrize(
    ("p_min_pu", "p_nom_max"),
    [
        (0.01, 300),
        (0.5, 500),
        (0.9, 300),
    ],
)
def test_extreme_min_pu(base_network, p_min_pu, p_nom_max):
    n = base_network
    n.loads_t.p_set["load"] = [200, 200, 200, 200]
    add_com_ext_generator(
        n, "gen", p_min_pu=p_min_pu, p_nom_max=p_nom_max, start_up_cost=100
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"


@pytest.mark.parametrize(
    ("p_min_pu", "p_nom_max", "snapshots"),
    [
        (0.0, 300, 4),
        (1.0, 250, 4),
        (0.3, np.inf, 3),
        (0.3, 800, 1),
    ],
)
def test_edge_cases(p_min_pu, p_nom_max, snapshots):
    n = pypsa.Network()
    n.set_snapshots(range(snapshots))
    n.add("Bus", "bus")
    load = (
        [200] * snapshots
        if p_min_pu == 1.0
        else [100 + 30 * i for i in range(snapshots)]
    )
    n.add("Load", "load", bus="bus", p_set=load[:snapshots])

    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        marginal_cost=50,
        capital_cost=80000,
        p_nom_max=p_nom_max,
        p_min_pu=p_min_pu,
        start_up_cost=500,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"


def test_committable_extendable_with_ramp_limits():
    n = pypsa.Network(snapshots=range(6))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[400, 600, 500, 700, 450, 550])

    n.add(
        "Generator",
        "slow_baseload",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        p_nom_max=800,
        p_min_pu=0.3,
        marginal_cost=30,
        capital_cost=400,
        ramp_limit_up=0.8,
        ramp_limit_down=0.8,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    p_nom_opt = n.c["Generator"].static.loc["slow_baseload", "p_nom_opt"]
    assert p_nom_opt > 0

    dispatch = n.c["Generator"].dynamic["p"]["slow_baseload"]
    ramp_up = n.c["Generator"].static.loc["slow_baseload", "ramp_limit_up"]
    ramp_down = n.c["Generator"].static.loc["slow_baseload", "ramp_limit_down"]

    for t in range(1, len(dispatch)):
        ramp = dispatch.iloc[t] - dispatch.iloc[t - 1]
        assert ramp <= ramp_up * p_nom_opt + 1e-6
        assert ramp >= -ramp_down * p_nom_opt - 1e-6


def test_committable_extendable_link_with_ramp_limits(two_bus_network):
    n = two_bus_network

    n.add(
        "Link",
        "ramp_link",
        bus0="bus0",
        bus1="bus1",
        p_nom_extendable=True,
        committable=True,
        p_nom_max=800,
        p_min_pu=0.2,
        marginal_cost=5,
        capital_cost=30000,
        ramp_limit_up=0.5,
        ramp_limit_down=0.5,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    p_nom_opt = n.c["Link"].static.loc["ramp_link", "p_nom_opt"]
    assert p_nom_opt > 0

    dispatch = n.c["Link"].dynamic["p0"]["ramp_link"]
    for t in range(1, len(dispatch)):
        ramp = dispatch.iloc[t] - dispatch.iloc[t - 1]
        assert ramp <= 0.5 * p_nom_opt + 1e-6
        assert ramp >= -0.5 * p_nom_opt - 1e-6


def test_committable_extendable_linearized_uc(base_network):
    n = base_network
    n.loads_t.p_set["load"] = [300, 600, 500, 400]

    add_com_ext_generator(
        n, "gas_com_ext", p_nom_max=800, start_up_cost=500, shut_down_cost=200
    )

    status, _ = n.optimize(solver_name="highs", linearized_unit_commitment=True)
    assert status == "ok"
    assert n.c["Generator"].static.loc["gas_com_ext", "p_nom_opt"] > 0


def test_committable_extendable_linearized_vs_milp():
    def create_network():
        n = pypsa.Network()
        n.set_snapshots(range(4))
        n.add("Bus", "bus")
        n.add("Load", "load", bus="bus", p_set=[200, 400, 300, 250])
        n.add(
            "Generator",
            "gen",
            bus="bus",
            p_nom_extendable=True,
            committable=True,
            marginal_cost=40,
            capital_cost=60000,
            p_nom_max=500,
            p_min_pu=0.25,
            start_up_cost=300,
            shut_down_cost=150,
        )
        return n

    n_milp = create_network()
    status_milp, _ = n_milp.optimize(solver_name="highs")
    assert status_milp == "ok"

    n_lin = create_network()
    status_lin, _ = n_lin.optimize(solver_name="highs", linearized_unit_commitment=True)
    assert status_lin == "ok"

    assert n_lin.objective <= n_milp.objective + 1e-3


@pytest.mark.parametrize(
    ("component", "mod_attr"),
    [
        ("Generator", "p_nom_mod"),
        ("Link", "p_nom_mod"),
    ],
)
def test_modular_linearized_uc_raises(component, mod_attr):
    n = pypsa.Network()
    n.set_snapshots(range(4))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Generator", "source", bus="bus0", p_nom=1000, marginal_cost=10)
    n.add("Load", "load", bus="bus1", p_set=[200, 400, 300, 250])

    if component == "Generator":
        n.add(
            "Generator",
            "modular_gen",
            bus="bus0",
            p_nom_extendable=True,
            committable=True,
            p_nom_mod=150,
            p_nom_max=750,
            marginal_cost=45,
            capital_cost=70000,
        )
    else:
        n.add(
            "Link",
            "modular_link",
            bus0="bus0",
            bus1="bus1",
            p_nom_extendable=True,
            committable=True,
            p_nom_mod=100,
            p_nom_max=500,
            marginal_cost=5,
            capital_cost=30000,
        )

    with pytest.raises(ValueError, match="linearized_unit_commitment.*modular"):
        n.optimize(solver_name="highs", linearized_unit_commitment=True)


def test_snapshot_interval_up_down_time_calculation(base_network):
    n = base_network
    n.set_snapshots(range(8))
    n.loads_t.p_set["load"] = [200] * 8

    add_com_ext_generator(n, "gen", min_up_time=2, min_down_time=2)

    status, _ = n.optimize(solver_name="highs", snapshots=n.snapshots[:4])
    assert status == "ok"

    status, _ = n.optimize(solver_name="highs", snapshots=n.snapshots[4:])
    assert status == "ok"


def test_no_start_costs(base_network):
    n = base_network
    n.loads_t.p_set["load"] = [100, 300, 200, 150]

    n.add(
        "Generator",
        "baseload",
        bus="bus",
        p_nom_extendable=True,
        marginal_cost=30,
        capital_cost=60000,
        p_nom_max=300,
    )
    add_com_ext_generator(
        n, "no_start_cost", p_min_pu=0.2, start_up_cost=0, shut_down_cost=0
    )

    status, _ = n.optimize(solver_name="highs")
    assert status in ["ok", "warning"]


def test_ramp_big_m_startup_shutdown_limits():
    n = pypsa.Network(snapshots=range(6))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[120, 520, 120, 520, 120, 120])

    n.add(
        "Generator",
        "cheap",
        bus="bus",
        p_nom=180,
        marginal_cost=10,
    )

    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        p_nom_max=800,
        p_min_pu=0.1,
        marginal_cost=30,
        capital_cost=400,
        ramp_limit_up=0.5,
        ramp_limit_down=0.5,
        ramp_limit_start_up=0.4,
        ramp_limit_shut_down=0.3,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    p = n.c["Generator"].dynamic["p"]["gen"]
    p_nom_opt = n.c["Generator"].static.loc["gen", "p_nom_opt"]
    status_var = n.model.variables["Generator-status"].solution.sel(name="gen")
    start_up_var = n.model.variables["Generator-start_up"].solution.sel(name="gen")
    shut_down_var = n.model.variables["Generator-shut_down"].solution.sel(name="gen")
    status_values = status_var.values
    start_up_values = start_up_var.values
    shut_down_values = shut_down_var.values
    assert p_nom_opt > 0
    assert (start_up_values > 0.5).any()
    assert (shut_down_values > 0.5).any()

    for t in range(1, len(p)):
        ramp = p.iloc[t] - p.iloc[t - 1]
        assert ramp <= 0.5 * p_nom_opt + 1e-5
        assert ramp >= -0.5 * p_nom_opt - 1e-5
        if start_up_values[t] > 0.5:
            assert ramp <= 0.4 * p_nom_opt + 1e-5
        if shut_down_values[t] > 0.5:
            assert ramp >= -0.3 * p_nom_opt - 1e-5
        if status_values[t - 1] > 0.5 and status_values[t] > 0.5:
            assert ramp <= 0.5 * p_nom_opt + 1e-5
            assert ramp >= -0.5 * p_nom_opt - 1e-5


def test_ramp_big_m_constraint_names():
    n = pypsa.Network(snapshots=range(4))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 200, 150, 120])

    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        p_nom_max=500,
        p_min_pu=0.3,
        marginal_cost=50,
        capital_cost=50000,
        ramp_limit_up=0.8,
        ramp_limit_down=0.8,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    constraints = list(n.model.constraints)
    assert "Generator-p-ramp_limit_up-run-bigM" in constraints
    assert "Generator-p-ramp_limit_up-start-bigM" in constraints
    assert "Generator-p-ramp_limit_down-run-bigM" in constraints
    assert "Generator-p-ramp_limit_down-shut-bigM" in constraints


def test_ramp_big_m_with_explicit_big_m():
    n = pypsa.Network(snapshots=range(4))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 300, 200, 150])

    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom_extendable=True,
        committable=True,
        p_nom_max=np.inf,
        p_min_pu=0.2,
        marginal_cost=40,
        capital_cost=50000,
        ramp_limit_up=0.6,
        ramp_limit_down=0.6,
    )

    status, _ = n.optimize(solver_name="highs", committable_big_m=1000)
    assert status == "ok"
    assert n.c["Generator"].static.loc["gen", "p_nom_opt"] > 0

    con_up_run = n.model.constraints["Generator-p-ramp_limit_up-run-bigM"]
    con_down_run = n.model.constraints["Generator-p-ramp_limit_down-run-bigM"]
    assert np.isclose(con_up_run.lhs.coeffs.values, 1000).any()
    assert np.isclose(con_down_run.lhs.coeffs.values, -1000).any()


def test_ramp_big_m_coexistence_com_fix_and_com_ext(base_network):
    n = base_network
    n.loads_t.p_set["load"] = [200, 400, 300, 250]

    n.add(
        "Generator",
        "com_fix",
        bus="bus",
        committable=True,
        p_nom=500,
        p_min_pu=0.3,
        marginal_cost=20,
        ramp_limit_up=0.6,
        ramp_limit_down=0.6,
    )

    add_com_ext_generator(
        n,
        "com_ext",
        p_nom_max=400,
        marginal_cost=50,
        ramp_limit_up=0.5,
        ramp_limit_down=0.5,
    )

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    constraints = list(n.model.constraints)
    assert "Generator-p-ramp_limit_up" in constraints
    assert "Generator-p-ramp_limit_up-run-bigM" in constraints
