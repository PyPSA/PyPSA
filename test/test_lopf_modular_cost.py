# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for semi-continuous (`build_mode="semi_continuous"`) and fixed-size modular components.

Both introduce an integer variable `{component}-n_mod`. With a module size (e.g.
`p_nom_mod > 0`) it counts fixed-size modules, `p_nom = p_nom_mod * n_mod`. With
`build_mode="semi_continuous"` it is a binary build decision,
`p_nom_min * n_mod <= p_nom <= p_nom_max * n_mod`. Each build costs
`build_cost` (or `build_cost_overnight` together with `discount_rate` and
`lifetime`), irrespective of its capacity.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from numpy import inf

import pypsa
from pypsa.costs import annuity

# Cost of serving the load entirely from the expensive backup generator.
BACKUP_ONLY_COST = 450 * 100


@pytest.fixture
def base_network():
    """Three snapshots, one bus and an expensive non-extendable backup generator."""
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 200, 150])
    n.add("Generator", "backup", bus="bus", p_nom=1000, marginal_cost=100)
    return n


def add_modular_generator(n, **kwargs):
    """Add a cheap, semi-continuous generator."""
    defaults = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "marginal_cost": 10,
        "capital_cost": 10,
        "build_mode": "semi_continuous",
        "build_cost": 500,
    }
    n.add("Generator", "gas", **(defaults | kwargs))
    return n


def test_module_variables_and_constraints(base_network):
    """A semi-continuous component adds a binary build decision."""
    n = add_modular_generator(base_network)
    n.optimize.create_model()

    n_mod = n.model["Generator-n_mod"]
    assert n_mod.attrs["integer"]
    # The module decision is per asset, not per snapshot.
    assert list(n_mod.dims) == ["name"]
    assert list(n_mod.indexes["name"]) == ["gas"]
    assert n_mod.lower.item() == 0
    assert n_mod.upper.item() == 1

    # The capacity bounds are scaled by the build decision.
    assert "Generator-ext-p_nom-lower-semi-continuous" in n.model.constraints
    assert "Generator-ext-p_nom-upper-semi-continuous" in n.model.constraints
    assert "Generator-p_nom_modularity" not in n.model.constraints


def test_no_module_variables_without_modular(base_network):
    """No module machinery is created if nothing is modular."""
    n = add_modular_generator(base_network, build_mode="continuous", build_cost=0)
    n.optimize.create_model()

    assert "Generator-n_mod" not in n.model.variables
    assert "Generator-ext-p_nom-upper-semi-continuous" not in n.model.constraints


@pytest.mark.parametrize(
    ("build_mode", "p_nom_mod", "build_cost", "n_mod"),
    [
        ("continuous", 0, 0, np.nan),
        ("semi_continuous", 0, 500, 1),
        ("modular", 100, 500, 2),
    ],
)
def test_build_modes(base_network, build_mode, p_nom_mod, build_cost, n_mod):
    """Each build mode builds the needed capacity and pays its build cost."""
    n = add_modular_generator(
        base_network, build_mode=build_mode, p_nom_mod=p_nom_mod, build_cost=build_cost
    )
    n.optimize()

    static = n.c.generators.static
    np.testing.assert_equal(static.at["gas", "n_mod_opt"], n_mod)
    assert static.at["gas", "p_nom_opt"] == pytest.approx(200)
    assert n.objective == pytest.approx(
        450 * 10 + 200 * 10 + build_cost * np.nan_to_num(n_mod)
    )


def test_implicit_modular_build_mode_is_deprecated(base_network):
    """A module size with `build_mode="continuous"` warns and builds modules."""
    n = add_modular_generator(
        base_network, build_mode="continuous", p_nom_mod=100, build_cost=500
    )
    with pytest.warns(FutureWarning, match='build_mode="modular"') as record:
        n.optimize()

    assert len(record) == 1
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 2


def test_module_not_worthwhile(base_network):
    """A prohibitive build cost blocks both the module and the capacity."""
    n = add_modular_generator(base_network, build_cost=1e6)
    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 0
    assert n.c.generators.static.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


SAVINGS = 450 * (100 - 10) - 200 * 10


@pytest.mark.parametrize(
    ("build_cost", "expected"),
    [(SAVINGS - 100, 1), (SAVINGS + 100, 0)],
)
def test_module_just_below_and_above_break_even(base_network, build_cost, expected):
    """The module decision flips from 1 to 0 as the build cost crosses break-even."""
    n = add_modular_generator(base_network, build_cost=build_cost)
    n.optimize()
    assert n.c.generators.static.at["gas", "n_mod_opt"] == expected


def test_n_mod_opt_only_for_modulars(base_network):
    """Non-modular components get no module result."""
    n = add_modular_generator(base_network)
    n.optimize()

    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert np.isnan(n.c.generators.static.at["backup", "n_mod_opt"])


def test_build_cost_overnight_is_annuitised(base_network):
    """`build_cost_overnight` is periodized like `overnight_cost`."""
    overnight, rate, lifetime = 1e6, 0.07, 25
    n = add_modular_generator(
        base_network,
        build_cost=0,
        build_cost_overnight=overnight,
        discount_rate=rate,
        lifetime=lifetime,
    )
    expected = overnight * annuity(rate, lifetime) * n.c.generators.nyears
    periodized = n.c.generators.periodized_build_cost.sel(name="gas").item()
    assert periodized == pytest.approx(expected)

    n.optimize()
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + expected)


def test_zero_build_cost_not_in_objective(base_network):
    """Without a build cost, the module decision is free but still available."""
    n = add_modular_generator(base_network, build_cost=0)
    n.optimize()

    assert "Generator-n_mod" in n.model.variables
    assert n.objective == pytest.approx(450 * 10 + 200 * 10)


COMPONENT_KWARGS = {
    "Generator": {"bus": "bus", "p_nom_extendable": True, "marginal_cost": 1},
    "Link": {"bus0": "bus1", "bus1": "bus", "p_nom_extendable": True},
    "Process": {"bus0": "bus1", "bus1": "bus", "p_nom_extendable": True},
    "Line": {"bus0": "bus1", "bus1": "bus", "x": 0.1, "s_nom_extendable": True},
    "Transformer": {"bus0": "bus1", "bus1": "bus", "x": 0.1, "s_nom_extendable": True},
    "Store": {"bus": "bus", "e_nom_extendable": True},
    "StorageUnit": {"bus": "bus", "p_nom_extendable": True},
}


@pytest.mark.parametrize("module_size", [0, 100], ids=["semi_continuous", "fixed"])
@pytest.mark.parametrize("component", list(COMPONENT_KWARGS))
def test_prohibitive_build_cost_blocks_all_components(
    base_network, component, module_size
):
    """A prohibitive build cost prevents any module for every component type."""
    n = base_network
    n.add("Bus", "bus1")
    n.add("Generator", "cheap", bus="bus1", p_nom=1000, marginal_cost=1)

    nom_attr = {"Line": "s_nom", "Transformer": "s_nom", "Store": "e_nom"}.get(
        component, "p_nom"
    )
    kwargs = COMPONENT_KWARGS[component] | {
        f"{nom_attr}_max": 500,
        f"{nom_attr}_mod": module_size,
        "build_mode": "semi_continuous" if module_size == 0 else "modular",
        "build_cost": 1e9,
    }
    n.add(component, "asset", **kwargs)

    status, _ = n.optimize()

    assert status == "ok"
    static = n.c[component].static
    assert static.at["asset", "n_mod_opt"] == 0
    assert static.at["asset", f"{nom_attr}_opt"] == 0


def test_modular_link(base_network):
    """A semi-continuous link is built when the transfer is worth it."""
    n = base_network
    n.add("Bus", "bus1")
    n.add("Generator", "cheap", bus="bus1", p_nom=1000, marginal_cost=10)
    n.add(
        "Link",
        "link",
        bus0="bus1",
        bus1="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        capital_cost=10,
        build_mode="semi_continuous",
        build_cost=500,
    )
    n.optimize()

    assert n.c.links.static.at["link", "n_mod_opt"] == 1
    assert n.c.links.static.at["link", "p_nom_opt"] == pytest.approx(200)
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


# --------------------------------------------------------------------------- #
# Fixed module sizes
# --------------------------------------------------------------------------- #


def test_build_cost_charged_per_fixed_size_module(base_network):
    """With a module size, the build cost is paid for every module built."""
    n = add_modular_generator(
        base_network, build_mode="modular", p_nom_mod=100, build_cost=500
    )
    assert "gas" in n.c.generators.modulars

    n.optimize()

    assert "Generator-p_nom_modularity" in n.model.constraints
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 2
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 2 * 500)


def test_small_module_size(base_network):
    """A small module size still reaches the capacity needed to serve the load."""
    n = pypsa.Network(snapshots=range(1))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=1)
    n.add("Generator", "backup", bus="bus", p_nom=1000, marginal_cost=100)
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=2,
        p_nom_mod=0.1,
        build_mode="modular",
        marginal_cost=10,
        build_cost=1,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == pytest.approx(10)
    assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(1.0)


def test_fixed_size_modular_committable_generator(base_network):
    """Modules with a fixed size can be combined with unit commitment."""
    n = base_network
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        p_nom_mod=100,
        build_mode="modular",
        committable=True,
        p_min_pu=0.3,
        marginal_cost=10,
        capital_cost=10,
        build_cost=1e6,
        # No module is committed before the horizon starts, so nothing forces
        # the generator to be built (see the modular committable formulation).
        status=0,
        up_time_before=0,
    )
    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 0
    assert n.c.generators.static.at["gas", "p_nom_opt"] == 0


# --------------------------------------------------------------------------- #
# Semi-continuous capacity bounds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"maintainable": True, "maintenance_duration": 1},
        {"maintainable": True, "maintenance_duration": 1, "committable": True},
    ],
    ids=["plain", "maintainable", "maintainable_committable"],
)
@pytest.mark.parametrize(
    ("p_nom_min", "build_cost", "n_mod", "p_nom"),
    [(50, 1e6, 0, 0), (300, 500, 1, 300)],
    ids=["unbuilt", "built"],
)
def test_lower_bound_only_applies_if_built(
    base_network, p_nom_min, build_cost, n_mod, p_nom, kwargs
):
    """`p_nom_min` only has to be respected if the component is built."""
    n = add_modular_generator(
        base_network, p_nom_min=p_nom_min, build_cost=build_cost, **kwargs
    )
    n.optimize()

    assert n.c.generators.static.at["gas", "n_mod_opt"] == n_mod
    assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(p_nom)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"p_nom_max": inf}, "and an infinite `p_nom_max`"),
        ({"build_mode": "binary"}, "invalid `build_mode`"),
        ({"build_mode": "modular"}, "no finite module size `p_nom_mod > 0`"),
        (
            {"build_mode": "modular", "p_nom_mod": -100},
            "no finite module size `p_nom_mod > 0`",
        ),
        (
            {"build_mode": "modular", "p_nom_mod": inf},
            "no finite module size `p_nom_mod > 0`",
        ),
        (
            {"build_mode": "modular", "p_nom_mod": np.nan},
            "no finite module size `p_nom_mod > 0`",
        ),
    ],
    ids=[
        "unbounded",
        "invalid",
        "modular_no_size",
        "modular_negative_size",
        "modular_infinite_size",
        "modular_nan_size",
    ],
)
def test_unsupported_build_mode_raises(base_network, kwargs, match):
    """Invalid, contradictory or unbounded build modes fail fast."""
    n = add_modular_generator(base_network, **kwargs)

    with pytest.raises(ValueError, match=match):
        n.optimize()


# --------------------------------------------------------------------------- #
# Unit commitment
# --------------------------------------------------------------------------- #


def add_committable_generator(n, **kwargs):
    """Add a cheap, committable, semi-continuous generator."""
    defaults = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "committable": True,
        "p_min_pu": 0.5,
        "marginal_cost": 1,
        "capital_cost": 1,
        "build_mode": "semi_continuous",
        "build_cost": 1,
        # Nothing is committed before the horizon starts.
        "status": 0,
        "up_time_before": 0,
    }
    n.add("Generator", "gas", **(defaults | kwargs))
    return n


def test_committable_module_constraints(base_network):
    """Semi-continuous committables reuse the committable extendable formulation."""
    n = add_committable_generator(base_network)
    n.optimize.create_model()

    for name in [
        "Generator-ext-p_nom-upper-semi-continuous",
        "Generator-status-p_nom-variable-upper",
        "Generator-com-ext-p-lower",
        "Generator-com-ext-p-upper-bigM",
        "Generator-com-ext-p-upper-cap",
    ]:
        assert name in n.model.constraints
    assert n.model["Generator-status"].attrs["binary"]


@pytest.mark.parametrize(
    ("build_cost", "n_mod", "p_nom", "status", "objective"),
    [(1, 1, 200, [1, 1, 0], 300 + 200 + 1), (1e6, 0, 0, [0, 0, 0], 300 * 100)],
    ids=["cheap", "prohibitive"],
)
def test_committable_module_build_decision(
    base_network, build_cost, n_mod, p_nom, status, objective
):
    """A semi-continuous committable is only committed when built and needed.

    The load is zero in the last snapshot, so the generator must be able to shut
    down without forfeiting its capacity.
    """
    n = base_network
    n.c.loads.dynamic.p_set["load"] = [100, 200, 0]
    add_committable_generator(n, build_cost=build_cost)
    n.optimize()

    assert n.c.generators.static.at["gas", "n_mod_opt"] == n_mod
    assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(p_nom)
    assert n.c.generators.dynamic.status["gas"].tolist() == status
    assert n.objective == pytest.approx(objective)


@pytest.mark.parametrize("min_up_time", [0, 3])
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"build_mode": "modular", "p_nom_mod": 100}],
    ids=["semi_continuous", "modular"],
)
def test_initially_up_committable_module_can_stay_unbuilt(
    base_network, kwargs, min_up_time
):
    """An initially committed component with `n_mod` may stay unbuilt."""
    n = add_committable_generator(
        base_network,
        build_cost=1e6,
        status=1,
        up_time_before=1,
        min_up_time=min_up_time,
        **kwargs,
    )
    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"build_mode": "modular", "p_nom_mod": 100}],
    ids=["semi_continuous", "modular"],
)
def test_initially_up_committable_module_must_stay_up_if_built(base_network, kwargs):
    """A built component that is initially up stays committed for `min_up_time`."""
    n = base_network
    n.c.loads.dynamic.p_set["load"] = [0, 0, 150]
    add_committable_generator(
        n,
        p_min_pu=0,
        marginal_cost=10,
        status=1,
        up_time_before=1,
        min_up_time=3,
        stand_by_cost=1,
        **kwargs,
    )
    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] >= 1
    assert n.c.generators.dynamic.status["gas"].iloc[:2].tolist() == [1, 1]


@pytest.mark.parametrize(
    ("build_cost", "n_mod"), [(1, 1), (1e9, 0)], ids=["cheap", "prohibitive"]
)
def test_linearized_unit_commitment_keeps_build_decision(
    base_network, build_cost, n_mod
):
    """Linearized unit commitment relaxes the status, not the build decision."""
    n = add_committable_generator(base_network, build_cost=build_cost)
    status, _ = n.optimize(linearized_unit_commitment=True)

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == n_mod


def test_linearized_unit_commitment_rejects_fixed_size_modules(base_network):
    """Relaxing the integrality of fixed-size modular committables is not allowed."""
    n = add_committable_generator(base_network, build_mode="modular", p_nom_mod=100)

    with pytest.raises(ValueError, match="linearized_unit_commitment.*modular"):
        n.optimize(linearized_unit_commitment=True)


@pytest.mark.parametrize(
    ("build_cost", "n_mod"), [(1, 1), (1e9, 0)], ids=["cheap", "prohibitive"]
)
def test_maintainable_committable_semi_continuous(base_network, build_cost, n_mod):
    """A semi-continuous component can be committable and maintainable."""
    n = add_committable_generator(
        base_network, maintainable=True, maintenance_duration=1, build_cost=build_cost
    )
    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == n_mod
    if n_mod == 0:
        assert n.objective == pytest.approx(BACKUP_ONLY_COST)


@pytest.mark.parametrize(
    ("build_cost", "n_mod"), [(1, 1), (1e9, 0)], ids=["cheap", "prohibitive"]
)
def test_committable_link_negative_p_min_pu_modular(base_network, build_cost, n_mod):
    """An unbuilt committable semi-continuous link carries no reverse flow."""
    n = base_network
    n.add("Bus", "bus1")
    n.add("Generator", "cheap", bus="bus1", p_nom=1000, marginal_cost=1)
    n.add(
        "Link",
        "link",
        bus0="bus",
        bus1="bus1",
        p_nom_extendable=True,
        p_nom_max=500,
        committable=True,
        p_min_pu=-1,
        status=0,
        up_time_before=0,
        build_mode="semi_continuous",
        build_cost=build_cost,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.links.static.at["link", "n_mod_opt"] == n_mod
    assert (n.c.links.dynamic.p0["link"] < -1e-6).any() == bool(n_mod)
    if n_mod == 0:
        assert n.objective == pytest.approx(BACKUP_ONLY_COST)


# --------------------------------------------------------------------------- #
# Scenarios, investment periods and edge cases
# --------------------------------------------------------------------------- #


def test_modules_with_scenarios(base_network):
    """The module decision is shared across scenarios."""
    n = base_network
    n.set_scenarios({"low": 0.5, "high": 0.5})
    add_modular_generator(n, build_cost=500)

    n.optimize.create_model()
    n_mod = n.model["Generator-n_mod"]
    assert "scenario" not in n_mod.dims

    n.optimize()
    n_mod_opt = n.c.generators.static.xs("gas", level="name")["n_mod_opt"]
    assert (n_mod_opt == 1).all()
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


def test_modules_with_investment_periods():
    """Build costs are weighted by the investment period objective weightings."""
    n = pypsa.Network()
    n.set_snapshots(pd.MultiIndex.from_product([[2020, 2030], range(2)]))
    n.investment_periods = [2020, 2030]
    n.investment_period_weightings["objective"] = [3, 7]
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=100)
    n.add("Generator", "backup", bus="bus", p_nom=1000, marginal_cost=100)
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        p_nom_mod=100,
        build_mode="modular",
        marginal_cost=10,
        build_cost=500,
        build_year=2020,
        lifetime=100,
    )
    status, _ = n.optimize(multi_investment_periods=True)

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert n.c.generators.static.at["gas", "p_nom_opt"] == 100
    operating = (3 + 7) * 2 * 100 * 10
    modules = 500 * (3 + 7)
    assert n.objective == pytest.approx(operating + modules)


def test_mixed_modular_flavours(base_network):
    """Semi-continuous, fixed-size and committable modulars coexist on one component."""
    n = base_network
    common = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "marginal_cost": 10,
        "capital_cost": 10,
        "build_mode": "semi_continuous",
        "build_cost": 1e6,
    }
    n.add("Generator", "plain", **common)
    n.add("Generator", "mod", **(common | {"build_mode": "modular", "p_nom_mod": 100}))
    n.add(
        "Generator",
        "com",
        committable=True,
        p_min_pu=0.5,
        status=0,
        up_time_before=0,
        **common,
    )

    status, _ = n.optimize()

    assert status == "ok"
    n_mod = n.c.generators.static.loc[["plain", "mod", "com"], "n_mod_opt"]
    assert (n_mod == 0).all()
    assert (n.c.generators.static.loc[["plain", "mod", "com"], "p_nom_opt"] == 0).all()
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_inactive_modular_extendable(base_network):
    """An inactive modular extendable does not crash and gets no module."""
    n = base_network
    add_modular_generator(n, build_cost=500)
    n.add(
        "Generator",
        "gas_off",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        marginal_cost=10,
        capital_cost=10,
        build_mode="semi_continuous",
        build_cost=500,
        active=False,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert "gas_off" not in n.model["Generator-n_mod"].indexes["name"]
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert np.isnan(n.c.generators.static.at["gas_off", "n_mod_opt"])


def test_mixed_modular_and_non_modular_with_min(base_network):
    """A forced non-modular extendable coexists with an unbuilt semi-continuous one."""
    n = base_network
    n.add(
        "Generator",
        "forced",
        bus="bus",
        p_nom_extendable=True,
        p_nom_min=50,
        marginal_cost=10,
        capital_cost=10,
    )
    n.add(
        "Generator",
        "opt",
        bus="bus",
        p_nom_extendable=True,
        p_nom_min=50,
        p_nom_max=500,
        marginal_cost=10,
        capital_cost=10,
        build_mode="semi_continuous",
        build_cost=1e9,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["forced", "p_nom_opt"] >= 50 - 1e-6
    assert n.c.generators.static.at["opt", "n_mod_opt"] == 0
    assert n.c.generators.static.at["opt", "p_nom_opt"] == pytest.approx(0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_capex_reconciles_with_objective(base_network):
    """`n.statistics.capex()` includes the build cost and reconciles the objective."""
    n = add_modular_generator(base_network, build_cost=500)
    n.optimize()

    capex = n.statistics.capex().sum()
    opex = n.statistics.opex().sum()
    assert capex == pytest.approx(200 * 10 + 500)
    assert capex + opex == pytest.approx(n.objective)

    capacity = n.statistics.capex(cost_attribute="capital_cost").sum()
    modules = n.statistics.capex(cost_attribute="build_cost").sum()
    assert capacity == pytest.approx(200 * 10)
    assert modules == pytest.approx(500)
    assert capex == pytest.approx(capacity + modules)
    assert n.statistics.installed_capex(cost_attribute="build_cost").empty
    expanded = n.statistics.expanded_capex(cost_attribute="build_cost").sum()
    assert expanded == pytest.approx(500)


def test_capex_counts_fixed_size_modules(base_network):
    """The build cost in `capex` scales with the number of fixed-size modules."""
    n = add_modular_generator(
        base_network, build_mode="modular", p_nom_mod=100, build_cost=500
    )
    n.optimize()

    modules = n.statistics.capex(cost_attribute="build_cost").sum()
    assert modules == pytest.approx(2 * 500)
    capex = n.statistics.capex().sum()
    assert capex + n.statistics.opex().sum() == pytest.approx(n.objective)


@pytest.mark.parametrize("cost_attribute", [None, "capital_cost", "build_cost"])
@pytest.mark.parametrize("module_size", [0, 100], ids=["semi_continuous", "fixed"])
def test_capex_expression_matches_statistics(base_network, module_size, cost_attribute):
    """`n.optimize.expressions.capex()` evaluates to the statistics capex."""
    n = add_modular_generator(
        base_network,
        build_mode="semi_continuous" if module_size == 0 else "modular",
        p_nom_mod=module_size,
    )
    n.optimize()

    expr = n.optimize.expressions.capex(cost_attribute=cost_attribute)
    stat = n.statistics.capex(cost_attribute=cost_attribute)
    assert expr.solution.sum().item() == pytest.approx(stat.sum())


def test_transmission_expansion_cost_limit_includes_build_cost():
    """The build cost counts towards the transmission expansion cost limit."""
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", ["a", "b"])
    n.add("Load", "load", bus="b", p_set=[100, 200, 150])
    n.add("Generator", "backup", bus="b", p_nom=1000, marginal_cost=100)
    n.add("Generator", "cheap", bus="a", p_nom=1000, marginal_cost=10)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.1,
        carrier="AC",
        s_nom_extendable=True,
        s_nom_max=500,
        capital_cost=10,
        build_mode="semi_continuous",
        build_cost=1000,
    )
    n.add(
        "GlobalConstraint",
        "limit",
        type="transmission_expansion_cost_limit",
        constant=2500,
        sense="<=",
        carrier_attribute="AC",
    )
    n.optimize()

    assert n.c.lines.static.at["ab", "n_mod_opt"] == 1
    assert n.c.lines.static.at["ab", "s_nom_opt"] == pytest.approx((2500 - 1000) / 10)


def test_transmission_expansion_cost_limit_weights_build_cost_by_period():
    """Across investment periods, the build cost is weighted like the capital cost."""
    n = pypsa.Network()
    n.set_snapshots(pd.MultiIndex.from_product([[2020, 2030], range(2)]))
    n.investment_periods = [2020, 2030]
    n.investment_period_weightings["objective"] = [3, 7]
    n.add("Bus", ["a", "b"])
    n.add("Load", "load", bus="b", p_set=200)
    n.add("Generator", "backup", bus="b", p_nom=1000, marginal_cost=100)
    n.add("Generator", "cheap", bus="a", p_nom=1000, marginal_cost=10)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.1,
        carrier="AC",
        s_nom_extendable=True,
        s_nom_max=500,
        capital_cost=10,
        build_mode="semi_continuous",
        build_cost=100,
        build_year=2020,
        lifetime=100,
    )
    n.add(
        "GlobalConstraint",
        "limit",
        type="transmission_expansion_cost_limit",
        constant=11000,
        sense="<=",
        carrier_attribute="AC",
    )
    n.optimize(multi_investment_periods=True)

    weight = 3 + 7
    assert n.c.lines.static.at["ab", "n_mod_opt"] == 1
    assert n.c.lines.static.at["ab", "s_nom_opt"] == pytest.approx(
        (11000 - 100 * weight) / (10 * weight)
    )


def test_overnight_cost_composes_modules(base_network):
    """`overnight_cost` composes the capacity and module overnight terms."""
    n = add_modular_generator(
        base_network, build_cost=500, discount_rate=0.07, lifetime=25
    )
    n.optimize()

    total = n.statistics.overnight_cost().sum()
    capacity = n.statistics.overnight_cost(cost_attribute="overnight_cost").sum()
    modules = n.statistics.overnight_cost(cost_attribute="build_cost").sum()
    assert capacity == pytest.approx(200 * n.c.generators.overnight_cost["gas"])
    assert modules == pytest.approx(n.c.generators.overnight_build_cost["gas"])
    assert total == pytest.approx(capacity + modules)

    with pytest.raises(ValueError, match="cost_attribute must be"):
        n.statistics.overnight_cost(cost_attribute="capital_cost")


# --------------------------------------------------------------------------- #
# Attribute matrix
#
# The module formulation interacts with the module size, unit commitment, minimum
# part-load and the lower capacity bound. The tests below sweep every combination of
# those switches and assert that the module decision responds to the build cost
# and that the resulting solution is self-consistent.
# --------------------------------------------------------------------------- #

MATRIX = list(itertools.product(*[[False, True]] * 4))
MATRIX_IDS = [
    "-".join(
        flag
        for flag, on in zip(["mod", "com", "p_min_pu", "p_nom_min"], combo, strict=True)
        if on
    )
    or "semi_continuous"
    for combo in MATRIX
]


def add_matrix_generator(n, mod, com, p_min_pu, p_nom_min, **cost):
    """Add a semi-continuous or fixed-size modular generator."""
    kwargs = {
        "bus": "bus",
        "marginal_cost": 10,
        "capital_cost": 10,
        "build_mode": "modular" if mod else "semi_continuous",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "p_nom_min": 100 if p_nom_min else 0,
    }
    if mod:
        kwargs["p_nom_mod"] = 100
    if com:
        # Nothing committed before the horizon, so the module is unforced.
        kwargs |= {"committable": True, "status": 0, "up_time_before": 0}
    if p_min_pu:
        kwargs["p_min_pu"] = 0.5
    n.add("Generator", "gas", **(kwargs | cost))
    return n


def assert_solution_consistent(n, mod, com, p_min_pu, p_nom_min):
    """Check invariants that must hold for any modular solution."""
    p_nom = n.c.generators.static.at["gas", "p_nom_opt"]
    n_mod = n.c.generators.static.at["gas", "n_mod_opt"]
    p = n.c.generators.dynamic.p["gas"]

    assert n_mod == pytest.approx(round(n_mod))
    assert (p >= -1e-6).all()
    assert p_nom <= 500 + 1e-6
    if mod:
        assert p_nom == pytest.approx(100 * n_mod, abs=1e-6)
    else:
        assert n_mod in (0.0, 1.0)
    if n_mod == 0:
        assert p_nom == pytest.approx(0, abs=1e-6)
    elif p_nom_min:
        assert p_nom >= 100 - 1e-6
    if com:
        status = n.c.generators.dynamic.status["gas"]
        assert (status <= n_mod + 1e-6).all()
        if mod:
            # For modular committables the status counts committed modules.
            assert (status * 100 <= p_nom + 1e-6).all()
        if p_min_pu:
            committed = status > 0.5
            min_p = 0.5 * (100 * status if mod else p_nom + 0 * status)
            assert (p[committed] >= min_p[committed] - 1e-4).all()

    # The objective must be exactly reproducible from the solution.
    expected = (
        10 * p.sum()
        + 100 * n.c.generators.dynamic.p["backup"].sum()
        + 10 * p_nom
        + float(n.c.generators.periodized_build_cost.sel(name="gas").item()) * n_mod
    )
    assert n.objective == pytest.approx(expected)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_cheap_build_cost_is_paid(base_network, combo):
    """A cheap build cost is paid for every attribute combination."""
    n = add_matrix_generator(base_network, *combo, build_cost=500)
    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    assert n.c.generators.static.at["gas", "n_mod_opt"] >= 1
    assert n.c.generators.static.at["gas", "p_nom_opt"] > 0
    assert_solution_consistent(n, *combo)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_prohibitive_build_cost_blocks_modules(base_network, combo):
    """A prohibitive build cost blocks the modules unless a minimum forces one.

    For fixed-size modules, `p_nom_min` is a plain lower bound, so it forces the
    first module to be built despite its cost.
    """
    mod, _, _, p_nom_min = combo
    n = add_matrix_generator(base_network, *combo, build_cost=1e9)
    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    if mod and p_nom_min:
        assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
        assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(100)
    else:
        assert n.c.generators.static.at["gas", "n_mod_opt"] == 0
        assert n.c.generators.static.at["gas", "p_nom_opt"] == 0
        assert n.objective == pytest.approx(BACKUP_ONLY_COST)
    assert_solution_consistent(n, *combo)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_build_cost_overnight(base_network, combo):
    """`build_cost_overnight` is annuitised and wins over `build_cost` throughout."""
    n = add_matrix_generator(
        base_network,
        *combo,
        build_cost=1e9,
        build_cost_overnight=1e6,
        discount_rate=0.07,
        lifetime=25,
    )
    expected = 1e6 * annuity(0.07, 25) * n.c.generators.nyears
    periodized = n.c.generators.periodized_build_cost.sel(name="gas").item()
    assert periodized == pytest.approx(expected)

    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    # The annuitised cost is small, so modules are built despite `build_cost`.
    assert n.c.generators.static.at["gas", "n_mod_opt"] >= 1
    assert_solution_consistent(n, *combo)
