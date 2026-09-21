# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for offset costs (`capital_cost_offset`, `overnight_cost_offset`).

A non-zero offset cost on an extendable asset introduces a binary build decision
`{component}-built`, which decides whether the asset is built at all,
independently of how much capacity is built.
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
    """Add a cheap, modular, offset-cost generator to the network."""
    defaults = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "p_nom_mod": 100,
        "marginal_cost": 10,
        "capital_cost": 10,
        "capital_cost_offset": 500,
    }
    n.add("Generator", "gas", **(defaults | kwargs))
    return n


def test_build_variables_and_constraints(base_network):
    """A offset-cost component adds a binary build variable to the model."""
    n = add_modular_generator(base_network)
    n.optimize.create_model()

    assert "Generator-built" in n.model.variables
    built = n.model["Generator-built"]
    assert built.attrs["binary"]
    # The build decision is per asset, not per snapshot.
    assert list(built.dims) == ["name"]
    assert list(built.indexes["name"]) == ["gas"]

    # The capacity is bounded by the build decision.
    assert "Generator-ext-p_nom-upper-built" in n.model.constraints


def test_no_build_variables_without_offset_cost(base_network):
    """No build decision is created if no offset cost is set."""
    n = add_modular_generator(base_network, capital_cost_offset=0)
    n.optimize.create_model()

    assert "Generator-built" not in n.model.variables
    assert "Generator-ext-p_nom-upper-built" not in n.model.constraints


def test_build_worthwhile(base_network):
    """A cheap offset cost is paid and the asset is built."""
    n = add_modular_generator(base_network, capital_cost_offset=500)
    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    assert n.generators.at["gas", "built"] == 1
    assert n.generators.at["gas", "p_nom_opt"] == 200

    # marginal + capital + offset cost
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


def test_build_not_worthwhile(base_network):
    """A prohibitive offset cost blocks both the build and the capacity."""
    n = add_modular_generator(base_network, capital_cost_offset=1e6)
    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "built"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


SAVINGS = 450 * (100 - 10) - 200 * 10


@pytest.mark.parametrize(
    ("capital_cost_offset", "expected"),
    [(SAVINGS - 100, 1), (SAVINGS + 100, 0)],
)
def test_build_just_below_and_above_break_even(
    base_network, capital_cost_offset, expected
):
    """Build flips from 1 to 0 as the offset cost crosses break-even."""
    n = add_modular_generator(base_network, capital_cost_offset=capital_cost_offset)
    n.optimize()
    assert n.generators.at["gas", "built"] == expected


def test_built_only_for_offset_costs(base_network):
    """Plain components get no build result."""
    n = add_modular_generator(base_network)
    n.optimize()

    assert n.generators.at["gas", "built"] == 1
    assert np.isnan(n.generators.at["backup", "built"])


def test_overnight_cost_offset_is_annuitised(base_network):
    """`overnight_cost_offset` is periodized like `overnight_cost`."""
    overnight, rate, lifetime = 1e6, 0.07, 25
    n = add_modular_generator(
        base_network,
        capital_cost_offset=0,
        overnight_cost_offset=overnight,
        discount_rate=rate,
        lifetime=lifetime,
    )
    nyears = n.c.generators.nyears
    expected = overnight * annuity(rate, lifetime) * nyears

    assert n.c.generators.capital_cost_offset["gas"] == pytest.approx(expected)

    n.optimize()
    assert n.generators.at["gas", "built"] == 1
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + expected)


def test_overnight_cost_offset_takes_precedence(base_network):
    """`overnight_cost_offset` overrides a directly given `capital_cost_offset`."""
    n = add_modular_generator(
        base_network,
        capital_cost_offset=12345,
        overnight_cost_offset=1e6,
        discount_rate=0.0,
        lifetime=10,
    )
    nyears = n.c.generators.nyears
    assert n.c.generators.capital_cost_offset["gas"] == pytest.approx(1e6 / 10 * nyears)


COMPONENT_KWARGS = {
    "Generator": {"bus": "bus", "p_nom_extendable": True, "marginal_cost": 1},
    "Link": {"bus0": "bus1", "bus1": "bus", "p_nom_extendable": True},
    "Process": {"bus0": "bus1", "bus1": "bus", "p_nom_extendable": True},
    "Line": {"bus0": "bus1", "bus1": "bus", "x": 0.1, "s_nom_extendable": True},
    "Store": {"bus": "bus", "e_nom_extendable": True},
    "StorageUnit": {"bus": "bus", "p_nom_extendable": True},
}


@pytest.mark.parametrize("component", list(COMPONENT_KWARGS))
def test_prohibitive_capital_cost_offset_blocks_all_components(base_network, component):
    """A prohibitive offset cost prevents the build for every component type."""
    n = base_network
    n.add("Bus", "bus1")
    n.add("Generator", "cheap", bus="bus1", p_nom=1000, marginal_cost=1)

    nom_attr = {"Line": "s_nom", "Store": "e_nom"}.get(component, "p_nom")
    kwargs = COMPONENT_KWARGS[component] | {
        f"{nom_attr}_max": 500,
        f"{nom_attr}_mod": 100,
        "capital_cost_offset": 1e9,
    }
    n.add(component, "asset", **kwargs)

    status, _ = n.optimize()

    assert status == "ok"
    static = n.c[component].static
    assert static.at["asset", "built"] == 0
    assert static.at["asset", f"{nom_attr}_opt"] == 0


def test_offset_cost_link(base_network):
    """A offset-cost link is built when the transfer is worth it."""
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
        p_nom_mod=100,
        capital_cost=10,
        capital_cost_offset=500,
    )
    n.optimize()

    assert n.links.at["link", "built"] == 1
    assert n.links.at["link", "p_nom_opt"] == 200
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


def test_offset_cost_committable_generator(base_network):
    """Build decision can be combined with unit commitment."""
    n = base_network
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        p_nom_mod=100,
        committable=True,
        p_min_pu=0.3,
        marginal_cost=10,
        capital_cost=10,
        capital_cost_offset=1e6,
        # No module is committed before the horizon starts, so nothing forces
        # the generator to be built (see the modular committable formulation).
        status=0,
        up_time_before=0,
    )
    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "built"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0


def add_committable_generator(n, **kwargs):
    """Add a cheap, non-modular, committable and offset-cost generator."""
    defaults = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "committable": True,
        "p_min_pu": 0.5,
        "marginal_cost": 1,
        "capital_cost": 1,
        "capital_cost_offset": 1,
        # Nothing is committed before the horizon starts.
        "status": 0,
        "up_time_before": 0,
    }
    n.add("Generator", "gas", **(defaults | kwargs))
    return n


def test_committable_build_constraints(base_network):
    """Committable offset-cost assets can only be committed if built."""
    n = add_committable_generator(base_network)
    n.optimize.create_model()

    assert "Generator-ext-p_nom-upper-built" in n.model.constraints
    assert "Generator-status-built" in n.model.constraints


def test_committable_build_worthwhile(base_network):
    """A committable offset-cost is built and only committed when needed.

    The load is zero in the last snapshot, so the generator must be able to shut
    down without forfeiting its capacity.
    """
    n = base_network
    n.loads_t.p_set["load"] = [100, 200, 0]
    add_committable_generator(n)

    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "built"] == 1
    assert n.generators.at["gas", "p_nom_opt"] == pytest.approx(200)
    assert n.generators_t.status["gas"].tolist() == [1, 1, 0]
    # marginal (300) + capital (200) + offset cost (1)
    assert n.objective == pytest.approx(501)


def test_committable_build_not_worthwhile(base_network):
    """A prohibitive offset cost blocks build, capacity and commitment."""
    n = base_network
    n.loads_t.p_set["load"] = [100, 200, 0]
    add_committable_generator(n, capital_cost_offset=1e6)

    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "built"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0
    assert (n.generators_t.status["gas"] == 0).all()
    assert n.objective == pytest.approx(300 * 100)


def test_status_requires_build(base_network):
    """An unbuilt committable component can never be committed."""
    n = base_network
    n.loads_t.p_set["load"] = [100, 200, 0]
    add_committable_generator(n, capital_cost_offset=1e6)
    n.optimize()

    built = n.generators.at["gas", "built"]
    assert (n.generators_t.status["gas"] <= built).all()


def test_uncommitted_offset_cost_does_not_consume(base_network):
    """A built but uncommitted asset with fixed per-unit output cannot run backwards.

    Negative dispatch would earn the marginal cost of 200 while the backup
    replaces it at 100, so any leak in the dispatch bounds is exploited.
    """
    n = add_committable_generator(
        base_network, p_min_pu=0.5, p_max_pu=0.5, marginal_cost=200
    )
    n.optimize()

    assert (n.generators_t.p["gas"] >= -1e-6).all()
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_linearized_unit_commitment_keeps_build_decision_binary(base_network):
    """Linearized unit commitment relaxes the status but not the build decision."""
    n = base_network
    n.loads_t.p_set["load"] = [100, 200, 0]
    add_committable_generator(n, capital_cost_offset=500, start_up_cost=50)
    n.optimize()
    milp_objective = n.objective

    n.optimize(linearized_unit_commitment=True)

    assert n.generators.at["gas", "built"] in (0, 1)
    assert n.objective <= milp_objective + 1e-6


def test_build_with_scenarios(base_network):
    """The build decision is shared across scenarios."""
    n = base_network
    n.set_scenarios({"low": 0.5, "high": 0.5})
    add_modular_generator(n, capital_cost_offset=500)

    n.optimize.create_model()
    built = n.model["Generator-built"]
    assert "scenario" not in built.dims

    n.optimize()
    built = n.generators.xs("gas", level="name")["built"]
    assert (built == 1).all()
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


def test_build_with_investment_periods():
    """Offset cost are weighted by the investment period objective weightings."""
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
        marginal_cost=10,
        capital_cost_offset=500,
        build_year=2020,
        lifetime=100,
    )
    status, _ = n.optimize(multi_investment_periods=True)

    assert status == "ok"
    assert n.generators.at["gas", "built"] == 1
    assert n.generators.at["gas", "p_nom_opt"] == 100
    operating = (3 + 7) * 2 * 100 * 10
    unit = 500 * (3 + 7)
    assert n.objective == pytest.approx(operating + unit)


def test_continuous_build_blocks_capacity(base_network):
    """A prohibitive offset cost blocks a non-modular build and its capacity."""
    n = add_modular_generator(base_network, p_nom_mod=0, capital_cost_offset=1e6)
    n.optimize()

    assert n.generators.at["gas", "built"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_continuous_build_worthwhile(base_network):
    """A cheap non-modular offset-cost is built and sized freely."""
    n = add_modular_generator(base_network, p_nom_mod=0, capital_cost_offset=500)
    n.optimize()

    assert n.generators.at["gas", "built"] == 1
    assert n.generators.at["gas", "p_nom_opt"] == pytest.approx(200)
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


def test_continuous_build_without_nom_max(base_network):
    """Build constraints fall back to big-M when the capacity is unbounded."""
    n = add_modular_generator(
        base_network, p_nom_mod=0, p_nom_max=inf, capital_cost_offset=1e9
    )
    n.optimize()

    assert n.generators.at["gas", "built"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_continuous_build_constraints(base_network):
    """Non-modular offset-cost assets link capacity to the build decision."""
    n = add_modular_generator(base_network, p_nom_mod=0)
    n.optimize.create_model()

    assert "Generator-ext-p_nom-upper-built" in n.model.constraints
    assert "Generator-status-built" not in n.model.constraints


def test_mixed_offset_cost_flavours(base_network):
    """Modular, committable and plain offset-cost assets can coexist on one component."""
    n = base_network
    common = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "marginal_cost": 10,
        "capital_cost": 10,
        "capital_cost_offset": 1e6,
    }
    n.add("Generator", "plain", **common)
    n.add("Generator", "mod", p_nom_mod=100, **common)
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
    built = n.generators.loc[["plain", "mod", "com"], "built"]
    assert (built == 0).all()
    assert (n.generators.loc[["plain", "mod", "com"], "p_nom_opt"] == 0).all()
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_offset_cost_ignored_for_non_extendable(base_network):
    """Like `capital_cost`, the offset cost only applies to extendable assets."""
    n = base_network
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom=300,
        marginal_cost=10,
        capital_cost_offset=500,
    )
    n.optimize()

    assert "Generator-built" not in n.model.variables
    assert n.objective == pytest.approx(450 * 10)


def test_maintainable_committable_offset_cost(base_network):
    """A non-modular committable offset-cost respects its maintenance outage."""
    n = base_network
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        committable=True,
        maintainable=True,
        maintenance_duration=1,
        marginal_cost=10,
        capital_cost_offset=500,
    )

    n.optimize()

    # Maintenance falls on the lowest-load snapshot, which the backup then serves.
    assert n.generators.at["gas", "built"] == 1
    assert n.generators_t.p["gas"].tolist() == pytest.approx([0, 200, 150])
    assert n.objective == pytest.approx(100 * 100 + 350 * 10 + 500)


def test_inactive_offset_cost_extendable(base_network):
    """An inactive offset-cost extendable does not crash and is not built."""
    n = base_network
    add_modular_generator(n, capital_cost_offset=500)
    n.add(
        "Generator",
        "gas_off",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        marginal_cost=10,
        capital_cost=10,
        capital_cost_offset=500,
        active=False,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert "gas_off" not in n.model["Generator-built"].indexes["name"]
    assert n.generators.at["gas", "built"] == 1
    assert np.isnan(n.generators.at["gas_off", "built"])


def test_mixed_offset_cost_and_non_offset_cost_with_min(base_network):
    """A forced plain extendable coexists with an unbought offset-cost."""
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
        capital_cost_offset=1e9,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["forced", "p_nom_opt"] >= 50 - 1e-6
    assert n.generators.at["opt", "built"] == 0
    assert n.generators.at["opt", "p_nom_opt"] == pytest.approx(0, abs=1e-6)


def test_small_module_size_offset_cost(base_network):
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
        marginal_cost=10,
        capital_cost_offset=1,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "built"] == 1
    assert n.generators.at["gas", "p_nom_opt"] == pytest.approx(1.0)


def test_infinite_nom_max_with_low_max_pu(base_network):
    """An unbounded capacity with max_pu < 1 is not silently capped by the big-M."""
    n = base_network
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=inf,
        p_max_pu=0.1,
        marginal_cost=10,
        capital_cost=0.01,
        capital_cost_offset=1,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "built"] == 1
    # Peak load is 200 and p_max_pu is 0.1, so 2000 MW of capacity are needed.
    # The old big-M scaled by max_pu would have capped this near the peak load.
    assert n.generators.at["gas", "p_nom_opt"] == pytest.approx(2000)
    assert n.generators_t.p["backup"].sum() == pytest.approx(0, abs=1e-6)


def test_committable_link_negative_p_min_pu_offset_cost(base_network):
    """A committable offset-cost link with a negative p_min_pu solves consistently."""
    n = base_network
    n.add("Bus", "bus1")
    n.add("Generator", "cheap", bus="bus1", p_nom=1000, marginal_cost=1)
    n.add(
        "Link",
        "link",
        bus0="bus1",
        bus1="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        committable=True,
        p_min_pu=-1,
        status=0,
        up_time_before=0,
        marginal_cost=1,
        capital_cost_offset=1,
    )

    status, _ = n.optimize()

    assert status == "ok"
    built = n.links.at["link", "built"]
    assert (n.links_t.status["link"] <= built + 1e-6).all()


def test_capex_reconciles_with_objective(base_network):
    """`n.statistics.capex()` includes the offset cost and reconciles the objective."""
    n = add_modular_generator(base_network, capital_cost_offset=500)
    n.optimize()

    capex = n.statistics.capex().sum()
    opex = n.statistics.opex().sum()
    assert capex == pytest.approx(200 * 10 + 500)
    assert capex + opex == pytest.approx(n.objective)
    assert n.statistics.expanded_capex().sum() == pytest.approx(capex)


def test_overnight_cost_includes_offset(base_network):
    """`overnight_cost` adds the overnight offset cost of built assets."""
    n = add_modular_generator(
        base_network, capital_cost_offset=500, discount_rate=0.07, lifetime=25
    )
    n.optimize()

    c = n.c.generators
    expected = 200 * c.overnight_cost["gas"] + c.overnight_cost_offset["gas"]
    assert n.statistics.overnight_cost().sum() == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Attribute matrix
#
# The build formulation interacts with modularity, unit commitment, minimum
# part-load and the capacity bounds. The tests below sweep every combination of
# those switches and assert that the build decision responds to the offset cost
# and that the resulting solution is self-consistent.
# --------------------------------------------------------------------------- #

MATRIX = list(itertools.product(*[[False, True]] * 5))
MATRIX_IDS = [
    "-".join(
        flag
        for flag, on in zip(
            ["mod", "com", "p_min_pu", "p_nom_max", "p_nom_min"], combo, strict=True
        )
        if on
    )
    or "plain"
    for combo in MATRIX
]


def add_matrix_generator(n, mod, com, p_min_pu, p_nom_max, p_nom_min, **cost):
    """Add a offset-cost generator with the given combination of attributes."""
    kwargs = {
        "bus": "bus",
        "marginal_cost": 10,
        "capital_cost": 10,
        "p_nom_extendable": True,
        "p_nom_max": 500 if p_nom_max else inf,
        "p_nom_min": 100 if p_nom_min else 0,
    }
    if mod:
        kwargs["p_nom_mod"] = 100
    if com:
        # Nothing committed before the horizon, so the build is unforced.
        kwargs |= {"committable": True, "status": 0, "up_time_before": 0}
    if p_min_pu:
        kwargs["p_min_pu"] = 0.5
    n.add("Generator", "gas", **(kwargs | cost))
    return n


def assert_solution_consistent(n, mod, com, p_min_pu, p_nom_max, p_nom_min):
    """Check invariants that must hold for any offset-cost solution."""
    p_nom = n.generators.at["gas", "p_nom_opt"]
    built = n.generators.at["gas", "built"]
    p = n.generators_t.p["gas"]

    assert built in (0.0, 1.0)
    assert (p >= -1e-6).all()
    if built == 0:
        assert p_nom == pytest.approx(0, abs=1e-6)
    elif p_nom_min:
        assert p_nom >= 100 - 1e-6
    if p_nom_max:
        assert p_nom <= 500 + 1e-6
    if mod:
        assert p_nom % 100 == pytest.approx(0, abs=1e-6)
    if com:
        status = n.generators_t.status["gas"]
        if mod:
            # For modular committables the status counts committed modules.
            assert (status * 100 <= p_nom + 1e-6).all()
            if built == 0:
                assert (status == 0).all()
        else:
            assert (status <= built + 1e-6).all()
        if p_min_pu:
            committed = status > 0.5
            assert (p[committed] >= 0.5 * p_nom - 1e-4).all()

    # The objective must be exactly reproducible from the solution.
    expected = (
        10 * p.sum()
        + 100 * n.generators_t.p["backup"].sum()
        + 10 * p_nom
        + n.c.generators.capital_cost_offset["gas"] * built
    )
    assert n.objective == pytest.approx(expected)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_cheap_capital_cost_offset_is_built(base_network, combo):
    """A cheap offset cost is paid for every attribute combination."""
    n = add_matrix_generator(base_network, *combo, capital_cost_offset=500)
    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    assert n.generators.at["gas", "built"] == 1
    assert n.generators.at["gas", "p_nom_opt"] > 0
    assert_solution_consistent(n, *combo)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_prohibitive_capital_cost_offset_blocks_build(base_network, combo):
    """A prohibitive offset cost blocks build and capacity in every combination."""
    n = add_matrix_generator(base_network, *combo, capital_cost_offset=1e9)
    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    assert n.generators.at["gas", "built"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)
    assert_solution_consistent(n, *combo)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_overnight_cost_offset(base_network, combo):
    """`overnight_cost_offset` is annuitised and wins over `capital_cost_offset` throughout."""
    n = add_matrix_generator(
        base_network,
        *combo,
        capital_cost_offset=1e9,
        overnight_cost_offset=1e6,
        discount_rate=0.07,
        lifetime=25,
    )
    expected = 1e6 * annuity(0.07, 25) * n.c.generators.nyears
    assert n.c.generators.capital_cost_offset["gas"] == pytest.approx(expected)

    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    # The annuitised cost is small, so the asset is built despite `capital_cost_offset`.
    assert n.generators.at["gas", "built"] == 1
    assert_solution_consistent(n, *combo)
