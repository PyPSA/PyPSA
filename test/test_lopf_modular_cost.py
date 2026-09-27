# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for modular components (`modular`, `n_mod_min`, `n_mod_max`) and module costs.

Modular components introduce an integer variable `{component}-n_mod` for the number
of modules. With a module size (e.g. `p_nom_mod > 0`) the capacity of each module
is fixed; without one (`modular=True`) the total capacity of the modules is a
continuous decision, by default of a single module. Each module costs
`module_cost` (or `module_cost_overnight` together with `discount_rate` and
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


def make_base_network():
    """Three snapshots, one bus and an expensive non-extendable backup generator."""
    n = pypsa.Network(snapshots=range(3))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[100, 200, 150])
    n.add("Generator", "backup", bus="bus", p_nom=1000, marginal_cost=100)
    return n


@pytest.fixture
def base_network():
    """Three snapshots, one bus and an expensive non-extendable backup generator."""
    return make_base_network()


def add_modular_generator(n, **kwargs):
    """Add a cheap generator with a single module of continuous capacity."""
    defaults = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "marginal_cost": 10,
        "capital_cost": 10,
        "modular": True,
        "module_cost": 500,
    }
    n.add("Generator", "gas", **(defaults | kwargs))
    return n


def test_module_variables_and_constraints(base_network):
    """A continuous modular component adds a module count bounded to one module."""
    n = add_modular_generator(base_network)
    n.optimize.create_model()

    n_mod = n.model["Generator-n_mod"]
    assert n_mod.attrs["integer"]
    # The module decision is per asset, not per snapshot.
    assert list(n_mod.dims) == ["name"]
    assert list(n_mod.indexes["name"]) == ["gas"]
    assert n_mod.lower.item() == 0
    assert n_mod.upper.item() == 1

    # The capacity is bounded by the module decision, not fixed by a module size.
    assert "Generator-p_nom_modularity_upper" in n.model.constraints
    assert "Generator-p_nom_modularity" not in n.model.constraints


def test_no_module_variables_without_modular(base_network):
    """No module machinery is created if nothing is modular."""
    n = add_modular_generator(base_network, modular=False)
    n.optimize.create_model()

    assert "Generator-n_mod" not in n.model.variables
    assert "Generator-p_nom_modularity_upper" not in n.model.constraints


def test_module_worthwhile(base_network):
    """A cheap module cost is paid and the asset is built."""
    n = add_modular_generator(base_network, module_cost=500)
    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(200)

    # marginal + capital + module cost
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


def test_module_not_worthwhile(base_network):
    """A prohibitive module cost blocks both the module and the capacity."""
    n = add_modular_generator(base_network, module_cost=1e6)
    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 0
    assert n.c.generators.static.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


SAVINGS = 450 * (100 - 10) - 200 * 10


@pytest.mark.parametrize(
    ("module_cost", "expected"),
    [(SAVINGS - 100, 1), (SAVINGS + 100, 0)],
)
def test_module_just_below_and_above_break_even(base_network, module_cost, expected):
    """The module decision flips from 1 to 0 as the module cost crosses break-even."""
    n = add_modular_generator(base_network, module_cost=module_cost)
    n.optimize()
    assert n.c.generators.static.at["gas", "n_mod_opt"] == expected


def test_n_mod_opt_only_for_modulars(base_network):
    """Non-modular components get no module result."""
    n = add_modular_generator(base_network)
    n.optimize()

    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert np.isnan(n.c.generators.static.at["backup", "n_mod_opt"])


def test_module_cost_overnight_is_annuitised(base_network):
    """`module_cost_overnight` is periodized like `overnight_cost`."""
    overnight, rate, lifetime = 1e6, 0.07, 25
    n = add_modular_generator(
        base_network,
        module_cost=0,
        module_cost_overnight=overnight,
        discount_rate=rate,
        lifetime=lifetime,
    )
    nyears = n.c.generators.nyears
    expected = overnight * annuity(rate, lifetime) * nyears

    assert n.c.generators.periodized_module_cost.sel(
        name="gas"
    ).item() == pytest.approx(expected)

    n.optimize()
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + expected)


def test_module_cost_overnight_takes_precedence(base_network):
    """`module_cost_overnight` overrides a directly given `module_cost`."""
    n = add_modular_generator(
        base_network,
        module_cost=12345,
        module_cost_overnight=1e6,
        discount_rate=0.0,
        lifetime=10,
    )
    nyears = n.c.generators.nyears
    assert n.c.generators.periodized_module_cost.sel(
        name="gas"
    ).item() == pytest.approx(1e6 / 10 * nyears)


def test_zero_module_cost_not_in_objective(base_network):
    """Without a module cost, the module decision is free but still available."""
    n = add_modular_generator(base_network, module_cost=0)
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


@pytest.mark.parametrize("module_size", [0, 100], ids=["continuous", "fixed"])
@pytest.mark.parametrize("component", list(COMPONENT_KWARGS))
def test_prohibitive_module_cost_blocks_all_components(
    base_network, component, module_size
):
    """A prohibitive module cost prevents any module for every component type."""
    n = base_network
    n.add("Bus", "bus1")
    n.add("Generator", "cheap", bus="bus1", p_nom=1000, marginal_cost=1)

    nom_attr = {"Line": "s_nom", "Transformer": "s_nom", "Store": "e_nom"}.get(
        component, "p_nom"
    )
    kwargs = COMPONENT_KWARGS[component] | {
        f"{nom_attr}_max": 500,
        f"{nom_attr}_mod": module_size,
        "modular": True,
        "module_cost": 1e9,
    }
    n.add(component, "asset", **kwargs)

    status, _ = n.optimize()

    assert status == "ok"
    static = n.c[component].static
    assert static.at["asset", "n_mod_opt"] == 0
    assert static.at["asset", f"{nom_attr}_opt"] == 0


def test_modular_link(base_network):
    """A continuous modular link is built when the transfer is worth it."""
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
        modular=True,
        module_cost=500,
    )
    n.optimize()

    assert n.c.links.static.at["link", "n_mod_opt"] == 1
    assert n.c.links.static.at["link", "p_nom_opt"] == pytest.approx(200)
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


# --------------------------------------------------------------------------- #
# Fixed module sizes
# --------------------------------------------------------------------------- #


def test_module_cost_charged_per_fixed_size_module(base_network):
    """With a module size, the module cost is paid for every module built."""
    n = add_modular_generator(base_network, p_nom_mod=100, module_cost=500)
    n.optimize()

    assert n.c.generators.static.at["gas", "n_mod_opt"] == 2
    assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(200)
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 2 * 500)


def test_module_size_implies_modular(base_network):
    """A module size makes a component modular without setting `modular=True`."""
    n = add_modular_generator(
        base_network, modular=False, p_nom_mod=100, module_cost=500
    )
    assert "gas" in n.c.generators.modulars

    n.optimize()

    assert "Generator-p_nom_modularity" in n.model.constraints
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 2
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 2 * 500)


def test_n_mod_max_caps_fixed_size_modules(base_network):
    """`n_mod_max` limits the number of fixed-size modules."""
    n = add_modular_generator(base_network, p_nom_mod=50, n_mod_max=2, module_cost=0)
    n.optimize()

    assert n.c.generators.static.at["gas", "n_mod_opt"] == 2
    assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(100)
    assert n.c.generators.dynamic.p["backup"].sum() == pytest.approx(150)


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
        marginal_cost=10,
        module_cost=1,
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
        committable=True,
        p_min_pu=0.3,
        marginal_cost=10,
        capital_cost=10,
        module_cost=1e6,
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
# Several continuous modules
# --------------------------------------------------------------------------- #


def test_n_mod_min_builds_several_continuous_modules(base_network):
    """`n_mod_min` forces several modules whose total capacity stays continuous."""
    n = add_modular_generator(base_network, n_mod_min=3, module_cost=500)
    n.optimize.create_model()
    assert n.model["Generator-n_mod"].upper.item() == 3

    n.optimize()

    assert n.c.generators.static.at["gas", "n_mod_opt"] == 3
    assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(200)
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 3 * 500)


def test_n_mod_max_allows_several_continuous_modules(base_network):
    """Additional continuous modules are only built if required."""
    n = add_modular_generator(base_network, n_mod_max=4, module_cost=500)
    n.optimize()

    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(200)


def test_lower_bound_only_semi_continuous_for_single_module(base_network):
    """`p_nom_min` only has to be respected if built for a single continuous module."""
    n = add_modular_generator(base_network, p_nom_min=50, module_cost=1e6)
    n.optimize()
    assert "Generator-ext-p_nom-lower-modular" in n.model.constraints
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 0
    assert n.c.generators.static.at["gas", "p_nom_opt"] == 0

    # With several modules allowed, `p_nom_min` is a plain lower bound again.
    n = add_modular_generator(
        make_base_network(), n_mod_max=3, p_nom_min=50, module_cost=1e6
    )
    n.optimize()
    assert "Generator-ext-p_nom-lower-modular" not in n.model.constraints
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert n.c.generators.static.at["gas", "p_nom_opt"] >= 50 - 1e-6


@pytest.mark.parametrize(
    "bounds",
    [{"n_mod_min": 2, "n_mod_max": 1}, {"n_mod_min": 1.5}, {"n_mod_max": 2.5}],
)
def test_invalid_module_bounds_raise(base_network, bounds):
    """The number of modules must be bounded by ordered, whole numbers."""
    n = add_modular_generator(base_network, **bounds)

    with pytest.raises(ValueError, match="invalid module bounds"):
        n.optimize()


# --------------------------------------------------------------------------- #
# Unit commitment
# --------------------------------------------------------------------------- #


def add_committable_generator(n, **kwargs):
    """Add a cheap, committable generator with a single continuous module."""
    defaults = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "committable": True,
        "p_min_pu": 0.5,
        "marginal_cost": 1,
        "capital_cost": 1,
        "modular": True,
        "module_cost": 1,
        # Nothing is committed before the horizon starts.
        "status": 0,
        "up_time_before": 0,
    }
    n.add("Generator", "gas", **(defaults | kwargs))
    return n


def test_committable_module_constraints(base_network):
    """Continuous modular committables track the capacity available per snapshot."""
    n = add_committable_generator(base_network)
    n.optimize.create_model()

    available = n.model["Generator-available_p_nom"]
    assert set(available.dims) == {"name", "snapshot"}

    for name in [
        "Generator-p_nom_modularity_upper",
        "Generator-status-p_nom-variable-upper",
        "Generator-p_nom_available_continuous",
        "Generator-p_nom_available_binary",
        "Generator-p_nom_available_switch",
        "Generator-com-mod-continuous-p-lower",
        "Generator-com-mod-continuous-p-upper",
    ]:
        assert name in n.model.constraints
    # The status of a continuous module stays binary.
    assert n.model["Generator-status"].attrs["binary"]


def test_committable_module_worthwhile(base_network):
    """A continuous modular committable is built and only committed when needed.

    The load is zero in the last snapshot, so the generator must be able to shut
    down without forfeiting its capacity.
    """
    n = base_network
    n.c.loads.dynamic.p_set["load"] = [100, 200, 0]
    add_committable_generator(n)

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(200)
    assert n.c.generators.dynamic.status["gas"].tolist() == [1, 1, 0]
    # marginal (300) + capital (200) + module cost (1)
    assert n.objective == pytest.approx(501)


def test_committable_module_not_worthwhile(base_network):
    """A prohibitive module cost blocks the module, capacity and commitment."""
    n = base_network
    n.c.loads.dynamic.p_set["load"] = [100, 200, 0]
    add_committable_generator(n, module_cost=1e6)

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 0
    assert n.c.generators.static.at["gas", "p_nom_opt"] == 0
    assert (n.c.generators.dynamic.status["gas"] == 0).all()
    assert n.objective == pytest.approx(300 * 100)


def test_status_requires_module(base_network):
    """A committable component without a module can never be committed."""
    n = base_network
    n.c.loads.dynamic.p_set["load"] = [100, 200, 0]
    add_committable_generator(n, module_cost=1e6)
    n.optimize()

    n_mod = n.c.generators.static.at["gas", "n_mod_opt"]
    assert (n.c.generators.dynamic.status["gas"] <= n_mod).all()


def test_initially_up_committable_module_can_stay_unbuilt(base_network):
    """An initially committed continuous module may shut down without being built."""
    n = add_committable_generator(
        base_network, module_cost=1e6, status=1, up_time_before=1
    )
    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_committable_several_continuous_modules_raise(base_network):
    """Unit commitment is only supported for a single continuous module."""
    n = add_committable_generator(base_network, n_mod_max=2)

    with pytest.raises(ValueError, match="allow more than one module"):
        n.optimize()


def test_linearized_unit_commitment_rejects_modulars(base_network):
    """Relaxing the integrality of modular committables is not allowed."""
    n = base_network
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        committable=True,
        marginal_cost=10,
        modular=True,
        module_cost=500,
    )

    with pytest.raises(ValueError, match="linearized_unit_commitment.*modular"):
        n.optimize(linearized_unit_commitment=True)


def test_maintainable_committable_continuous_modular_raises(base_network):
    """A continuous modular that is committable and maintainable is rejected."""
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
        modular=True,
        module_cost=500,
    )

    with pytest.raises(ValueError, match="also maintainable"):
        n.optimize()


def test_committable_link_negative_p_min_pu_modular(base_network):
    """A committable modular link with a negative p_min_pu solves consistently."""
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
        modular=True,
        module_cost=1,
    )

    status, _ = n.optimize()

    assert status == "ok"
    n_mod = n.c.links.static.at["link", "n_mod_opt"]
    assert (n.c.links.dynamic.status["link"] <= n_mod + 1e-6).all()


# --------------------------------------------------------------------------- #
# Scenarios, investment periods and edge cases
# --------------------------------------------------------------------------- #


def test_modules_with_scenarios(base_network):
    """The module decision is shared across scenarios."""
    n = base_network
    n.set_scenarios({"low": 0.5, "high": 0.5})
    add_modular_generator(n, module_cost=500)

    n.optimize.create_model()
    n_mod = n.model["Generator-n_mod"]
    assert "scenario" not in n_mod.dims

    n.optimize()
    n_mod_opt = n.c.generators.static.xs("gas", level="name")["n_mod_opt"]
    assert (n_mod_opt == 1).all()
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


def test_modules_with_investment_periods():
    """Module costs are weighted by the investment period objective weightings."""
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
        module_cost=500,
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


def test_continuous_module_without_nom_max(base_network):
    """Module constraints fall back to big-M when the capacity is unbounded."""
    n = add_modular_generator(base_network, p_nom_max=inf, module_cost=1e9)
    n.optimize()

    assert n.c.generators.static.at["gas", "n_mod_opt"] == 0
    assert n.c.generators.static.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_continuous_module_constraints_without_commitment(base_network):
    """Non-committable continuous modulars need no availability machinery."""
    n = add_modular_generator(base_network)
    n.optimize.create_model()

    assert "Generator-available_p_nom" not in n.model.variables
    assert "Generator-p_nom_available_switch" not in n.model.constraints


def test_mixed_modular_flavours(base_network):
    """Continuous, fixed-size and committable modulars can coexist on one component."""
    n = base_network
    common = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "marginal_cost": 10,
        "capital_cost": 10,
        "modular": True,
        "module_cost": 1e6,
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
    n_mod = n.c.generators.static.loc[["plain", "mod", "com"], "n_mod_opt"]
    assert (n_mod == 0).all()
    assert (n.c.generators.static.loc[["plain", "mod", "com"], "p_nom_opt"] == 0).all()
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_non_extendable_continuous_modular_raises_clear_error(base_network):
    """A continuous modular is meaningless without an extendable capacity."""
    n = base_network
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom=300,
        marginal_cost=10,
        modular=True,
        module_cost=500,
    )

    with pytest.raises(ValueError, match="only supported if `p_nom_extendable=True`"):
        n.optimize()


def test_inactive_modular_extendable(base_network):
    """An inactive modular extendable does not crash and gets no module."""
    n = base_network
    add_modular_generator(n, module_cost=500)
    n.add(
        "Generator",
        "gas_off",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        marginal_cost=10,
        capital_cost=10,
        modular=True,
        module_cost=500,
        active=False,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert "gas_off" not in n.model["Generator-n_mod"].indexes["name"]
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    assert np.isnan(n.c.generators.static.at["gas_off", "n_mod_opt"])


def test_mixed_modular_and_non_modular_with_min(base_network):
    """A forced non-modular extendable coexists with an unbuilt continuous modular."""
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
        modular=True,
        module_cost=1e9,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["forced", "p_nom_opt"] >= 50 - 1e-6
    assert n.c.generators.static.at["opt", "n_mod_opt"] == 0
    assert n.c.generators.static.at["opt", "p_nom_opt"] == pytest.approx(0, abs=1e-6)


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
        modular=True,
        module_cost=1,
    )

    status, _ = n.optimize()

    assert status == "ok"
    assert n.c.generators.static.at["gas", "n_mod_opt"] == 1
    # Peak load is 200 and p_max_pu is 0.1, so 2000 MW of capacity are needed.
    # The old big-M scaled by max_pu would have capped this near the peak load.
    assert n.c.generators.static.at["gas", "p_nom_opt"] == pytest.approx(2000)
    assert n.c.generators.dynamic.p["backup"].sum() == pytest.approx(0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_capex_reconciles_with_objective(base_network):
    """`n.statistics.capex()` includes the module cost and reconciles the objective."""
    n = add_modular_generator(base_network, module_cost=500)
    n.optimize()

    capex = n.statistics.capex().sum()
    opex = n.statistics.opex().sum()
    assert capex == pytest.approx(200 * 10 + 500)
    assert capex + opex == pytest.approx(n.objective)

    capacity = n.statistics.capex(cost_attribute="capital_cost").sum()
    modules = n.statistics.capex(cost_attribute="module_cost").sum()
    assert capacity == pytest.approx(200 * 10)
    assert modules == pytest.approx(500)
    assert capex == pytest.approx(capacity + modules)
    assert n.statistics.installed_capex(cost_attribute="module_cost").empty
    assert n.statistics.expanded_capex(
        cost_attribute="module_cost"
    ).sum() == pytest.approx(500)


def test_capex_counts_fixed_size_modules(base_network):
    """The module cost in `capex` scales with the number of fixed-size modules."""
    n = add_modular_generator(base_network, p_nom_mod=100, module_cost=500)
    n.optimize()

    modules = n.statistics.capex(cost_attribute="module_cost").sum()
    assert modules == pytest.approx(2 * 500)
    capex = n.statistics.capex().sum()
    assert capex + n.statistics.opex().sum() == pytest.approx(n.objective)


def test_overnight_cost_composes_modules(base_network):
    """`overnight_cost` composes the capacity and module overnight terms."""
    n = add_modular_generator(
        base_network, module_cost=500, discount_rate=0.07, lifetime=25
    )
    n.optimize()

    total = n.statistics.overnight_cost().sum()
    capacity = n.statistics.overnight_cost(cost_attribute="overnight_cost").sum()
    modules = n.statistics.overnight_cost(cost_attribute="module_cost").sum()
    assert capacity == pytest.approx(200 * n.c.generators.overnight_cost["gas"])
    assert modules == pytest.approx(n.c.generators.overnight_module_cost["gas"])
    assert total == pytest.approx(capacity + modules)

    with pytest.raises(ValueError, match="cost_attribute must be"):
        n.statistics.overnight_cost(cost_attribute="capital_cost")


# --------------------------------------------------------------------------- #
# Attribute matrix
#
# The module formulation interacts with the module size, unit commitment, minimum
# part-load and the capacity bounds. The tests below sweep every combination of
# those switches and assert that the module decision responds to the module cost
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
    """Add a modular generator with the given combination of attributes."""
    kwargs = {
        "bus": "bus",
        "marginal_cost": 10,
        "capital_cost": 10,
        "modular": True,
        "p_nom_extendable": True,
        "p_nom_max": 500 if p_nom_max else inf,
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


def assert_solution_consistent(n, mod, com, p_min_pu, p_nom_max, p_nom_min):
    """Check invariants that must hold for any modular solution."""
    p_nom = n.c.generators.static.at["gas", "p_nom_opt"]
    n_mod = n.c.generators.static.at["gas", "n_mod_opt"]
    p = n.c.generators.dynamic.p["gas"]

    assert n_mod == pytest.approx(round(n_mod))
    assert (p >= -1e-6).all()
    if mod:
        assert p_nom == pytest.approx(100 * n_mod, abs=1e-6)
    else:
        assert n_mod in (0.0, 1.0)
    if n_mod == 0:
        assert p_nom == pytest.approx(0, abs=1e-6)
    elif p_nom_min:
        assert p_nom >= 100 - 1e-6
    if p_nom_max:
        assert p_nom <= 500 + 1e-6
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
        + float(n.c.generators.periodized_module_cost.sel(name="gas").item()) * n_mod
    )
    assert n.objective == pytest.approx(expected)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_cheap_module_cost_is_paid(base_network, combo):
    """A cheap module cost is paid for every attribute combination."""
    n = add_matrix_generator(base_network, *combo, module_cost=500)
    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    assert n.c.generators.static.at["gas", "n_mod_opt"] >= 1
    assert n.c.generators.static.at["gas", "p_nom_opt"] > 0
    assert_solution_consistent(n, *combo)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_prohibitive_module_cost_blocks_modules(base_network, combo):
    """A prohibitive module cost blocks the modules unless a minimum forces one.

    For fixed-size modules, `p_nom_min` is a plain lower bound, so it forces the
    first module to be built despite its cost.
    """
    mod, _, _, _, p_nom_min = combo
    n = add_matrix_generator(base_network, *combo, module_cost=1e9)
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
def test_matrix_module_cost_overnight(base_network, combo):
    """`module_cost_overnight` is annuitised and wins over `module_cost` throughout."""
    n = add_matrix_generator(
        base_network,
        *combo,
        module_cost=1e9,
        module_cost_overnight=1e6,
        discount_rate=0.07,
        lifetime=25,
    )
    expected = 1e6 * annuity(0.07, 25) * n.c.generators.nyears
    assert n.c.generators.periodized_module_cost.sel(
        name="gas"
    ).item() == pytest.approx(expected)

    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    # The annuitised cost is small, so modules are built despite `module_cost`.
    assert n.c.generators.static.at["gas", "n_mod_opt"] >= 1
    assert_solution_consistent(n, *combo)
