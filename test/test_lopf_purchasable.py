# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for unit purchase decisions (`purchasable`) and unit costs.

Purchasable components introduce a binary variable `{component}-purchased` which
decides whether an asset is bought at all, independently of how much capacity is
built. The associated fixed cost is given by `unit_cost` (or `unit_cost_overnight`
together with `discount_rate` and `lifetime`).
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
    """Add a cheap, modular, purchasable generator to the network."""
    defaults = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "p_nom_mod": 100,
        "marginal_cost": 10,
        "capital_cost": 10,
        "purchasable": True,
        "unit_cost": 500,
    }
    n.add("Generator", "gas", **(defaults | kwargs))
    return n


def test_purchase_variables_and_constraints(base_network):
    """A purchasable component adds a binary purchase variable to the model."""
    n = add_modular_generator(base_network)
    n.optimize.create_model()

    assert "Generator-purchased" in n.model.variables
    purchased = n.model["Generator-purchased"]
    assert purchased.attrs["binary"]
    # The purchase decision is per asset, not per snapshot.
    assert list(purchased.dims) == ["name"]
    assert list(purchased.indexes["name"]) == ["gas"]

    # Modular purchasables tie the number of modules to the purchase decision.
    assert "Generator-p_nom_modularity_purchased_bigM" in n.model.constraints


def test_no_purchase_variables_without_purchasable(base_network):
    """No purchase machinery is created if nothing is flagged purchasable."""
    n = add_modular_generator(base_network, purchasable=False)
    n.optimize.create_model()

    assert "Generator-purchased" not in n.model.variables
    assert "Generator-p_nom_modularity_purchased_bigM" not in n.model.constraints


def test_purchase_worthwhile(base_network):
    """A cheap unit cost is paid and the asset is built."""
    n = add_modular_generator(base_network, unit_cost=500)
    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    assert n.generators.at["gas", "purchased_opt"] == 1
    assert n.generators.at["gas", "p_nom_opt"] == 200

    # marginal + capital + unit cost
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


def test_purchase_not_worthwhile(base_network):
    """A prohibitive unit cost blocks both the purchase and the capacity."""
    n = add_modular_generator(base_network, unit_cost=1e6)
    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "purchased_opt"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_purchase_just_below_and_above_break_even(base_network):
    """Purchase flips from 1 to 0 as the unit cost crosses break-even."""
    savings = 450 * (100 - 10) - 200 * 10

    for unit_cost, expected in [(savings - 100, 1), (savings + 100, 0)]:
        n = add_modular_generator(base_network.copy(), unit_cost=unit_cost)
        n.optimize()
        assert n.generators.at["gas", "purchased_opt"] == expected


def test_purchased_opt_only_for_purchasables(base_network):
    """Non-purchasable components get no purchase result."""
    n = add_modular_generator(base_network)
    n.optimize()

    assert n.generators.at["gas", "purchased_opt"] == 1
    assert np.isnan(n.generators.at["backup", "purchased_opt"])


def test_unit_cost_overnight_is_annuitised(base_network):
    """`unit_cost_overnight` is periodized like `overnight_cost`."""
    overnight, rate, lifetime = 1e6, 0.07, 25
    n = add_modular_generator(
        base_network,
        unit_cost=0,
        unit_cost_overnight=overnight,
        discount_rate=rate,
        lifetime=lifetime,
    )
    nyears = n.c.generators.nyears
    expected = overnight * annuity(rate, lifetime) * nyears

    assert n.c.generators.unit_cost.sel(name="gas").item() == pytest.approx(expected)

    n.optimize()
    assert n.generators.at["gas", "purchased_opt"] == 1
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + expected)


def test_unit_cost_overnight_takes_precedence(base_network):
    """`unit_cost_overnight` overrides a directly given `unit_cost`."""
    n = add_modular_generator(
        base_network,
        unit_cost=12345,
        unit_cost_overnight=1e6,
        discount_rate=0.0,
        lifetime=10,
    )
    nyears = n.c.generators.nyears
    assert n.c.generators.unit_cost.sel(name="gas").item() == pytest.approx(
        1e6 / 10 * nyears
    )


def test_zero_unit_cost_not_in_objective(base_network):
    """Without a unit cost, the purchase decision is free but still available."""
    n = add_modular_generator(base_network, unit_cost=0)
    n.optimize()

    assert "Generator-purchased" in n.model.variables
    assert n.objective == pytest.approx(450 * 10 + 200 * 10)


@pytest.mark.parametrize(
    "component",
    [
        "Generator",
        "Link",
        "Line",
        "Process",
        "Store",
        pytest.param(
            "StorageUnit",
            marks=pytest.mark.xfail(
                reason="define_purchase_constraints derives big-M values via "
                "get_bounds_pu with the component's base attribute, which "
                "StorageUnit does not accept",
                raises=ValueError,
                strict=True,
            ),
        ),
    ],
)
def test_prohibitive_unit_cost_blocks_all_components(base_network, component):
    """A prohibitive unit cost prevents the purchase for every component type."""
    n = base_network
    n.add("Bus", "bus1")
    n.add("Generator", "cheap", bus="bus1", p_nom=1000, marginal_cost=1)

    common = {"purchasable": True, "unit_cost": 1e9}
    if component == "Generator":
        n.add(
            "Generator",
            "asset",
            bus="bus",
            p_nom_extendable=True,
            p_nom_max=500,
            p_nom_mod=100,
            marginal_cost=1,
            **common,
        )
        nom_attr = "p_nom"
    elif component == "Link":
        n.add(
            "Link",
            "asset",
            bus0="bus1",
            bus1="bus",
            p_nom_extendable=True,
            p_nom_max=500,
            p_nom_mod=100,
            **common,
        )
        nom_attr = "p_nom"
    elif component == "Process":
        n.add(
            "Process",
            "asset",
            bus0="bus1",
            bus1="bus",
            p_nom_extendable=True,
            p_nom_max=500,
            p_nom_mod=100,
            **common,
        )
        nom_attr = "p_nom"
    elif component == "Line":
        n.add(
            "Line",
            "asset",
            bus0="bus1",
            bus1="bus",
            x=0.1,
            s_nom_extendable=True,
            s_nom_max=500,
            s_nom_mod=100,
            **common,
        )
        nom_attr = "s_nom"
    elif component == "Store":
        n.add(
            "Store",
            "asset",
            bus="bus",
            e_nom_extendable=True,
            e_nom_max=500,
            e_nom_mod=100,
            **common,
        )
        nom_attr = "e_nom"
    else:
        n.add(
            "StorageUnit",
            "asset",
            bus="bus",
            p_nom_extendable=True,
            p_nom_max=500,
            p_nom_mod=100,
            **common,
        )
        nom_attr = "p_nom"

    status, _ = n.optimize()

    assert status == "ok"
    static = n.c[component].static
    assert static.at["asset", "purchased_opt"] == 0
    assert static.at["asset", f"{nom_attr}_opt"] == 0


def test_purchasable_link(base_network):
    """A purchasable link is bought when the transfer is worth it."""
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
        purchasable=True,
        unit_cost=500,
    )
    n.optimize()

    assert n.links.at["link", "purchased_opt"] == 1
    assert n.links.at["link", "p_nom_opt"] == 200
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


def test_purchasable_committable_generator(base_network):
    """Purchase decisions can be combined with unit commitment."""
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
        purchasable=True,
        unit_cost=1e6,
        # No module is committed before the horizon starts, so nothing forces
        # the generator to be built (see the modular committable formulation).
        status=0,
        up_time_before=0,
    )
    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "purchased_opt"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0


def add_committable_generator(n, **kwargs):
    """Add a cheap, non-modular, committable and purchasable generator."""
    defaults = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "committable": True,
        "p_min_pu": 0.5,
        "marginal_cost": 1,
        "capital_cost": 1,
        "purchasable": True,
        "unit_cost": 1,
        # Nothing is committed before the horizon starts.
        "status": 0,
        "up_time_before": 0,
    }
    n.add("Generator", "gas", **(defaults | kwargs))
    return n


def test_committable_purchase_constraints(base_network):
    """Committable purchasables track the capacity available per snapshot."""
    n = add_committable_generator(base_network)
    n.optimize.create_model()

    available = n.model["Generator-available_p_nom"]
    assert set(available.dims) == {"name", "snapshot"}

    for name in [
        "Generator-p_nom_cap_binary",
        "Generator-p_nom_status_purchased_limit",
        "Generator-p_nom_available_continuous",
        "Generator-p_nom_available_binary",
        "Generator-p_nom_available_switch",
        "Generator-com-purchase-p-lower",
        "Generator-com-purchase-p-upper",
    ]:
        assert name in n.model.constraints


def test_committable_purchase_worthwhile(base_network):
    """A committable purchasable is bought and only committed when needed.

    The load is zero in the last snapshot, so the generator must be able to shut
    down without forfeiting its capacity.
    """
    n = base_network
    n.loads_t.p_set["load"] = [100, 200, 0]
    add_committable_generator(n)

    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "purchased_opt"] == 1
    assert n.generators.at["gas", "p_nom_opt"] == pytest.approx(200)
    assert n.generators_t.status["gas"].tolist() == [1, 1, 0]
    # marginal (300) + capital (200) + unit cost (1)
    assert n.objective == pytest.approx(501)


def test_committable_purchase_not_worthwhile(base_network):
    """A prohibitive unit cost blocks purchase, capacity and commitment."""
    n = base_network
    n.loads_t.p_set["load"] = [100, 200, 0]
    add_committable_generator(n, unit_cost=1e6)

    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "purchased_opt"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0
    assert (n.generators_t.status["gas"] == 0).all()
    assert n.objective == pytest.approx(300 * 100)


def test_status_requires_purchase(base_network):
    """An unpurchased committable component can never be committed."""
    n = base_network
    n.loads_t.p_set["load"] = [100, 200, 0]
    add_committable_generator(n, unit_cost=1e6)
    n.optimize()

    purchased = n.generators.at["gas", "purchased_opt"]
    assert (n.generators_t.status["gas"] <= purchased).all()


def test_linearized_unit_commitment_rejects_purchasables(base_network):
    """Relaxing the integrality of purchasable committables is not allowed."""
    n = base_network
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom_extendable=True,
        p_nom_max=500,
        committable=True,
        marginal_cost=10,
        purchasable=True,
        unit_cost=500,
    )

    with pytest.raises(
        ValueError, match="linearized_unit_commitment.*modular/purchasable"
    ):
        n.optimize(linearized_unit_commitment=True)


@pytest.mark.xfail(
    reason="the purchase-dependent lower capacity bound in "
    "`define_nominal_constraints_for_extendables` does not broadcast the "
    "scenario-less purchase variable against the scenario-indexed bound",
    strict=True,
)
def test_purchase_with_scenarios(base_network):
    """The purchase decision is shared across scenarios."""
    n = base_network
    n.set_scenarios({"low": 0.5, "high": 0.5})
    add_modular_generator(n, unit_cost=500)

    n.optimize.create_model()
    purchased = n.model["Generator-purchased"]
    assert "scenario" not in purchased.dims

    n.optimize()
    purchased_opt = n.generators.xs("gas", level="name")["purchased_opt"]
    assert (purchased_opt == 1).all()


def test_purchase_with_investment_periods():
    """Unit costs are weighted by the investment period weightings."""
    n = pypsa.Network()
    n.set_snapshots(pd.MultiIndex.from_product([[2020, 2030], range(2)]))
    n.investment_periods = [2020, 2030]
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
        purchasable=True,
        unit_cost=500,
        build_year=2020,
        lifetime=100,
    )
    status, _ = n.optimize()

    assert status == "ok"
    assert n.generators.at["gas", "purchased_opt"] == 1
    assert n.generators.at["gas", "p_nom_opt"] == 100


def test_continuous_purchase_blocks_capacity(base_network):
    """A prohibitive unit cost blocks a non-modular purchase and its capacity."""
    n = add_modular_generator(base_network, p_nom_mod=0, unit_cost=1e6)
    n.optimize()

    assert n.generators.at["gas", "purchased_opt"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_continuous_purchase_worthwhile(base_network):
    """A cheap non-modular purchasable is bought and sized freely."""
    n = add_modular_generator(base_network, p_nom_mod=0, unit_cost=500)
    n.optimize()

    assert n.generators.at["gas", "purchased_opt"] == 1
    assert n.generators.at["gas", "p_nom_opt"] == pytest.approx(200)
    assert n.objective == pytest.approx(450 * 10 + 200 * 10 + 500)


def test_continuous_purchase_without_nom_max(base_network):
    """Purchase constraints fall back to big-M when the capacity is unbounded."""
    n = add_modular_generator(base_network, p_nom_mod=0, p_nom_max=inf, unit_cost=1e9)
    n.optimize()

    assert n.generators.at["gas", "purchased_opt"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


def test_continuous_purchase_constraints(base_network):
    """Non-modular purchasables link capacity to the purchase decision."""
    n = add_modular_generator(base_network, p_nom_mod=0)
    n.optimize.create_model()

    assert "Generator-p_nom_cap_binary" in n.model.constraints
    # No commitment, so no availability machinery is needed.
    assert "Generator-available_p_nom" not in n.model.variables
    assert "Generator-p_nom_available_switch" not in n.model.constraints


def test_mixed_purchasable_flavours(base_network):
    """Modular, committable and plain purchasables can coexist on one component."""
    n = base_network
    common = {
        "bus": "bus",
        "p_nom_extendable": True,
        "p_nom_max": 500,
        "marginal_cost": 10,
        "capital_cost": 10,
        "purchasable": True,
        "unit_cost": 1e6,
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
    purchased = n.generators.loc[["plain", "mod", "com"], "purchased_opt"]
    assert (purchased == 0).all()
    assert (n.generators.loc[["plain", "mod", "com"], "p_nom_opt"] == 0).all()
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)


@pytest.mark.xfail(
    reason="`purchasable` is only supported for extendable components, but a "
    "non-extendable purchasable raises an opaque `KeyError: 'Generator-p_nom'` from "
    "`define_purchase_constraints` instead of a consistency error",
    strict=True,
)
def test_non_extendable_purchasable_raises_clear_error(base_network):
    """Purchasable is meaningless without an extendable capacity."""
    n = base_network
    n.add(
        "Generator",
        "gas",
        bus="bus",
        p_nom=300,
        marginal_cost=10,
        purchasable=True,
        unit_cost=500,
    )

    with pytest.raises(ValueError, match="purchasable"):
        n.optimize()


# --------------------------------------------------------------------------- #
# Attribute matrix
#
# The purchase formulation interacts with modularity, unit commitment, minimum
# part-load and the capacity bounds. The tests below sweep every combination of
# those switches and assert that the purchase decision responds to the unit cost
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
    """Add a purchasable generator with the given combination of attributes."""
    kwargs = {
        "bus": "bus",
        "marginal_cost": 10,
        "capital_cost": 10,
        "purchasable": True,
        "p_nom_extendable": True,
        "p_nom_max": 500 if p_nom_max else inf,
        "p_nom_min": 100 if p_nom_min else 0,
    }
    if mod:
        kwargs["p_nom_mod"] = 100
    if com:
        # Nothing committed before the horizon, so the purchase is unforced.
        kwargs |= {"committable": True, "status": 0, "up_time_before": 0}
    if p_min_pu:
        kwargs["p_min_pu"] = 0.5
    n.add("Generator", "gas", **(kwargs | cost))
    return n


def assert_solution_consistent(n, mod, com, p_min_pu, p_nom_max, p_nom_min):
    """Check invariants that must hold for any purchasable solution."""
    p_nom = n.generators.at["gas", "p_nom_opt"]
    purchased = n.generators.at["gas", "purchased_opt"]
    p = n.generators_t.p["gas"]

    assert purchased in (0.0, 1.0)
    assert (p >= -1e-6).all()
    if purchased == 0:
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
            if purchased == 0:
                assert (status == 0).all()
        else:
            assert (status <= purchased + 1e-6).all()
        if p_min_pu:
            committed = status > 0.5
            assert (p[committed] >= 0.5 * p_nom - 1e-4).all()

    # The objective must be exactly reproducible from the solution.
    expected = (
        10 * p.sum()
        + 100 * n.generators_t.p["backup"].sum()
        + 10 * p_nom
        + float(n.c.generators.unit_cost.sel(name="gas").item()) * purchased
    )
    assert n.objective == pytest.approx(expected)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_cheap_unit_cost_is_purchased(base_network, combo):
    """A cheap unit cost is paid for every attribute combination."""
    n = add_matrix_generator(base_network, *combo, unit_cost=500)
    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    assert n.generators.at["gas", "purchased_opt"] == 1
    assert n.generators.at["gas", "p_nom_opt"] > 0
    assert_solution_consistent(n, *combo)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_prohibitive_unit_cost_blocks_purchase(base_network, combo):
    """A prohibitive unit cost blocks purchase and capacity in every combination."""
    n = add_matrix_generator(base_network, *combo, unit_cost=1e9)
    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    assert n.generators.at["gas", "purchased_opt"] == 0
    assert n.generators.at["gas", "p_nom_opt"] == 0
    assert n.objective == pytest.approx(BACKUP_ONLY_COST)
    assert_solution_consistent(n, *combo)


@pytest.mark.parametrize("combo", MATRIX, ids=MATRIX_IDS)
def test_matrix_unit_cost_overnight(base_network, combo):
    """`unit_cost_overnight` is annuitised and wins over `unit_cost` throughout."""
    n = add_matrix_generator(
        base_network,
        *combo,
        unit_cost=1e9,
        unit_cost_overnight=1e6,
        discount_rate=0.07,
        lifetime=25,
    )
    expected = 1e6 * annuity(0.07, 25) * n.c.generators.nyears
    assert n.c.generators.unit_cost.sel(name="gas").item() == pytest.approx(expected)

    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    # The annuitised cost is small, so the asset is bought despite `unit_cost`.
    assert n.generators.at["gas", "purchased_opt"] == 1
    assert_solution_consistent(n, *combo)
