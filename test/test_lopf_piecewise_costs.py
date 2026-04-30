# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for piecewise linear cost curves in the optimisation objective."""

import pandas as pd
import pytest

import pypsa


def test_piecewise_marginal_cost():
    """gen1 runs at 25 MW (where its marginal cost then becomes more expensive than gen0); gen0 covers 55 MW."""

    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Generator", "gen0", bus="bus0", p_nom=100, marginal_cost=20)
    n.add(
        "Generator",
        "gen1",
        bus="bus0",
        p_nom=100,
        marginal_cost={0.1: 60, 0.5: 35.0, 1.0: 100.0},
    )
    n.add("Load", "load", bus="bus0", p_set=80)
    n.optimize()
    assert n.generators_t.p["gen1"].iloc[0] == pytest.approx(10.0, rel=1e-3)
    assert n.generators_t.p["gen0"].iloc[0] == pytest.approx(70.0, rel=1e-3)
    assert n.objective < 4000.0


# TODO-1603: unmark once we have dealt with initial cost
@pytest.mark.xfail(reason="Need to fix how we handle initial cost (at zero p_nom)")
def test_piecewise_capital_cost():
    """Network with piecewise capital cost (diseconomies of scale = convex total cost).

    gen2: extendable, p_nom_extendable=True, p_nom_max=100 MW
          piecewise capital_cost — marginal capital cost (€/MW) at breakpoints:
          p_nom=0 -> 0.5 €/MW, p_nom=100 -> 1.5 €/MW  (linearly increasing)
    load: 80 MW

    Optimal: gen2 builds exactly 80 MW.
    Total cost at 80 MW = ∫₀^80 (0.5 + p/100) dp = [0.5p + p²/200]₀^80 = 40 + 32 = 72 €
    """
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add(
        "Generator",
        "gen2",
        bus="bus0",
        p_nom_extendable=True,
        p_nom_max=100,
        capital_cost=pd.DataFrame({"p_nom": [0.0, 100.0], "capital_cost": [0.5, 1.5]}),
    )
    n.add("Load", "load", bus="bus0", p_set=80)
    n.optimize()

    assert n.objective == pytest.approx(70.0, rel=1e-3)
    assert n.generators.p_nom_opt["gen2"] == pytest.approx(80.0, rel=1e-3)
