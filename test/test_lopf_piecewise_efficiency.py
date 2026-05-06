# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for piecewise linear part-load efficiency curves in the optimisation."""

from __future__ import annotations

import pytest

import pypsa


@pytest.fixture
def piecewise_efficiency_network() -> pypsa.Network:
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Carrier", "gas", co2_emissions=1.0)
    n.add(
        "Generator",
        "gen",
        carrier="gas",
        bus="bus0",
        p_nom=70,
        marginal_cost=20,
        efficiency={0.1: 0.3, 0.5: 0.4, 1.0: 0.6},
    )
    n.add(
        "GlobalConstraint",
        "co2_limit",
        sense="<=",
        carrier_attribute="co2_emissions",
        constant=160,
    )
    n.add("Load", "load", bus="bus0", p_set=50)
    return n


def test_piecewise_efficiency_uses_equality_formulation(
    piecewise_efficiency_network: pypsa.Network,
) -> None:
    model = piecewise_efficiency_network.optimize.create_model(
        include_objective_constant=False
    )
    formulation = model._piecewise_formulations["Generator_p_primary_piecewise"]
    assert formulation.method == "incremental"


def test_piecewise_efficiency_rejects_lp_equality(
    piecewise_efficiency_network: pypsa.Network,
) -> None:
    with pytest.raises(ValueError, match="method 'lp' requires PyPSA operator"):
        piecewise_efficiency_network.optimize.create_model(
            include_objective_constant=False,
            piecewise_options=[
                {
                    "component": "Generator",
                    "attribute": "efficiency",
                    "operator": "==",
                    "method": "lp",
                }
            ],
        )


def test_piecewise_efficiency_co2_constraint_with_only_segmented_gens(
    piecewise_efficiency_network: pypsa.Network,
) -> None:
    """Regression: when every CO2-emitting generator has piecewise efficiency,
    ``linear_names`` is empty and ``primary_energy`` must still be initialised."""
    n = piecewise_efficiency_network
    n.optimize()
    assert n.generators_t.p["gen"].iloc[0] == pytest.approx(50.0, rel=1e-3)


def test_piecewise_efficiency_gen() -> None:
    """We get a somewhat arbitrary solution as the optimisation problem aims to hit the CO2 limit exactly."""
    n = pypsa.Network()
    n.add("Bus", "bus0")
    n.add("Carrier", "gas", co2_emissions=1.0)
    n.add(
        "Generator",
        "gen0",
        carrier="gas",
        bus="bus0",
        p_nom=70,
        marginal_cost=20,
        efficiency=0.5,
    )
    n.add(
        "Generator",
        "gen1",
        carrier="gas",
        bus="bus0",
        p_nom=70,
        marginal_cost=20,
        efficiency={0.1: 0.3, 0.5: 0.4, 1.0: 0.6},
    )
    n.add(
        "GlobalConstraint",
        "co2_limit",
        sense="<=",
        carrier_attribute="co2_emissions",
        constant=160,
    )
    n.add("Load", "load", bus="bus0", p_set=80)
    n.optimize()
    assert n.generators_t.p["gen1"].item() == pytest.approx(50.0, rel=1e-3)
    assert n.generators_t.p["gen0"].item() == pytest.approx(30.0, rel=1e-3)
