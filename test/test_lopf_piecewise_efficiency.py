# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for piecewise linear part-load efficiency curves in the optimisation."""

import pytest

import pypsa


def test_piecewise_efficiency_gen():
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
