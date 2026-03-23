# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

import pypsa
from pypsa.optimization.common import get_bus_counts, get_strongly_meshed_buses


def test_meshed_threshold():
    """
    This test is based on https://docs.pypsa.org/en/latest/examples/unit-
    commitment.html and is not very comprehensive.
    """
    # marginal costs in EUR/MWh
    marginal_costs = {"Wind": 0, "Hydro": 0, "Coal": 30, "Gas": 60, "Oil": 80}

    # power plant capacities (nominal powers in MW) in each country (not necessarily realistic)
    power_plant_p_nom = {
        "Country1": {
            "Coal": 35000,
            "Wind": 3000,
            "Gas": 8000,
            "Oil": 2000,
        },
        "Country2": {
            "Gas": 600,
        },
        "Country3": {
            "Gas": 600,
        },
        "Country4": {
            "Gas": 600,
        },
        "Country5": {
            "Gas": 600,
        },
        "Country6": {
            "Gas": 600,
        },
        "Country7": {
            "Gas": 600,
        },
        "Country8": {
            "Gas": 600,
        },
        "Country9": {
            "Gas": 600,
        },
    }

    # transmission capacities in MW (not necessarily realistic)
    transmission = {
        "Country1": {"Country2": 100, "Country4": 250, "Country7": 250},
        "Country2": {"Country3": 100, "Country5": 250, "Country8": 250},
        "Country3": {"Country4": 100, "Country6": 250, "Country9": 250},
        "Country4": {"Country5": 100, "Country7": 250, "Country1": 250},
        "Country5": {"Country6": 100, "Country8": 250, "Country2": 250},
        "Country6": {"Country7": 100, "Country9": 250, "Country3": 250},
        "Country7": {"Country8": 100, "Country1": 250, "Country4": 250},
        "Country8": {"Country9": 100, "Country2": 250, "Country5": 250},
        "Country9": {"Country1": 100, "Country3": 250, "Country6": 250},
    }

    # country electrical loads in MW (not necessarily realistic)
    loads = {
        "Country1": 42000,
        "Country2": 650,
        "Country3": 250,
        "Country4": 250,
        "Country5": 250,
        "Country6": 250,
        "Country7": 250,
        "Country8": 250,
        "Country9": 250,
    }

    network = pypsa.Network()

    network.add("carriers", "AC")

    countries = []
    for i in range(1, 10):
        countries.extend(["Country" + str(i)])

    for country in countries:
        network.add("Bus", country)

    for country in countries:
        for tech in power_plant_p_nom[country]:
            network.add(
                "Generator",
                f"{country} {tech}",
                bus=country,
                p_nom=power_plant_p_nom[country][tech],
                marginal_cost=marginal_costs[tech],
            )
        network.add("Load", f"{country} load", bus=country, p_set=loads[country])
        #
        # add transmission as controllable Link
        if country not in transmission:
            continue
        for other_country in countries:
            if other_country not in transmission[country]:
                continue
            #
            # NB: Link is by default unidirectional, so have to set p_min_pu = -1
            # to allow bidirectional (i.e. also negative) flow
            network.add(
                "Link",
                f"{country} - {other_country} link",
                bus0=country,
                bus1=other_country,
                p_nom=transmission[country][other_country],
                p_min_pu=-1,
            )

    expected_retCode = pd.Index([], name="Bus-meshed")
    with pytest.deprecated_call():
        retCode = get_strongly_meshed_buses(network)

    assert len(retCode) == len(expected_retCode)

    expected_retCode = pd.Index(["Country1"], name="Bus-meshed")
    with pytest.deprecated_call():
        retCode = get_strongly_meshed_buses(network, threshold=10)

    assert len(retCode) == len(expected_retCode)


def test_get_bus_counts_basic():
    n = pypsa.Network()
    n.add("Bus", "b0")
    n.add("Bus", "b1")
    n.add("Generator", "g", bus="b0", p_nom=1)
    n.add("Load", "l", bus="b0", p_set=1)
    n.add("Link", "k", bus0="b0", bus1="b1", p_nom=1)

    counts = get_bus_counts(n)

    assert counts["b0"] == 3
    assert counts["b1"] == 1


def test_get_strongly_meshed_buses_deprecated():
    n = pypsa.Network()
    n.add("Bus", "b0")

    with pytest.deprecated_call():
        get_strongly_meshed_buses(n)


def test_meshed_thresholds_skip_isolated_buses():
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Bus", "b0")
    n.add("Bus", "b1")
    n.add("Generator", "g", bus="b0", p_nom=1, marginal_cost=1)
    n.add("Load", "l", bus="b0", p_set=1)

    n.optimize.create_model()

    names_dim = n.model.constraints["Bus-nodal_balance"].coords["name"].to_index()
    assert "b0" in names_dim
    assert "b1" not in names_dim


def test_meshed_threshold_backward_compatibility_warning():
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Bus", "b0")
    n.add("Generator", "g", bus="b0", p_nom=1, marginal_cost=1)
    n.add("Load", "l", bus="b0", p_set=1)

    with pytest.warns(FutureWarning, match="meshed_threshold"):
        n.optimize.create_model(meshed_threshold=10)
