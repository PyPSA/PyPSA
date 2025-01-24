import pytest

import pypsa


@pytest.fixture
def n():
    n = pypsa.Network()
    n.set_snapshots(range(10))

    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=1.0)

    n.add("Bus", "transport")
    n.add("Load", "transport", bus="transport", p_set=1.0)

    n.add("Bus", "diesel")

    n.add("Store", "diesel", bus="diesel", e_cyclic=True, e_nom=1000.0)

    n.add("Bus", "hydrogen")

    n.add("Store", "hydrogen", bus="hydrogen", e_cyclic=True, e_nom=1000.0)

    n.add(
        "Link", "electrolysis", p_nom=2.0, efficiency=0.8, bus0="bus", bus1="hydrogen"
    )

    n.add(
        "Link",
        "FT",
        p_nom=4,
        bus0="hydrogen",
        bus1="diesel",
        bus2="co2 stored",
        efficiency=1.0,
        efficiency2=-1,
    )

    # minus sign because opposite to how fossil fuels used:
    # CH4 burning puts CH4 down, atmosphere up
    n.add("Carrier", "co2", co2_emissions=-1.0)

    # this tracks CO2 in the atmosphere
    n.add("Bus", "co2 atmosphere", carrier="co2")

    # NB: can also be negative
    n.add("Store", "co2 atmosphere", e_nom=1000, e_min_pu=-1, bus="co2 atmosphere")

    # this tracks CO2 stored, e.g. underground
    n.add("Bus", "co2 stored")

    # NB: can also be negative
    n.add("Store", "co2 stored", e_nom=1000, e_min_pu=-1, bus="co2 stored")

    n.add(
        "Link",
        "DAC",
        bus0="bus",
        bus1="co2 stored",
        bus2="co2 atmosphere",
        efficiency=1,
        efficiency2=-1,
        p_nom=5.0,
    )

    n.add(
        "Link",
        "diesel car",
        bus0="diesel",
        bus1="transport",
        bus2="co2 atmosphere",
        efficiency=1.0,
        efficiency2=1.0,
        p_nom=2.0,
    )

    n.add("Bus", "gas")

    n.add("Store", "gas", e_initial=50, e_nom=50, marginal_cost=20, bus="gas")

    n.add(
        "Link",
        "OCGT",
        bus0="gas",
        bus1="bus",
        bus2="co2 atmosphere",
        p_nom_extendable=True,
        efficiency=0.5,
        efficiency2=1,
    )

    n.add(
        "Link",
        "OCGT+CCS",
        bus0="gas",
        bus1="bus",
        bus2="co2 stored",
        bus3="co2 atmosphere",
        p_nom_extendable=True,
        efficiency=0.4,
        efficiency2=0.9,
        efficiency3=0.1,
    )

    # Add a cheap and a expensive biomass generator.
    biomass_marginal_cost = [20.0, 50.0]
    biomass_stored = [40.0, 15.0]

    for i in range(2):
        n.add("Bus", f"biomass{str(i)}")

        n.add(
            "Store",
            f"biomass{str(i)}",
            bus=f"biomass{str(i)}",
            e_nom_extendable=True,
            marginal_cost=biomass_marginal_cost[i],
            e_nom=biomass_stored[i],
            e_initial=biomass_stored[i],
        )

        # simultaneously empties and refills co2 atmosphere
        n.add(
            "Link",
            f"biomass{str(i)}",
            bus0=f"biomass{str(i)}",
            bus1="bus",
            p_nom_extendable=True,
            efficiency=0.5,
        )

        n.add(
            "Link",
            f"biomass+CCS{str(i)}",
            bus0=f"biomass{str(i)}",
            bus1="bus",
            bus2="co2 stored",
            bus3="co2 atmosphere",
            p_nom_extendable=True,
            efficiency=0.4,
            efficiency2=1.0,
            efficiency3=-1,
        )

    # can go to -50, but at some point can't generate enough electricity for DAC and demand
    target = -50
    n.add(
        "GlobalConstraint",
        "co2_limit",
        sense="<=",
        carrier_attribute="co2_emissions",
        constant=target,
    )
    return n


def test_attribution_assignment(n):
    assert "bus2" in n.components["Link"]["attrs"].index
    assert n.components["Link"]["attrs"].loc["bus2", "default"] == ""


def test_optimize(n):
    status, condition = n.optimize()
    assert status == "ok"
