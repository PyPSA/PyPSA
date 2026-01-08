# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from pypsa import Network


def test_add_missing_buses():
    """Test add_missing_buses: basic, with existing, no missing, custom kwargs."""
    n = Network()
    # Test 1: Basic - add components referencing missing buses
    n.add("Generator", ["gen1", "gen2"], bus=["bus1", "bus2"], p_nom=100)
    n.add("Load", "load1", bus="bus3", p_set=50)
    assert len(n.c.buses.static) == 0

    added = n.c.buses.add_missing_buses(v_nom=380)
    assert set(added) == {"bus1", "bus2", "bus3"}
    assert len(n.c.buses.static) == 3
    assert all(n.c.buses.static["v_nom"] == 380)

    # Test 2: No missing buses
    added2 = n.c.buses.add_missing_buses()
    assert len(added2) == 0

    # Test 3: With existing bus and custom kwargs
    n2 = Network()
    n2.add("Bus", "bus1", v_nom=110, carrier="DC")
    n2.add("Line", "line1", bus0="bus1", bus1="bus2", x=0.1)
    added3 = n2.c.buses.add_missing_buses(v_nom=220, carrier="AC")
    assert set(added3) == {"bus2"}
    assert n2.c.buses.static.loc["bus1", "v_nom"] == 110  # Not changed
    assert n2.c.buses.static.loc["bus2", "v_nom"] == 220

    # Test 4: Link with empty bus2/bus3 doesn't create phantom buses
    n3 = Network()
    n3.add("Bus", "bus0")
    n3.add("Link", "link1", bus0="bus0", bus1="bus1")
    added4 = n3.c.buses.add_missing_buses()
    assert set(added4) == {"bus1"}
    assert "" not in n3.c.buses.names


def test_add_missing_buses_stochastic():
    """Test add_missing_buses with stochastic networks."""
    n = Network()
    n.add("Bus", "bus1")
    n.add("Generator", "gen1", bus="bus1", p_nom=100)
    n.set_scenarios(["s1", "s2"])

    # Add components referencing missing buses after scenarios
    n.add("Generator", "gen2", bus="bus2", p_nom=150)
    n.add("Line", "line1", bus0="bus2", bus1="bus3", x=0.1)

    added = n.c.buses.add_missing_buses(v_nom=380)
    assert set(added) == {"bus2", "bus3"}

    # Verify scenario structure
    assert n.c.buses.static.index.nlevels == 2
    assert list(n.c.buses.static.index.names) == ["scenario", "name"]
    for scenario in ["s1", "s2"]:
        for bus in ["bus1", "bus2", "bus3"]:
            assert (scenario, bus) in n.c.buses.static.index
        # bus1 has default v_nom, bus2/bus3 have custom v_nom
        assert n.c.buses.static.loc[(scenario, "bus2"), "v_nom"] == 380
        assert n.c.buses.static.loc[(scenario, "bus3"), "v_nom"] == 380
