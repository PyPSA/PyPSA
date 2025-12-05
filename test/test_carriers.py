# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT


from pypsa import Network


def test_add_missing_carriers():
    """Test add_missing_carriers: basic, existing, no missing, empty/NaN ignored, kwargs."""
    # Test 1: Basic - all carriers missing, multiple component types, includes AC
    n = Network()
    n.add("Bus", ["bus1", "bus2"])
    n.add("Generator", ["g1", "g2"], bus=["bus1", "bus2"], carrier=["wind", "solar"])
    n.add("Load", "l1", bus="bus1", carrier="electricity")
    n.add("Link", "link1", bus0="bus1", bus1="bus2", carrier="H2")
    added = n.c.carriers.add_missing_carriers()
    assert set(added) == {"AC", "H2", "electricity", "solar", "wind"}
    assert len(n.c.carriers.static) == 5

    # Test 2: With existing - doesn't overwrite, custom kwargs
    n2 = Network()
    n2.add("Bus", "bus1")
    n2.add("Carrier", "wind", co2_emissions=0.0)
    n2.add("Generator", ["g1", "g2"], bus="bus1", carrier=["wind", "solar"])
    added2 = n2.c.carriers.add_missing_carriers(co2_emissions=0.5)
    assert set(added2) == {"AC", "solar"}
    assert n2.c.carriers.static.loc["wind", "co2_emissions"] == 0.0  # Not changed
    assert n2.c.carriers.static.loc["solar", "co2_emissions"] == 0.5

    # Test 3: No missing, empty/NaN ignored
    added3 = n2.c.carriers.add_missing_carriers()
    assert len(added3) == 0
    n3 = Network()
    n3.add("Bus", "bus1")
    n3.add("Generator", ["g1", "g2", "g3"], bus="bus1", carrier=["wind", "", None])
    added4 = n3.c.carriers.add_missing_carriers()
    assert set(added4) == {"AC", "wind"}


def test_add_missing_carriers_stochastic():
    """Test add_missing_carriers with stochastic networks."""
    n = Network()
    n.add("Bus", ["bus1", "bus2"])
    n.add("Carrier", "wind")  # Add before scenarios
    n.add("Generator", "g1", bus="bus1", carrier="wind")
    n.set_scenarios({"low": 0.5, "high": 0.5})
    n.add("Generator", "g2", bus="bus2", carrier="solar")
    added = n.c.carriers.add_missing_carriers()
    assert set(added) == {"AC", "solar"}

    # Verify scenario structure
    assert n.c.carriers.static.index.nlevels == 2
    for scenario in ["low", "high"]:
        for carrier in ["AC", "solar", "wind"]:
            assert (scenario, carrier) in n.c.carriers.static.index


def test_assign_colors():
    """Test assign_colors: basic, specific carriers, palette, overwrite, partial."""
    # Test 1: Basic - assigns colors to all carriers
    n = Network()
    n.add("Bus", "bus1")
    n.add("Carrier", ["wind", "solar", "gas"])
    assert all(n.c.carriers.static["color"] == "")
    n.c.carriers.assign_colors()
    assert all(n.c.carriers.static["color"] != "")
    assert len(set(n.c.carriers.static["color"])) == 3  # Different colors

    # Test 2: Specific carriers, single string
    n2 = Network()
    n2.add("Bus", "bus1")
    n2.add("Carrier", ["wind", "solar", "gas"])
    n2.c.carriers.assign_colors(["wind", "solar"])
    assert n2.c.carriers.static.loc["wind", "color"] != ""
    assert n2.c.carriers.static.loc["solar", "color"] != ""
    assert n2.c.carriers.static.loc["gas", "color"] == ""
    n2.c.carriers.assign_colors("gas")  # Single string
    assert n2.c.carriers.static.loc["gas", "color"] != ""

    # Test 3: Custom palette, many carriers (cycling)
    n3 = Network()
    n3.add("Bus", "b1")
    n3.add("Carrier", [f"c{i}" for i in range(15)])
    n3.c.carriers.assign_colors(palette="tab20")
    assert all(n3.c.carriers.static["color"] != "")

    # Test 4: Overwrite, partial colors
    n4 = Network()
    n4.add("Bus", "bus1")
    n4.add("Carrier", ["wind", "solar"], color=["blue", ""])
    n4.c.carriers.assign_colors(overwrite=False)
    assert n4.c.carriers.static.loc["wind", "color"] == "blue"  # Not changed
    assert n4.c.carriers.static.loc["solar", "color"] != ""  # Assigned
    n4.c.carriers.assign_colors(overwrite=True)
    assert n4.c.carriers.static.loc["wind", "color"] != "blue"  # Changed


def test_assign_colors_stochastic():
    """Test assign_colors with stochastic networks - colors same across scenarios."""
    n = Network()
    n.add("Bus", "bus1")
    n.add("Carrier", ["wind", "solar"])
    n.set_scenarios(["low", "high"])
    n.c.carriers.assign_colors()
    # Colors assigned for both scenarios
    assert n.c.carriers.static.loc[("low", "wind"), "color"] != ""
    assert n.c.carriers.static.loc[("high", "wind"), "color"] != ""
    # Same carrier has same color across scenarios
    assert (
        n.c.carriers.static.loc[("low", "wind"), "color"]
        == n.c.carriers.static.loc[("high", "wind"), "color"]
    )


def test_unique_carriers():
    """Test unique_carriers property - gets unique carriers from components."""
    n = Network()
    n.add("Bus", ["bus1", "bus2"])
    n.add(
        "Generator",
        ["g1", "g2", "g3"],
        bus=["bus1", "bus1", "bus2"],
        carrier=["wind", "wind", "solar"],
    )
    n.add("Generator", "g4", bus="bus2", carrier="")  # Empty ignored
    carriers = n.c.generators.unique_carriers
    assert carriers == {"solar", "wind"}
    # Empty component
    assert n.c.loads.unique_carriers == set()
    # With default carriers
    assert "AC" in n.c.buses.unique_carriers
