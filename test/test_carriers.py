# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from pypsa import Network


def test_add_missing_carriers_basic():
    """Test basic functionality of add_missing_carriers."""
    n = Network()
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")

    # Add generators with carriers that don't exist yet
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus2", carrier="solar")
    n.add("Generator", "gen3", bus="bus1", carrier="gas")

    # Initially no carriers should exist
    assert len(n.c.carriers.static) == 0

    # Add missing carriers
    added = n.c.carriers.add_missing_carriers()

    # Check that all carriers were added
    assert len(added) == 3
    assert set(added) == {"gas", "solar", "wind"}  # alphabetically sorted
    assert len(n.c.carriers.static) == 3

    # Check that colors were assigned
    assert all(n.c.carriers.static["color"].notna())


def test_add_missing_carriers_multiple_components():
    """Test add_missing_carriers with multiple component types."""
    n = Network()
    n.add("Bus", "bus1")

    # Add different component types with carriers
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Load", "load1", bus="bus1", carrier="electricity")
    n.add("StorageUnit", "storage1", bus="bus1", carrier="battery")

    added = n.c.carriers.add_missing_carriers()

    assert len(added) == 3
    assert set(added) == {"battery", "electricity", "wind"}
    assert len(n.c.carriers.static) == 3


def test_add_missing_carriers_with_existing():
    """Test add_missing_carriers when some carriers already exist."""
    n = Network()
    n.add("Bus", "bus1")

    # Add a carrier manually
    n.add("Carrier", "wind", color="blue")

    # Add generators with carriers
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="solar")

    # Add missing carriers
    added = n.c.carriers.add_missing_carriers()

    # Only solar should be added (wind already exists)
    assert len(added) == 1
    assert added[0] == "solar"
    assert len(n.c.carriers.static) == 2

    # Check that wind's color wasn't changed
    assert n.c.carriers.static.loc["wind", "color"] == "blue"


def test_add_missing_carriers_no_missing():
    """Test add_missing_carriers when no carriers are missing."""
    n = Network()
    n.add("Bus", "bus1")

    # Add carrier first
    n.add("Carrier", "wind", color="blue")

    # Add generator with existing carrier
    n.add("Generator", "gen1", bus="bus1", carrier="wind")

    # Try to add missing carriers
    added = n.c.carriers.add_missing_carriers()

    # No carriers should be added
    assert len(added) == 0
    assert len(n.c.carriers.static) == 1


def test_add_missing_carriers_with_empty_and_nan():
    """Test that empty strings and NaN carriers are ignored."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators with various carrier values
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="")  # empty string
    n.add("Generator", "gen3", bus="bus1")  # carrier defaults to empty

    added = n.c.carriers.add_missing_carriers()

    # Only wind should be added (empty strings and NaN are ignored)
    assert len(added) == 1
    assert added[0] == "wind"


def test_add_missing_carriers_custom_palette():
    """Test add_missing_carriers with different color palettes."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="solar")

    # Add carriers with tab20 palette
    added = n.c.carriers.add_missing_carriers(color_palette="tab20")

    assert len(added) == 2
    assert all(n.c.carriers.static["color"].notna())

    # Colors should be different
    colors = n.c.carriers.static["color"].values
    assert colors[0] != colors[1]


def test_add_missing_carriers_with_kwargs():
    """Test add_missing_carriers with additional kwargs."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="gas")

    # Add carriers with co2_emissions
    added = n.c.carriers.add_missing_carriers(co2_emissions=0.5)

    assert len(added) == 2
    # Check that co2_emissions was set
    assert all(n.c.carriers.static["co2_emissions"] == 0.5)


def test_add_missing_carriers_many_carriers():
    """Test add_missing_carriers with more carriers than colors in palette."""
    n = Network()
    n.add("Bus", "bus1")

    # Add 15 generators (more than tab10's 10 colors)
    for i in range(15):
        n.add("Generator", f"gen{i}", bus="bus1", carrier=f"carrier_{i}")

    added = n.c.carriers.add_missing_carriers()

    # All carriers should be added
    assert len(added) == 15
    # All should have colors (cycling through palette)
    assert all(n.c.carriers.static["color"].notna())


def test_generate_colors_tab10():
    """Test _generate_colors with tab10 palette."""
    n = Network()

    colors = n.c.carriers._generate_colors(5, "tab10")

    assert len(colors) == 5
    # All colors should be valid hex strings
    assert all(c.startswith("#") for c in colors)
    # Colors should be unique for small numbers
    assert len(set(colors)) == 5


def test_generate_colors_invalid_palette():
    """Test _generate_colors with invalid palette falls back to tab10."""
    n = Network()

    colors = n.c.carriers._generate_colors(5, "invalid_palette_name")

    # Should still generate colors using fallback
    assert len(colors) == 5
    assert all(c.startswith("#") for c in colors)


def test_generate_colors_cycling():
    """Test that colors cycle when more carriers than palette colors."""
    n = Network()

    # Get 12 colors from tab10 (which has 10 colors)
    colors = n.c.carriers._generate_colors(12, "tab10")

    assert len(colors) == 12
    # First and 11th color should be the same (cycling)
    assert colors[0] == colors[10]
    assert colors[1] == colors[11]


def test_add_missing_carriers_deterministic():
    """Test that carrier colors are deterministic across runs."""
    # Create two networks with same carriers
    n1 = Network()
    n1.add("Bus", "bus1")
    n1.add("Generator", "gen1", bus="bus1", carrier="wind")
    n1.add("Generator", "gen2", bus="bus1", carrier="solar")
    n1.c.carriers.add_missing_carriers()

    n2 = Network()
    n2.add("Bus", "bus1")
    n2.add("Generator", "gen1", bus="bus1", carrier="wind")
    n2.add("Generator", "gen2", bus="bus1", carrier="solar")
    n2.c.carriers.add_missing_carriers()

    # Colors should be identical
    assert (
        n1.c.carriers.static["color"]["wind"] == n2.c.carriers.static["color"]["wind"]
    )
    assert (
        n1.c.carriers.static["color"]["solar"] == n2.c.carriers.static["color"]["solar"]
    )


def test_add_missing_carriers_links():
    """Test add_missing_carriers with Link components."""
    n = Network()
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")

    # Add links with carriers
    n.add("Link", "link1", bus0="bus1", bus1="bus2", carrier="HVDC")
    n.add("Link", "link2", bus0="bus1", bus1="bus2", carrier="H2")

    added = n.c.carriers.add_missing_carriers()

    assert len(added) == 2
    assert set(added) == {"H2", "HVDC"}


def test_add_missing_carriers_empty_network():
    """Test add_missing_carriers on empty network."""
    n = Network()

    added = n.c.carriers.add_missing_carriers()

    assert len(added) == 0
    assert len(n.c.carriers.static) == 0


def test_add_missing_carriers_no_colors():
    """Test add_missing_carriers without automatic color assignment."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="solar")

    # Add carriers without colors
    added = n.c.carriers.add_missing_carriers(assign_colors=False)

    assert len(added) == 2
    assert set(added) == {"solar", "wind"}
    # Check that no colors were assigned (should be empty or NaN)
    assert n.c.carriers.static["color"]["wind"] == ""
    assert n.c.carriers.static["color"]["solar"] == ""


def test_add_missing_carriers_explicit_color_in_kwargs():
    """Test that explicit color in kwargs overrides auto-assignment."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="solar")

    # Add carriers with explicit colors, even with assign_colors=True
    added = n.c.carriers.add_missing_carriers(assign_colors=True, color=["red", "blue"])

    assert len(added) == 2
    # Check that explicit colors were used (not auto-generated)
    # Note: solar comes before wind alphabetically
    assert n.c.carriers.static["color"]["solar"] == "red"
    assert n.c.carriers.static["color"]["wind"] == "blue"


def test_add_missing_carriers_no_colors_with_other_kwargs():
    """Test add_missing_carriers without colors but with other attributes."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="gas")

    # Add carriers without colors but with other attributes
    added = n.c.carriers.add_missing_carriers(
        assign_colors=False, co2_emissions=0.2, nice_name="Test Carrier"
    )

    assert len(added) == 2
    # Check that co2_emissions was set but colors were not
    assert all(n.c.carriers.static["co2_emissions"] == 0.2)
    assert all(n.c.carriers.static["color"] == "")
    assert all(n.c.carriers.static["nice_name"] == "Test Carrier")


def test_add_missing_carriers_stochastic_network():
    """Test add_missing_carriers works correctly with stochastic networks."""
    n = Network()
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")

    # Add components with carriers before setting scenarios
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus2", carrier="solar")

    # Convert to stochastic network
    n.set_scenarios({"scenario_1": 0.5, "scenario_2": 0.5})

    # Add more components after scenarios are set
    n.c.generators.static.loc[("scenario_1", "gen1"), "carrier"] = "wind"
    n.c.generators.static.loc[("scenario_2", "gen1"), "carrier"] = "wind"
    n.add("Generator", "gen3", bus="bus1", carrier="gas")

    # Initially no carriers should exist
    assert len(n.c.carriers.static) == 0

    # Add missing carriers
    added = n.c.carriers.add_missing_carriers()

    # Check that all carriers were added
    assert len(added) == 3
    assert set(added) == {"gas", "solar", "wind"}  # alphabetically sorted

    # Check that carriers were added for all scenarios
    assert n.c.carriers.static.index.nlevels == 2
    assert list(n.c.carriers.static.index.names) == ["scenario", "name"]

    # Each carrier should exist in both scenarios
    scenarios = ["scenario_1", "scenario_2"]
    carriers = ["gas", "solar", "wind"]
    for scenario in scenarios:
        for carrier in carriers:
            assert (scenario, carrier) in n.c.carriers.static.index

    # Check that colors were assigned
    assert all(n.c.carriers.static["color"].notna())


def test_add_missing_carriers_stochastic_with_existing():
    """Test add_missing_carriers when some carriers already exist in stochastic network."""
    n = Network()
    n.add("Bus", "bus1")

    # Add a carrier manually before scenarios
    n.add("Carrier", "wind", color="blue")

    # Convert to stochastic network
    n.set_scenarios({"low": 0.5, "high": 0.5})

    # Now wind carrier should exist for both scenarios with blue color
    assert ("low", "wind") in n.c.carriers.static.index
    assert ("high", "wind") in n.c.carriers.static.index
    assert n.c.carriers.static.loc[("low", "wind"), "color"] == "blue"
    assert n.c.carriers.static.loc[("high", "wind"), "color"] == "blue"

    # Add generators with carriers
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="solar")

    # Add missing carriers
    added = n.c.carriers.add_missing_carriers()

    # Only solar should be added (wind already exists)
    assert len(added) == 1
    assert added[0] == "solar"

    # Check that wind's color wasn't changed in both scenarios
    wind_low = n.c.carriers.static.xs("low", level="scenario").loc["wind", "color"]
    wind_high = n.c.carriers.static.xs("high", level="scenario").loc["wind", "color"]
    assert wind_low == "blue"
    assert wind_high == "blue"

    # Check that solar was added to both scenarios
    assert ("low", "solar") in n.c.carriers.static.index
    assert ("high", "solar") in n.c.carriers.static.index


def test_add_missing_carriers_stochastic_no_missing():
    """Test add_missing_carriers when no carriers are missing in stochastic network."""
    n = Network()
    n.add("Bus", "bus1")

    # Add carrier first
    n.add("Carrier", "wind", color="blue")

    # Convert to stochastic network
    n.set_scenarios(["scenario_a", "scenario_b", "scenario_c"])

    # Add generator with existing carrier
    n.add("Generator", "gen1", bus="bus1", carrier="wind")

    # Try to add missing carriers
    added = n.c.carriers.add_missing_carriers()

    # No carriers should be added
    assert len(added) == 0

    # Wind should still exist in all scenarios
    assert ("scenario_a", "wind") in n.c.carriers.static.index
    assert ("scenario_b", "wind") in n.c.carriers.static.index
    assert ("scenario_c", "wind") in n.c.carriers.static.index
