# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest

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

    # Check that all carriers were added (including "AC" from buses)
    assert len(added) == 4
    assert set(added) == {"AC", "gas", "solar", "wind"}  # alphabetically sorted
    assert len(n.c.carriers.static) == 4

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

    assert len(added) == 4
    assert set(added) == {"AC", "battery", "electricity", "wind"}
    assert len(n.c.carriers.static) == 4


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

    # Solar and AC should be added (wind already exists)
    assert len(added) == 2
    assert set(added) == {"AC", "solar"}
    assert len(n.c.carriers.static) == 3

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

    # Only AC carrier should be added (from buses)
    assert len(added) == 1
    assert added[0] == "AC"
    assert len(n.c.carriers.static) == 2


def test_add_missing_carriers_with_empty_and_nan():
    """Test that empty strings and NaN carriers are ignored."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators with various carrier values
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="")  # empty string
    n.add("Generator", "gen3", bus="bus1")  # carrier defaults to empty

    added = n.c.carriers.add_missing_carriers()

    # Wind and AC should be added (empty strings and NaN are ignored)
    assert len(added) == 2
    assert set(added) == {"AC", "wind"}


def test_add_missing_carriers_custom_palette():
    """Test add_missing_carriers with different color palettes."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="solar")

    # Add carriers with tab20 palette
    added = n.c.carriers.add_missing_carriers(palette="tab20")

    assert len(added) == 3
    assert all(n.c.carriers.static["color"].notna())

    # Colors should be different for at least some carriers
    colors = n.c.carriers.static["color"].values
    assert len(set(colors)) >= 2  # At least 2 different colors


def test_add_missing_carriers_with_kwargs():
    """Test add_missing_carriers with additional kwargs."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="gas")

    # Add carriers with co2_emissions
    added = n.c.carriers.add_missing_carriers(co2_emissions=0.5)

    assert len(added) == 3
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

    # All carriers should be added (including AC from bus)
    assert len(added) == 16
    # All should have colors (cycling through palette)
    assert all(n.c.carriers.static["color"].notna())


def test_generate_colors_tab10():
    """Test generate_colors with tab10 palette."""
    from pypsa.common import generate_colors

    colors = generate_colors(5, "tab10")

    assert len(colors) == 5
    # All colors should be valid hex strings
    assert all(c.startswith("#") for c in colors)
    # Colors should be unique for small numbers
    assert len(set(colors)) == 5


def test_generate_colors_invalid_palette():
    """Test generate_colors with invalid palette falls back to tab10."""
    from pypsa.common import generate_colors

    colors = generate_colors(5, "invalid_palette_name")

    # Should still generate colors using fallback
    assert len(colors) == 5
    assert all(c.startswith("#") for c in colors)


def test_generate_colors_cycling():
    """Test that colors cycle when more carriers than palette colors."""
    from pypsa.common import generate_colors

    # Get 12 colors from tab10 (which has 10 colors)
    colors = generate_colors(12, "tab10")

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

    assert len(added) == 3
    assert set(added) == {"AC", "H2", "HVDC"}


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
    added = n.c.carriers.add_missing_carriers(palette=None)

    assert len(added) == 3
    assert set(added) == {"AC", "solar", "wind"}
    # Check that no colors were assigned (should be empty or NaN)
    assert n.c.carriers.static["color"]["wind"] == ""
    assert n.c.carriers.static["color"]["solar"] == ""
    assert n.c.carriers.static["color"]["AC"] == ""


def test_add_missing_carriers_explicit_color_in_kwargs():
    """Test that explicit color in kwargs overrides auto-assignment."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="solar")

    # Add carriers with explicit colors
    added = n.c.carriers.add_missing_carriers(
        palette=None, color=["red", "blue", "green"]
    )

    assert len(added) == 3
    # Check that explicit colors were used (not auto-generated)
    # Note: carriers are added alphabetically (AC, solar, wind)
    assert n.c.carriers.static["color"]["AC"] == "red"
    assert n.c.carriers.static["color"]["solar"] == "blue"
    assert n.c.carriers.static["color"]["wind"] == "green"


def test_add_missing_carriers_no_colors_with_other_kwargs():
    """Test add_missing_carriers without colors but with other attributes."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="gas")

    # Add carriers without colors but with other attributes
    added = n.c.carriers.add_missing_carriers(
        palette=None, co2_emissions=0.2, nice_name="Test Carrier"
    )

    assert len(added) == 3
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

    # Check that all carriers were added (including AC from buses)
    assert len(added) == 4
    assert set(added) == {"AC", "gas", "solar", "wind"}  # alphabetically sorted

    # Check that carriers were added for all scenarios
    assert n.c.carriers.static.index.nlevels == 2
    assert list(n.c.carriers.static.index.names) == ["scenario", "name"]

    # Each carrier should exist in both scenarios
    scenarios = ["scenario_1", "scenario_2"]
    carriers = ["AC", "gas", "solar", "wind"]
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

    # Solar and AC should be added (wind already exists)
    assert len(added) == 2
    assert set(added) == {"AC", "solar"}

    # Check that wind's color wasn't changed in both scenarios
    wind_low = n.c.carriers.static.xs("low", level="scenario").loc["wind", "color"]
    wind_high = n.c.carriers.static.xs("high", level="scenario").loc["wind", "color"]
    assert wind_low == "blue"
    assert wind_high == "blue"

    # Check that solar and AC were added to both scenarios
    assert ("low", "solar") in n.c.carriers.static.index
    assert ("high", "solar") in n.c.carriers.static.index
    assert ("low", "AC") in n.c.carriers.static.index
    assert ("high", "AC") in n.c.carriers.static.index


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

    # Only AC carrier should be added (from buses)
    assert len(added) == 1
    assert added[0] == "AC"

    # Wind should still exist in all scenarios
    assert ("scenario_a", "wind") in n.c.carriers.static.index
    assert ("scenario_b", "wind") in n.c.carriers.static.index
    assert ("scenario_c", "wind") in n.c.carriers.static.index

    # AC should be added to all scenarios
    assert ("scenario_a", "AC") in n.c.carriers.static.index
    assert ("scenario_b", "AC") in n.c.carriers.static.index
    assert ("scenario_c", "AC") in n.c.carriers.static.index


def test_assign_colors_basic():
    """Test basic functionality of assign_colors."""
    n = Network()
    n.add("Bus", "bus1")

    # Add carriers without colors
    n.add("Carrier", "wind")
    n.add("Carrier", "solar")
    n.add("Carrier", "gas")

    # Initially no colors
    assert all(n.c.carriers.static["color"] == "")

    # Assign colors
    n.c.carriers.assign_colors()

    # All carriers should now have colors
    assert all(n.c.carriers.static["color"].notna())
    assert all(n.c.carriers.static["color"] != "")
    # Colors should be different
    colors = n.c.carriers.static["color"].values
    assert len(set(colors)) == 3


def test_assign_colors_specific_carriers():
    """Test assign_colors on specific carriers only."""
    n = Network()
    n.add("Bus", "bus1")

    # Add carriers
    n.add("Carrier", "wind")
    n.add("Carrier", "solar")
    n.add("Carrier", "gas")

    # Assign colors only to wind and solar
    n.c.carriers.assign_colors(["wind", "solar"])

    # Wind and solar should have colors, gas should not
    assert n.c.carriers.static.loc["wind", "color"] != ""
    assert n.c.carriers.static.loc["solar", "color"] != ""
    assert n.c.carriers.static.loc["gas", "color"] == ""


def test_assign_colors_different_palette():
    """Test assign_colors with different palettes."""
    n = Network()
    n.add("Bus", "bus1")

    # Add carriers
    n.add("Carrier", "wind")
    n.add("Carrier", "solar")

    # Assign colors with tab20 palette
    n.c.carriers.assign_colors(palette="tab20")

    # Both carriers should have colors
    assert all(n.c.carriers.static["color"].notna())
    assert all(n.c.carriers.static["color"] != "")


def test_assign_colors_overwrite():
    """Test assign_colors with overwrite=True."""
    n = Network()
    n.add("Bus", "bus1")

    # Add carriers with existing colors
    n.add("Carrier", "wind", color="blue")
    n.add("Carrier", "solar", color="yellow")

    # Store original colors
    original_wind = n.c.carriers.static.loc["wind", "color"]
    original_solar = n.c.carriers.static.loc["solar", "color"]

    # Assign colors without overwrite (default)
    n.c.carriers.assign_colors(overwrite=False)

    # Colors should not have changed
    assert n.c.carriers.static.loc["wind", "color"] == original_wind
    assert n.c.carriers.static.loc["solar", "color"] == original_solar

    # Assign colors with overwrite
    n.c.carriers.assign_colors(overwrite=True)

    # Colors should have changed
    assert n.c.carriers.static.loc["wind", "color"] != original_wind
    assert n.c.carriers.static.loc["solar", "color"] != original_solar


def test_assign_colors_partial_colors():
    """Test assign_colors when some carriers have colors and some don't."""
    n = Network()
    n.add("Bus", "bus1")

    # Add carriers, some with colors
    n.add("Carrier", "wind", color="blue")
    n.add("Carrier", "solar")  # no color
    n.add("Carrier", "gas")  # no color

    # Assign colors (should only assign to solar and gas)
    n.c.carriers.assign_colors()

    # Wind should keep its color
    assert n.c.carriers.static.loc["wind", "color"] == "blue"

    # Solar and gas should have new colors
    assert n.c.carriers.static.loc["solar", "color"] != ""
    assert n.c.carriers.static.loc["gas", "color"] != ""


def test_assign_colors_stochastic():
    """Test assign_colors with stochastic networks."""
    n = Network()
    n.add("Bus", "bus1")

    # Add carriers before scenarios
    n.add("Carrier", "wind")
    n.add("Carrier", "solar")

    # Convert to stochastic network
    n.set_scenarios(["low", "high"])

    # Assign colors
    n.c.carriers.assign_colors()

    # Colors should be assigned for both scenarios
    assert n.c.carriers.static.loc[("low", "wind"), "color"] != ""
    assert n.c.carriers.static.loc[("high", "wind"), "color"] != ""
    assert n.c.carriers.static.loc[("low", "solar"), "color"] != ""
    assert n.c.carriers.static.loc[("high", "solar"), "color"] != ""

    # Colors should be the same across scenarios for same carrier
    assert (
        n.c.carriers.static.loc[("low", "wind"), "color"]
        == n.c.carriers.static.loc[("high", "wind"), "color"]
    )
    assert (
        n.c.carriers.static.loc[("low", "solar"), "color"]
        == n.c.carriers.static.loc[("high", "solar"), "color"]
    )


def test_unique_carriers_property():
    """Test unique_carriers property on components."""
    n = Network()
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")

    # Add generators with different carriers
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="wind")
    n.add("Generator", "gen3", bus="bus2", carrier="solar")
    n.add("Generator", "gen4", bus="bus2", carrier="gas")

    # Get unique carriers from generators
    carriers = n.c.generators.unique_carriers

    assert isinstance(carriers, set)
    assert len(carriers) == 3
    assert carriers == {"wind", "solar", "gas"}


def test_unique_carriers_empty_component():
    """Test unique_carriers on empty component."""
    n = Network()

    # Generators is empty
    carriers = n.c.generators.unique_carriers

    assert isinstance(carriers, set)
    assert len(carriers) == 0


def test_unique_carriers_no_carrier_attribute():
    """Test unique_carriers on component without carrier attribute."""
    n = Network()
    n.add("Bus", "bus1")
    n.add("Bus", "bus2")
    n.add("Bus", "bus3", carrier="DC")

    # Buses have carrier attribute
    carriers = n.c.buses.unique_carriers

    assert isinstance(carriers, set)
    assert len(carriers) == 2
    assert carriers == {"AC", "DC"}


def test_unique_carriers_with_empty_strings():
    """Test unique_carriers ignores empty strings and NaN."""
    n = Network()
    n.add("Bus", "bus1")

    # Add generators with various carrier values
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="")  # empty string
    n.add("Generator", "gen3", bus="bus1")  # defaults to empty

    # Get unique carriers
    carriers = n.c.generators.unique_carriers

    # Only wind should be in the set (empty strings filtered out)
    assert carriers == {"wind"}


def test_add_missing_carriers_color_palette_conflict():
    """Test that add_missing_carriers raises error when both palette and color provided."""
    n = Network()
    n.add("Bus", "bus1")
    n.add("Generator", "gen1", bus="bus1", carrier="wind")

    # Should raise ValueError when both palette and color are provided
    with pytest.raises(ValueError, match="Cannot specify both 'palette' and 'color'"):
        n.c.carriers.add_missing_carriers(palette="tab10", color="red")


def test_assign_colors_single_string():
    """Test assign_colors with a single carrier name as string."""
    n = Network()
    n.add("Bus", "bus1")

    # Add carriers
    n.add("Carrier", "wind")
    n.add("Carrier", "solar")
    n.add("Carrier", "gas")

    # Assign color to only "wind" using a string instead of list
    n.c.carriers.assign_colors("wind")

    # Only wind should have a color
    assert n.c.carriers.static.loc["wind", "color"] != ""
    assert n.c.carriers.static.loc["solar", "color"] == ""
    assert n.c.carriers.static.loc["gas", "color"] == ""


def test_add_missing_carriers_stochastic_no_colors():
    """Test add_missing_carriers on stochastic network without automatic color assignment."""
    n = Network()
    n.add("Bus", "bus1")

    # Add components before scenarios
    n.add("Generator", "gen1", bus="bus1", carrier="wind")
    n.add("Generator", "gen2", bus="bus1", carrier="solar")

    # Convert to stochastic network
    n.set_scenarios(["low", "high"])

    # Add missing carriers without colors
    added = n.c.carriers.add_missing_carriers(palette=None)

    # Check carriers were added
    assert len(added) == 3
    assert set(added) == {"AC", "solar", "wind"}

    # Check that carriers exist in both scenarios
    for scenario in ["low", "high"]:
        for carrier in ["AC", "solar", "wind"]:
            assert (scenario, carrier) in n.c.carriers.static.index
            # Check no colors were assigned
            assert n.c.carriers.static.loc[(scenario, carrier), "color"] == ""
