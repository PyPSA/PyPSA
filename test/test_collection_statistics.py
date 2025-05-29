"""Tests for NetworkCollection statistics with various groupers."""

import pandas as pd
import pytest

import pypsa
from pypsa.statistics.grouping import groupers


@pytest.fixture
def optimized_network_collection_from_ac_dc(ac_dc_network_r):
    """Create a NetworkCollection from the already optimized ac_dc_network_r fixture."""
    # Create two scenarios using the same optimized network
    return pypsa.NetworkCollection(
        [ac_dc_network_r, ac_dc_network_r],
        index=pd.Index(["base", "variant"], name="scenario"),
    )


@pytest.fixture
def simple_network():
    """Create a simple network with different bus carriers for tests that don't need optimization."""
    n = pypsa.Network()
    n.snapshots = pd.date_range("2023-01-01", periods=3, freq="h")

    # Add carriers
    n.add("Carrier", ["AC", "DC", "gas", "wind", "load"])

    # Add buses with different carriers
    n.add("Bus", ["bus_ac1", "bus_ac2", "bus_dc1"], carrier=["AC", "AC", "DC"])

    # Add generators on different buses with costs
    n.add(
        "Generator",
        "gas_gen",
        bus="bus_ac1",
        carrier="gas",
        p_nom=100,
        marginal_cost=50,
        capital_cost=1000,
    )
    n.add(
        "Generator",
        "wind_gen",
        bus="bus_ac2",
        carrier="wind",
        p_nom=50,
        marginal_cost=0,
        capital_cost=2000,
    )

    # Add load
    n.add("Load", "load1", bus="bus_ac1", p_set=30)
    n.add("Load", "load2", bus="bus_dc1", p_set=20)

    # Add a link between AC and DC buses
    n.add("Link", "ac_dc_link", bus0="bus_ac1", bus1="bus_dc1", p_nom=40)

    return n


class TestNetworkCollectionIndexValidation:
    """Test index validation for NetworkCollection."""

    def test_single_index_without_name_gets_default(self, simple_network):
        """Test that single index without name gets default name."""
        n1 = simple_network.copy()
        n2 = simple_network.copy()

        index = pd.Index(["a", "b"])  # No name
        nc = pypsa.NetworkCollection([n1, n2], index=index)
        assert nc.networks.index.name == "network"  # Should get default name

    def test_multiindex_without_names_raises_error(self, simple_network):
        """Test that MultiIndex without names raises ValueError."""
        networks = [simple_network.copy() for _ in range(4)]

        # Create MultiIndex without names
        index = pd.MultiIndex.from_product([["a", "b"], ["1", "2"]])
        with pytest.raises(
            ValueError, match="All levels of MultiIndex must have names"
        ):
            pypsa.NetworkCollection(networks, index=index)

    def test_multiindex_with_partial_names_raises_error(self, simple_network):
        """Test that MultiIndex with partial names raises ValueError."""
        networks = [simple_network.copy() for _ in range(4)]

        # Create MultiIndex with only one name
        index = pd.MultiIndex.from_product([["a", "b"], ["1", "2"]])
        index = index.set_names(["scenario", None])
        with pytest.raises(
            ValueError, match="All levels of MultiIndex must have names"
        ):
            pypsa.NetworkCollection(networks, index=index)


def test_network_collection_statistics_basic(optimized_network_collection_from_ac_dc):
    """Test basic statistics functionality on NetworkCollection."""
    nc = optimized_network_collection_from_ac_dc

    # Test that statistics accessor exists
    assert hasattr(nc, "statistics")

    # Test basic energy balance (using the optimized network)
    result = nc.statistics.energy_balance(groupby="carrier")
    assert isinstance(result, pd.Series)
    assert not result.empty
    assert "scenario" in result.index.names
    assert "carrier" in result.index.names


def test_network_collection_bus_carrier_grouper(
    optimized_network_collection_from_ac_dc,
):
    """Test bus_carrier grouper with NetworkCollection."""
    nc = optimized_network_collection_from_ac_dc

    # Test energy balance with bus_carrier grouping
    result = nc.statistics.energy_balance(groupby="bus_carrier")
    assert not result.empty
    assert "bus_carrier" in result.index.names
    assert "AC" in result.index.get_level_values("bus_carrier")

    # Energy balance should sum to approximately zero for each scenario
    for scenario in ["base", "variant"]:
        scenario_result = result.xs(scenario, level="scenario")
        balance = scenario_result.sum()
        assert abs(balance) < 1e-6, f"Energy balance not zero for {scenario}"


def test_network_collection_carrier_bus_carrier_grouper(
    optimized_network_collection_from_ac_dc,
):
    """Test combined carrier and bus_carrier grouping with NetworkCollection."""
    nc = optimized_network_collection_from_ac_dc

    # Test with list groupby
    result = nc.statistics.energy_balance(groupby=["carrier", "bus_carrier"])
    assert not result.empty
    assert "carrier" in result.index.names
    assert "bus_carrier" in result.index.names
    assert "scenario" in result.index.names

    # Check that gas generators are only on AC buses
    if "gas" in result.index.get_level_values("carrier"):
        gas_results = result.xs("gas", level="carrier", drop_level=False)
        assert all(gas_results.index.get_level_values("bus_carrier") == "AC")

    # Check that wind generators are only on AC buses
    if "wind" in result.index.get_level_values("carrier"):
        wind_results = result.xs("wind", level="carrier", drop_level=False)
        assert all(wind_results.index.get_level_values("bus_carrier") == "AC")


def test_network_collection_country_grouper(simple_network):
    """Test country grouper with NetworkCollection."""
    # Add country information to buses
    simple_network.buses["country"] = ["DE", "DE", "FR"]

    nc = pypsa.NetworkCollection(
        [simple_network, simple_network.copy()],
        index=pd.Index(["s1", "s2"], name="scenario"),
    )

    # No need to optimize for installed_capacity
    # Test grouping by country with installed_capacity
    result = nc.statistics.installed_capacity(
        groupby=["carrier", "bus_carrier", "country"]
    )
    assert not result.empty
    assert "country" in result.index.names
    assert "DE" in result.index.get_level_values("country")


def test_network_collection_location_grouper(simple_network):
    """Test location grouper with NetworkCollection."""
    # Add location information to buses
    simple_network.buses["location"] = ["Berlin", "Munich", "Paris"]

    nc = pypsa.NetworkCollection(
        [simple_network, simple_network.copy()],
        index=pd.Index(["s1", "s2"], name="scenario"),
    )

    # No need to optimize for installed_capacity
    # Test grouping by location with installed_capacity
    result = nc.statistics.installed_capacity(
        groupby=["carrier", "bus_carrier", "location"]
    )
    assert not result.empty
    assert "location" in result.index.names
    # Check that we have the expected locations (only those with components)
    locations = result.index.get_level_values("location").unique()
    assert "Berlin" in locations  # bus_ac1 has gas_gen
    assert "Munich" in locations  # bus_ac2 has wind_gen
    # Note: Paris (bus_dc1) has no generators, only loads, so won't appear in installed_capacity


def test_network_collection_unit_grouper(simple_network):
    """Test unit grouper with NetworkCollection."""
    # Add unit information to buses
    simple_network.buses["unit"] = ["MW", "MW", "MVA"]

    nc = pypsa.NetworkCollection(
        [simple_network, simple_network.copy()],
        index=pd.Index(["s1", "s2"], name="scenario"),
    )

    # No need to optimize for installed_capacity
    # Test grouping by unit with installed_capacity
    result = nc.statistics.installed_capacity(groupby=["unit"])
    assert not result.empty
    assert "unit" in result.index.names
    units = result.index.get_level_values("unit").unique()
    assert "MW" in units  # Generators are on MW buses
    # Note: MVA bus has no generators, only loads, so won't appear in installed_capacity


def test_network_collection_multiple_groupers(optimized_network_collection_from_ac_dc):
    """Test multiple groupers simultaneously with NetworkCollection."""
    nc = optimized_network_collection_from_ac_dc

    # Test with callable grouper from groupers module
    result = nc.statistics.energy_balance(groupby=groupers[["carrier", "bus_carrier"]])
    assert not result.empty
    assert "carrier" in result.index.names
    assert "bus_carrier" in result.index.names

    # Test mixing string and callable groupers with installed_capacity (doesn't need optimization)
    result2 = nc.statistics.installed_capacity(
        groupby=["carrier", groupers.bus_carrier]
    )
    assert not result2.empty
    assert "carrier" in result2.index.names
    assert "bus_carrier" in result2.index.names


def test_network_collection_multiindex_scenarios(ac_dc_network_r):
    """Test NetworkCollection with MultiIndex scenarios."""
    # Create multiple copies of the already optimized network
    networks = [ac_dc_network_r.copy() for _ in range(4)]

    # Create MultiIndex for scenarios
    index = pd.MultiIndex.from_product(
        [["low", "high"], [2030, 2040]], names=["demand", "year"]
    )

    nc = pypsa.NetworkCollection(networks, index=index)

    # Test statistics with MultiIndex using energy_balance (since networks are already optimized)
    result = nc.statistics.energy_balance(groupby=["carrier", "bus_carrier"])
    assert not result.empty
    assert result.index.names == [
        "component",
        "demand",
        "year",
        "carrier",
        "bus_carrier",
    ]

    # Check we can access specific scenarios
    high_2040 = result.xs(("high", 2040), level=["demand", "year"])
    assert not high_2040.empty


def test_network_collection_vs_single_network(ac_dc_network_r):
    """Test that NetworkCollection statistics match single network statistics."""
    # Get single network results using installed_capacity (no optimization needed)
    single_result = ac_dc_network_r.statistics.installed_capacity(
        groupby=["carrier", "bus_carrier"]
    )

    # Create NetworkCollection with one network
    nc = pypsa.NetworkCollection([ac_dc_network_r], index=pd.Index(["s1"]))

    # Get collection results
    collection_result = nc.statistics.installed_capacity(
        groupby=["carrier", "bus_carrier"]
    )

    # Results should match (except for the network index level)
    collection_values = collection_result.xs("s1", level="network")
    pd.testing.assert_series_equal(
        single_result.sort_index(), collection_values.sort_index(), check_names=False
    )


def test_network_collection_empty_results(ac_dc_network_r):
    """Test NetworkCollection behavior with filters that produce empty results."""
    nc = pypsa.NetworkCollection([ac_dc_network_r], index=pd.Index(["s1"]))

    # Filter for non-existent bus_carrier should raise ValueError for now
    # (bus_carrier_unit function doesn't handle non-existent carriers well)
    with pytest.raises(ValueError, match="Bus carriers.*not in network"):
        nc.statistics.energy_balance(
            bus_carrier="NonExistent", groupby=["carrier", "bus_carrier"]
        )


def test_network_collection_custom_grouper(ac_dc_network_r):
    """Test NetworkCollection with custom grouper function."""
    nc = pypsa.NetworkCollection([ac_dc_network_r], index=pd.Index(["s1"]))

    # Define custom grouper that groups by first letter of generator name
    def first_letter_grouper(n, c, **kwargs):
        idx = n.static(c).index
        # Handle MultiIndex case (NetworkCollection)
        if isinstance(idx, pd.MultiIndex):
            # Get the last level (component names)
            component_names = idx.get_level_values(-1)
            first_letters = component_names.str[0]
            # Recreate the full index
            return pd.Series(first_letters.values, index=idx, name="first_letter")
        else:
            # Single network case
            return idx.str[0].rename("first_letter")

    # Register the custom grouper
    groupers.add_grouper("first_letter", first_letter_grouper)

    # Test with custom grouper using installed_capacity (no optimization needed)
    result = nc.statistics.installed_capacity(
        comps=["Generator"], groupby=["carrier", "bus_carrier", "first_letter"]
    )
    assert not result.empty
    assert "first_letter" in result.index.names


def test_network_collection_get_transmission_branches(simple_network):
    """Test get_transmission_branches function with NetworkCollection."""
    from pypsa.statistics.expressions import get_transmission_branches

    # Add a line between AC buses to test transmission branch detection
    simple_network.add("Line", "ac_line", bus0="bus_ac1", bus1="bus_ac2", r=0.1, x=0.1)

    # Create a NetworkCollection with different bus carriers
    nc = pypsa.NetworkCollection(
        [simple_network, simple_network.copy()],
        index=pd.Index(["s1", "s2"], name="scenario"),
    )

    # Test getting transmission branches for AC buses
    branches = get_transmission_branches(nc, bus_carrier="AC")
    assert isinstance(branches, pd.MultiIndex)
    # Should include scenario level in the index
    assert branches.names == ["scenario", "component", "name"]

    # Should find the line between AC buses for both scenarios
    assert "Line" in branches.get_level_values("component")
    assert "ac_line" in branches.get_level_values("name")
    assert "s1" in branches.get_level_values("scenario")
    assert "s2" in branches.get_level_values("scenario")

    # Should have 2 entries (one for each scenario)
    assert len(branches) == 2

    # Should NOT find the ac_dc_link since it connects different carriers
    assert "ac_dc_link" not in branches.get_level_values("name")

    # Test with DC buses - should be empty since we only have one DC bus
    branches_dc = get_transmission_branches(nc, bus_carrier="DC")
    assert len(branches_dc) == 0
    # Should still have correct structure even when empty
    assert branches_dc.names == ["scenario", "component", "name"]

    # Test with multiple bus carriers
    branches_multi = get_transmission_branches(nc, bus_carrier=["AC", "DC"])
    assert isinstance(branches_multi, pd.MultiIndex)
    # Should only find AC line since DC has no transmission branches
    assert "ac_line" in branches_multi.get_level_values("name")
    assert len(branches_multi) == 2  # Two scenarios

    # Test with None (all bus carriers)
    branches_all = get_transmission_branches(nc, bus_carrier=None)
    assert isinstance(branches_all, pd.MultiIndex)
    assert "ac_line" in branches_all.get_level_values("name")
    assert len(branches_all) == 2  # Two scenarios


def test_network_collection_get_transmission_carriers(simple_network):
    """Test get_transmission_carriers function with NetworkCollection."""
    from pypsa.statistics.expressions import get_transmission_carriers

    # Add carriers for transmission components
    simple_network.add("Carrier", "transmission")

    # Add a line between AC buses with transmission carrier
    simple_network.add(
        "Line",
        "ac_line",
        bus0="bus_ac1",
        bus1="bus_ac2",
        r=0.1,
        x=0.1,
        carrier="transmission",
    )

    # Create a NetworkCollection
    nc = pypsa.NetworkCollection(
        [simple_network, simple_network.copy()],
        index=pd.Index(["s1", "s2"], name="scenario"),
    )

    # Test getting transmission carriers for AC buses
    carriers = get_transmission_carriers(nc, bus_carrier="AC")
    assert isinstance(carriers, pd.MultiIndex)
    # Should include scenario level in the index
    assert carriers.names == ["scenario", "component", "carrier"]

    # Should find the transmission carrier for lines in both scenarios
    assert "Line" in carriers.get_level_values("component")
    assert "transmission" in carriers.get_level_values("carrier")
    assert "s1" in carriers.get_level_values("scenario")
    assert "s2" in carriers.get_level_values("scenario")

    # Should have 2 entries (one for each scenario)
    assert len(carriers) == 2

    # Test with DC buses - should be empty
    carriers_dc = get_transmission_carriers(nc, bus_carrier="DC")
    assert len(carriers_dc) == 0
    # Should still have correct structure even when empty
    assert carriers_dc.names == ["scenario", "component", "carrier"]


def test_network_collection_default_energy_balance_groupby(
    optimized_network_collection_from_ac_dc,
):
    """Test that default groupby for energy_balance works with NetworkCollection."""
    nc = optimized_network_collection_from_ac_dc

    # Test with default groupby (should be ["carrier", "bus_carrier"])
    result = nc.statistics.energy_balance()
    assert not result.empty
    assert "carrier" in result.index.names
    assert "bus_carrier" in result.index.names
    assert "scenario" in result.index.names  # Uses 'scenario' as index name


def test_network_collection_opex_and_capex(
    optimized_network_collection_from_ac_dc,
):
    """Test OPEX and CAPEX statistics with NetworkCollection."""
    nc = optimized_network_collection_from_ac_dc

    # Test OPEX calculation
    opex_result = nc.statistics.opex()
    assert isinstance(opex_result, pd.Series)
    assert not opex_result.empty
    assert "scenario" in opex_result.index.names

    # Test CAPEX calculation
    capex_result = nc.statistics.capex()
    assert isinstance(capex_result, pd.Series)
    assert not capex_result.empty
    assert "scenario" in capex_result.index.names


def test_network_collection_transmission(
    optimized_network_collection_from_ac_dc,
):
    """Test transmission statistics with NetworkCollection."""
    nc = optimized_network_collection_from_ac_dc

    # Test transmission statistics
    transmission_result = nc.statistics.transmission()
    assert isinstance(transmission_result, pd.Series)
    assert not transmission_result.empty
    assert "scenario" in transmission_result.index.names


def test_network_collection_revenue(
    optimized_network_collection_from_ac_dc,
):
    """Test revenue statistics with NetworkCollection."""
    nc = optimized_network_collection_from_ac_dc

    # Test revenue calculation
    revenue_result = nc.statistics.revenue()
    assert isinstance(revenue_result, pd.Series)
    assert not revenue_result.empty
    assert "scenario" in revenue_result.index.names

    # Check that revenue is non-negative for each scenario (typical for generators)
    for scenario in nc.networks.index:
        scenario_revenue = revenue_result.xs(scenario, level="scenario")
        network_revenue = nc.networks.loc[scenario].statistics.revenue()
        assert (scenario_revenue == network_revenue).all(), (
            f"Revenue mismatch for scenario {scenario}"
        )
