import pandas as pd
import pytest

import pypsa


@pytest.fixture
def network1():
    n = pypsa.Network()
    n.add("Carrier", "wind")
    n.add("Carrier", "gas")  # Add gas carrier for consistency if needed later
    n.add("Bus", "bus1")
    n.add(
        "Generator",
        "gen_wind",
        bus="bus1",
        carrier="wind",
        p_nom=100,
        capital_cost=1000,
    )  # Added capital_cost
    return n


@pytest.fixture
def network2():
    n = pypsa.Network()
    n.add("Carrier", "solar")
    n.add("Carrier", "gas", co2_emissions=0.2)
    n.add("Bus", "bus2")
    n.add(
        "Generator",
        "gen_solar",
        bus="bus2",
        carrier="solar",
        p_nom=50,
        capital_cost=800,
    )  # Added capital_cost
    n.add(
        "Generator", "gen_gas", bus="bus2", carrier="gas", p_nom=200, capital_cost=500
    )  # Added capital_cost
    return n


@pytest.fixture
def network3():
    n = pypsa.Network()
    n.add("Carrier", "hydro")
    n.add("Bus", "bus3")
    n.add("StorageUnit", "storage_hydro", bus="bus3", carrier="hydro", p_nom=80)
    return n


def test_networks_init_list(network1, network2):
    """Test initialization with a list of networks."""
    networks_list = [network1, network2]
    networks_obj = pypsa.Networks(networks_list)
    assert len(networks_obj) == 2
    assert isinstance(networks_obj._networks, pd.Series)
    assert networks_obj._networks.index.equals(pd.RangeIndex(2, name="network"))
    assert networks_obj[0] == network1
    assert networks_obj[1] == network2


def test_networks_init_list_with_index(network1, network2):
    """Test initialization with a list and custom index."""
    networks_list = [network1, network2]
    custom_index = pd.Index(["net_A", "net_B"], name="scenario")
    networks_obj = pypsa.Networks(networks_list, index=custom_index)
    assert len(networks_obj) == 2
    assert networks_obj._networks.index.equals(custom_index)
    assert networks_obj["net_A"] == network1
    assert networks_obj["net_B"] == network2


def test_networks_init_series(network1, network2):
    """Test initialization with a pandas Series."""
    index = pd.MultiIndex.from_tuples(
        [("base", 2030), ("high_renewables", 2030)], names=["scenario", "year"]
    )
    networks_series = pd.Series([network1, network2], index=index)
    networks_obj = pypsa.Networks(networks_series)
    assert len(networks_obj) == 2
    assert networks_obj._networks.index.equals(index)
    assert networks_obj[("base", 2030)] == network1
    assert networks_obj[("high_renewables", 2030)] == network2


def test_networks_init_invalid_type():
    """Test initialization with invalid types."""
    with pytest.raises(TypeError):
        pypsa.Networks([pypsa.Network(), "not a network"])
    with pytest.raises(TypeError):
        pypsa.Networks(pd.Series([pypsa.Network(), 5]))
    with pytest.raises(TypeError):
        pypsa.Networks("just a string")


def test_networks_init_index_mismatch(network1, network2):
    """Test initialization with mismatched index length."""
    with pytest.raises(ValueError):
        pypsa.Networks([network1, network2], index=pd.Index(["A"]))


def test_networks_carriers_property(network1, network2, network3):
    """Test the carriers property."""
    networks_obj = pypsa.Networks([network1, network2, network3])
    expected_carriers = pd.concat(
        [network1.carriers, network2.carriers, network3.carriers]
    )
    expected_carriers = expected_carriers[
        ~expected_carriers.index.duplicated(keep="first")
    ].sort_index()

    pd.testing.assert_frame_equal(networks_obj.carriers, expected_carriers)


def test_networks_carriers_property_empty():
    """Test the carriers property when networks have no carriers."""
    n1 = pypsa.Network()
    n2 = pypsa.Network()
    networks_obj = pypsa.Networks([n1, n2])
    expected_carriers = pd.DataFrame(
        index=pd.Index([], name="Carrier"), columns=n1.carriers.columns, dtype=int
    )
    pd.testing.assert_frame_equal(
        networks_obj.carriers,
        expected_carriers,
        check_dtype=False,
        check_index_type=False,
    )


def test_networks_getitem_slice(network1, network2, network3):
    """Test slicing the Networks object."""
    networks_list = [network1, network2, network3]
    networks_obj = pypsa.Networks(networks_list)
    sliced_networks = networks_obj[1:]
    assert isinstance(sliced_networks, pypsa.Networks)
    assert len(sliced_networks) == 2
    assert sliced_networks[1] == network2  # Original index 1
    assert sliced_networks[2] == network3  # Original index 2
    pd.testing.assert_index_equal(
        sliced_networks._networks.index, pd.RangeIndex(1, 3, name="network")
    )


def test_networks_iteration(network1, network2):
    """Test iterating over the Networks object."""
    networks_list = [network1, network2]
    networks_obj = pypsa.Networks(networks_list)
    iterated_list = [n for n in networks_obj]
    assert iterated_list == networks_list


# Tests for NetworksStatisticsAccessor


def test_networks_statistics_capex_per_network(network1, network2):
    """Test capex calculation per network."""
    networks = pypsa.Networks([network1, network2])
    capacity = networks.statistics.installed_capacity()

    assert not capacity.empty

    capacity.plot()


def test_networks_statistics_nonexistent_method(network1):
    """Test calling a method that doesn't exist on the accessor."""
    networks_obj = pypsa.Networks([network1])
    with pytest.raises(
        AttributeError,
        match="'NetworksStatisticsAccessor' object has no attribute 'nonexistent_method'",
    ):
        networks_obj.statistics.nonexistent_method()
