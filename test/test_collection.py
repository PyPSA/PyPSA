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


def test_collection_init_list(network1, network2):
    """Test initialization with a list of networks."""
    networks = [network1, network2]
    collection = pypsa.NetworkCollection(networks)
    assert len(collection) == 2
    assert isinstance(collection.networks, pd.Series)
    assert collection.networks.index.equals(pd.RangeIndex(2, name="network"))
    assert collection[0] == network1
    assert collection[1] == network2


def test_collection_init_list_with_index(network1, network2):
    """Test initialization with a list and custom index."""
    networks = [network1, network2]
    custom_index = pd.Index(["net_A", "net_B"], name="scenario")
    collection = pypsa.NetworkCollection(networks, index=custom_index)
    assert len(collection) == 2
    assert collection.networks.index.equals(custom_index)
    assert collection["net_A"] == network1
    assert collection["net_B"] == network2


def test_collection_init_series(network1, network2):
    """Test initialization with a pandas Series."""
    index = pd.MultiIndex.from_tuples(
        [("base", 2030), ("high_renewables", 2030)], names=["scenario", "year"]
    )
    networks_series = pd.Series([network1, network2], index=index)
    collection = pypsa.NetworkCollection(networks_series)
    assert len(collection) == 2
    assert collection.networks.index.equals(index)
    assert collection[("base", 2030)] == network1
    assert collection[("high_renewables", 2030)] == network2


def test_collection_init_invalid_type():
    """Test initialization with invalid types."""
    with pytest.raises(TypeError):
        pypsa.NetworkCollection([pypsa.Network(), "not a network"])
    with pytest.raises(TypeError):
        pypsa.NetworkCollection(pd.Series([pypsa.Network(), 5]))
    with pytest.raises(TypeError):
        pypsa.NetworkCollection("just a string")


def test_collection_init_index_mismatch(network1, network2):
    """Test initialization with mismatched index length."""
    with pytest.raises(ValueError):
        pypsa.NetworkCollection([network1, network2], index=pd.Index(["A"]))


def test_collection_index_names(network1, network2):
    """Test the index names of the NetworkCollection object."""
    networks = [network1, network2]
    collection = pypsa.NetworkCollection(networks)
    assert collection.index.names == ["network"]
    assert collection.networks.index.name == "network"
    assert collection._index_names == ["network"]

    # Test with custom index
    custom_index = pd.Index(["net_A", "net_B"], name="scenario")
    collection_custom = pypsa.NetworkCollection(networks, index=custom_index)
    assert collection_custom.index.names == ["scenario"]
    assert collection_custom.networks.index.name == "scenario"
    assert collection_custom._index_names == ["scenario"]

    # test with multiindex
    multi_index = pd.MultiIndex.from_tuples(
        [("base", 2030), ("high_renewables", 2030)], names=["scenario", "year"]
    )
    collection_multi = pypsa.NetworkCollection(networks, index=multi_index)
    assert collection_multi.index.names == ["scenario", "year"]
    assert collection_multi._index_names == ["scenario", "year"]


def test_collection_not_implemented_members(network1, network2):
    """Test that not implemented members raise NotImplementedError."""
    collection = pypsa.NetworkCollection([network1, network2])
    with pytest.raises(NotImplementedError):
        collection.add("Generator", "gen", bus="bus1", p_nom=100)
    with pytest.raises(NotImplementedError):
        collection.remove("Generator", "gen")
    with pytest.raises(NotImplementedError):
        collection.set_snapshots([0, 1, 2])


def test_collection_carriers_property(network1, network2, network3):
    """Test the carriers property."""
    collection = pypsa.NetworkCollection([network1, network2, network3])
    expected_carriers = pd.concat(
        [network1.carriers, network2.carriers, network3.carriers]
    )
    expected_carriers = expected_carriers[
        ~expected_carriers.index.duplicated(keep="first")
    ].sort_index()

    pd.testing.assert_frame_equal(collection.carriers, expected_carriers)


def test_collection_carriers_property_empty():
    """Test the carriers property when networks have no carriers."""
    n1 = pypsa.Network()
    n2 = pypsa.Network()
    collection = pypsa.NetworkCollection([n1, n2])
    expected_carriers = pd.DataFrame(
        index=pd.Index([], name="Carrier"), columns=n1.carriers.columns, dtype=int
    )
    pd.testing.assert_frame_equal(
        collection.carriers,
        expected_carriers,
        check_dtype=False,
        check_index_type=False,
    )


def test_collection_getitem_slice(network1, network2, network3):
    """Test slicing the NetworkCollection object."""
    networks = [network1, network2, network3]
    collection = pypsa.NetworkCollection(networks)
    sliced_networks = collection[1:]
    assert isinstance(sliced_networks, pypsa.NetworkCollection)
    assert len(sliced_networks) == 2
    assert sliced_networks[1] == network2  # Original index 1
    assert sliced_networks[2] == network3  # Original index 2
    pd.testing.assert_index_equal(
        sliced_networks.networks.index, pd.RangeIndex(1, 3, name="network")
    )


def test_collection_iteration(network1, network2):
    """Test iterating over the NetworkCollection object."""
    networks = [network1, network2]
    collection = pypsa.NetworkCollection(networks)
    iterated_list = [n for n in collection]
    assert iterated_list == networks


def test_collection_static_data(network1, network2):
    """Test static data access."""
    collection = pypsa.NetworkCollection(
        [network1, network2], index=pd.Series(["net1", "net2"], name="scenario")
    )

    generators = collection.generators

    assert not generators.empty
    assert set(collection.index.names).issubset(generators.index.names)
    assert "Generator" in generators.index.names


def test_collection_statistics(network1, network2):
    """Test capex calculation per network."""
    collection = pypsa.NetworkCollection(
        [network1, network2], index=pd.Series(["net1", "net2"], name="scenario")
    )
    statistics = collection.statistics()
    assert not statistics.empty
    assert set(collection.index.names).issubset(statistics.index.names)

    capacity = collection.statistics.installed_capacity()

    assert not capacity.empty
    assert set(collection.index.names).issubset(capacity.index.names)
    assert "carrier" in capacity.index.names


def test_collection_statistics_nonexistent_method(network1):
    """Test calling a method that doesn't exist on the accessor."""
    collection = pypsa.NetworkCollection([network1])
    with pytest.raises(
        AttributeError,
        match="Only members as they are defined in any Network class can be accessed.",
    ):
        collection.nonexistent_method()
