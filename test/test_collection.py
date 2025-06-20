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
    # Give networks unique names
    network1.name = "net1"
    network2.name = "net2"
    networks = [network1, network2]
    collection = pypsa.NetworkCollection(networks)
    assert len(collection) == 2
    assert isinstance(collection.networks, pd.Series)
    assert collection.networks.index.equals(pd.Index(["net1", "net2"], name="network"))
    assert collection["net1"] == network1
    assert collection["net2"] == network2


def test_collection_init_list_with_index(network1, network2):
    """Test initialization with a list and custom index."""
    networks = [network1, network2]
    custom_index = pd.Index(["net_A", "net_B"], name="scenario")
    collection = pypsa.NetworkCollection(networks, index=custom_index)
    assert len(collection) == 2
    assert collection.networks.index.equals(custom_index)
    assert collection["net_A"] == network1
    assert collection["net_B"] == network2


def test_collection_initi_list_with_multiindex(network1, network2):
    """Test initialization with a list and MultiIndex."""
    networks = [network1, network2]
    multi_index = pd.MultiIndex.from_tuples(
        [("base", 2030), ("high_renewables", 2030)], names=["scenario", "year"]
    )
    collection = pypsa.NetworkCollection(networks, index=multi_index)
    assert len(collection) == 2
    assert collection.networks.index.equals(multi_index)
    assert "network" not in collection.networks.index.names
    assert collection[("base", 2030)] == network1
    assert collection[("high_renewables", 2030)] == network2


def test_collection_init_series(network1, network2):
    """Test initialization with a pandas Series."""
    networks_series = pd.Series([network1, network2], index=["net_A", "net_B"])
    collection = pypsa.NetworkCollection(networks_series)
    assert len(collection) == 2
    assert isinstance(collection.networks, pd.Series)
    assert collection.networks.index.equals(
        pd.Index(["net_A", "net_B"], name="network")
    )
    assert collection["net_A"] == network1
    assert collection["net_B"] == network2


def test_collection_init_series_with_multiindex(network1, network2):
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
        pypsa.NetworkCollection([pypsa.Network(), 123])
    with pytest.raises(TypeError):
        pypsa.NetworkCollection(pd.Series([pypsa.Network(), 5]))
    with pytest.raises(TypeError):
        pypsa.NetworkCollection("single_string")


def test_collection_init_duplicate_names():
    """Test that duplicate network names raise an error."""
    # Create networks with duplicate names
    n1 = pypsa.Network(name="base")
    n2 = pypsa.Network(name="base")
    n3 = pypsa.Network(name="scenario")

    with pytest.raises(ValueError, match="Duplicate network names found: \\['base'\\]"):
        pypsa.NetworkCollection([n1, n2, n3])

    # Test with default names (empty name)
    n1 = pypsa.Network()
    n2 = pypsa.Network()

    with pytest.raises(
        ValueError, match="Duplicate network names found: \\['Unnamed Network'\\]"
    ):
        pypsa.NetworkCollection([n1, n2])

    # Should work with custom index even if names are duplicated
    collection = pypsa.NetworkCollection([n1, n2], index=["net_A", "net_B"])
    assert len(collection) == 2


def test_collection_init_index_mismatch(network1, network2):
    """Test initialization with mismatched index length."""
    with pytest.raises(ValueError):
        pypsa.NetworkCollection([network1, network2], index=pd.Index(["A"]))


def test_collection_index_names(network1, network2):
    """Test the index names of the NetworkCollection object."""
    # Give networks unique names
    network1.name = "net1"
    network2.name = "net2"
    networks = [network1, network2]
    collection = pypsa.NetworkCollection(networks)
    assert collection.index.names == ["network"]
    assert collection.networks.index.name == "network"
    assert collection._index_names == ["network"]

    # Test custom single index overwrite
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
    # Give networks unique names
    network1.name = "net1"
    network2.name = "net2"
    collection = pypsa.NetworkCollection([network1, network2])
    with pytest.raises(NotImplementedError):
        collection.add("Generator", "gen", bus="bus1", p_nom=100)
    with pytest.raises(NotImplementedError):
        collection.remove("Generator", "gen")
    with pytest.raises(NotImplementedError):
        collection.set_snapshots([0, 1, 2])


def test_collection_carriers_property(network1, network2, network3):
    """Test the carriers property."""
    # Give networks unique names
    network1.name = "net1"
    network2.name = "net2"
    network3.name = "net3"
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
    n1 = pypsa.Network(name="n1")
    n2 = pypsa.Network(name="n2")
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
    # Give networks unique names
    network1.name = "net1"
    network2.name = "net2"
    network3.name = "net3"
    networks = [network1, network2, network3]
    collection = pypsa.NetworkCollection(networks)
    sliced_networks = collection[1:]
    assert isinstance(sliced_networks, pypsa.NetworkCollection)
    assert len(sliced_networks) == 2
    assert sliced_networks["net2"] == network2  # Original index 1
    assert sliced_networks["net3"] == network3  # Original index 2
    pd.testing.assert_index_equal(
        sliced_networks.networks.index,
        pd.Index(["net2", "net3"], name="network"),
    )


def test_collection_iteration(network1, network2):
    """Test iterating over the NetworkCollection object."""
    # Give networks unique names
    network1.name = "net1"
    network2.name = "net2"
    networks = [network1, network2]
    collection = pypsa.NetworkCollection(networks)
    iterated_list = list(collection)
    assert iterated_list == networks


def test_collection_static_data(network1, network2):
    """Test static data access."""
    # Give networks unique names
    network1.name = "net1"
    network2.name = "net2"
    collection = pypsa.NetworkCollection(
        [network1, network2], index=pd.Index(["net1", "net2"], name="scenario")
    )

    assert collection.generators.loc["net1"].equals(network1.generators)
    assert "Generator" in collection.generators.index.names
    assert "scenario" in collection.generators.index.names


def test_collection_dynamic_data(network1, network2):
    """Test dynamic data access."""
    # Give networks unique names
    network1.name = "net1"
    network2.name = "net2"
    network1.snapshots = [1, 2]
    network2.snapshots = [1, 2]
    network1.add("Generator", "dyn-gen", bus="bus1", p_min_pu=[0, 1])
    network2.add("Generator", "dyn-gen", bus="bus1", p_min_pu=[0, 1])
    collection = pypsa.NetworkCollection(
        [network1, network2], index=pd.Index(["net1", "net2"], name="scenario")
    )

    assert collection.generators_t.p_min_pu["net1"].equals(
        network1.generators_t.p_min_pu
    )
    assert "scenario" in collection.generators_t.p_min_pu.columns.names


def test_collection_statistics_nonexistent_method(network1):
    """Test calling a method that doesn't exist on the accessor."""
    # Give network a unique name
    network1.name = "net1"
    collection = pypsa.NetworkCollection([network1])
    with pytest.raises(
        AttributeError,
        match="Only members as they are defined in any Network class can be accessed.",
    ):
        collection.nonexistent_method()


def test_collection_repr(network1, network2, network3):
    """Test the string representation of NetworkCollection."""
    # Test with networks having unique names
    network1.name = "net1"
    network2.name = "net2"
    network3.name = "net3"

    collection = pypsa.NetworkCollection([network1, network2])
    repr_str = repr(collection)
    assert "Networks: 2" in repr_str
    assert "Index name: 'network'" in repr_str
    assert "Entries: ['net1', 'net2']" in repr_str

    # Test with custom index
    custom_index = pd.Index(["net_A", "net_B", "net_C"], name="scenario")
    collection = pypsa.NetworkCollection(
        [network1, network2, network3], index=custom_index
    )
    repr_str = repr(collection)
    assert "Networks: 3" in repr_str
    assert "Index name: 'scenario'" in repr_str
    assert "Entries: ['net_A', 'net_B', 'net_C']" in repr_str

    # Test with MultiIndex
    multi_index = pd.MultiIndex.from_tuples(
        [("base", 2030), ("high_renewables", 2030)], names=["scenario", "year"]
    )
    collection = pypsa.NetworkCollection([network1, network2], index=multi_index)
    repr_str = repr(collection)
    assert "Networks: 2" in repr_str
    assert "MultiIndex with 2 levels: ['scenario', 'year']" in repr_str
    assert "First 2 entries: [('base', 2030), ('high_renewables', 2030)]" in repr_str

    # Test with many networks (to check truncation)
    many_networks = [pypsa.Network(name=f"net_{i}") for i in range(10)]
    collection = pypsa.NetworkCollection(many_networks)
    repr_str = repr(collection)
    assert "Networks: 10" in repr_str
    assert "... and 5 more" in repr_str


def test_collection_init_with_strings():
    """Test initialization with string paths."""
    # Use example networks from the examples directory
    example_path1 = "examples/networks/ac-dc-meshed/ac-dc-meshed.nc"
    example_path2 = "examples/networks/scigrid-de/scigrid-de.nc"

    # Test with list of strings
    collection = pypsa.NetworkCollection([example_path1, example_path2])
    assert len(collection) == 2
    assert all(isinstance(n, pypsa.Network) for n in collection.networks)

    # Test with pandas Series containing strings
    networks_series = pd.Series([example_path1, example_path2], index=["net1", "net2"])
    collection_series = pypsa.NetworkCollection(networks_series)
    assert len(collection_series) == 2
    assert all(isinstance(n, pypsa.Network) for n in collection_series.networks)


def test_collection_init_mixed_networks_and_strings(network1):
    """Test initialization with mixed Network objects and strings."""
    network1.name = "manual_net"
    example_path = "examples/networks/ac-dc-meshed/ac-dc-meshed.nc"

    # Test with mixed list
    collection = pypsa.NetworkCollection([network1, example_path])
    assert len(collection) == 2
    assert all(isinstance(n, pypsa.Network) for n in collection.networks)
    assert collection["manual_net"] == network1

    # Test with custom index
    custom_index = pd.Index(["net_A", "net_B"], name="scenario")
    collection_custom = pypsa.NetworkCollection(
        [network1, example_path], index=custom_index
    )
    assert len(collection_custom) == 2
    assert collection_custom["net_A"] == network1


def test_collection_init_empty_string():
    """Test initialization with empty string (creates empty network)."""
    collection = pypsa.NetworkCollection([""])
    assert len(collection) == 1
    assert isinstance(collection.networks.iloc[0], pypsa.Network)
