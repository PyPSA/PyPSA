import pytest

import pypsa


@pytest.fixture
def stochastic_network(ac_dc_network):
    scenarios = {"scenario_1": 0.5, "scenario_2": 0.5}
    return pypsa.StochasticNetwork(ac_dc_network, scenarios)


def test_initilization_with_auto_normalization(ac_dc_network):
    scenarios = {"scenario_1": 0.6, "scenario_2": 0.4, "scenario_3": 0.5}
    stochastic_network = pypsa.StochasticNetwork(ac_dc_network, scenarios)
    assert set(stochastic_network.scenarios.keys()) == {
        "scenario_1",
        "scenario_2",
        "scenario_3",
    }
    assert sum(stochastic_network.scenarios.values()) == 1


def test_stochastic_network_initialization(stochastic_network):
    assert isinstance(stochastic_network, pypsa.StochasticNetwork)
    assert set(stochastic_network.scenarios.keys()) == {"scenario_1", "scenario_2"}
    assert stochastic_network.snapshots.names == ["scenario", "snapshot"]
    assert len(stochastic_network.snapshots) == 20  # 2 scenarios * 10 snapshots


def test_stochastic_network_static_components(stochastic_network):
    assert "scenario" in stochastic_network.generators.columns.names
    assert "scenario_1" in stochastic_network.generators.columns.levels[1]
    assert "scenario_2" in stochastic_network.generators.columns.levels[1]


def test_stochastic_network_time_dependent_data(stochastic_network):
    assert "scenario" in stochastic_network.generators_t.p_max_pu.index.names
    assert (
        len(stochastic_network.generators_t.p_max_pu) == 20
    )  # 2 scenarios * 10 snapshots


def test_df(stochastic_network):
    df = stochastic_network.df("Generator")
    assert df.index.name == "Generator"
    assert df.columns.names == ["attr", "scenario"]


def test_get_switchable_as_dense(stochastic_network):
    c_name = "Generator"
    df = stochastic_network.get_switchable_as_dense(c_name, "p_max_pu")
    assert df.index.names == ["scenario", "snapshot"]
    assert list(df.columns) == list(stochastic_network.df(c_name).index)


def test_get_bounds_pu_for_generators(stochastic_network):
    min_pu, max_pu = stochastic_network.get_bounds_pu(
        "Generator", stochastic_network.snapshots
    )
    assert min_pu.index.names == ["scenario", "snapshot"]
    assert max_pu.index.names == ["scenario", "snapshot"]
    generators = stochastic_network.df("Generator").index.tolist()
    assert list(min_pu.columns) == generators
    assert list(max_pu.columns) == generators


def test_bounds_pu_for_storage_units(stochastic_network):
    min_pu, max_pu = stochastic_network.get_bounds_pu(
        "StorageUnit", stochastic_network.snapshots
    )
    assert min_pu.index.names == ["scenario", "snapshot"]
    assert max_pu.index.names == ["scenario", "snapshot"]
    storage_units = stochastic_network.df("StorageUnit").index.tolist()
    assert list(min_pu.columns) == storage_units
    assert list(max_pu.columns) == storage_units
