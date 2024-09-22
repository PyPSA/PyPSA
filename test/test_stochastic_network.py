import linopy
import pytest
import xarray

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
    assert stochastic_network.snapshots.names == ["scenario", "timestep"]
    assert stochastic_network.snapshots.name == "snapshot"
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
    assert df.index.names == ["scenario", "timestep"]
    assert list(df.columns) == list(stochastic_network.df(c_name).index)


def test_get_bounds_pu_for_generators(stochastic_network):
    min_pu, max_pu = stochastic_network.get_bounds_pu(
        "Generator", stochastic_network.snapshots, explicit_scenario_dim=False
    )
    assert min_pu.index.names == ["scenario", "timestep"]
    assert max_pu.index.names == ["scenario", "timestep"]
    assert max_pu.index.name == "snapshot"
    generators = stochastic_network.df("Generator").index.tolist()
    assert list(min_pu.columns) == generators
    assert list(max_pu.columns) == generators


def test_bounds_pu_for_storage_units(stochastic_network):
    min_pu, max_pu = stochastic_network.get_bounds_pu(
        "StorageUnit", stochastic_network.snapshots, explicit_scenario_dim=False
    )
    assert min_pu.index.names == ["scenario", "timestep"]
    assert max_pu.index.names == ["scenario", "timestep"]
    storage_units = stochastic_network.df("StorageUnit").index.tolist()
    assert list(min_pu.columns) == storage_units
    assert list(max_pu.columns) == storage_units


def test_get_bounds_pu_explicit_scenario_dim(stochastic_network):
    min_pu, max_pu = stochastic_network.get_bounds_pu(
        "Generator", stochastic_network.snapshots, explicit_scenario_dim=True
    )
    assert isinstance(min_pu, xarray.DataArray)
    assert isinstance(max_pu, xarray.DataArray)
    assert set(min_pu.dims) == set(["scenario", "timestep", "Generator"])
    assert set(max_pu.dims) == set(["scenario", "timestep", "Generator"])


def test_model_variable_assignment(stochastic_network):
    n = stochastic_network
    n.model = linopy.Model()

    pypsa.optimization.variables.define_nominal_variables(n, "Generator", "p_nom")
    n.model.variables["Generator-p_nom"].dims == ["Generator"]

    pypsa.optimization.variables.define_operational_variables(
        n, n.snapshots, "Generator", "p"
    )
    n.model.variables["Generator-p"].dims == ["Generator", "scenario", "timestep"]


def test_model_constaint_assignment(stochastic_network):
    n = stochastic_network
    n.model = linopy.Model()

    pypsa.optimization.variables.define_nominal_variables(n, "Generator", "p_nom")
    pypsa.optimization.variables.define_operational_variables(
        n, n.snapshots, "Generator", "p"
    )

    pypsa.optimization.constraints.define_operational_constraints_for_extendables(
        n, n.snapshots, "Generator", "p", transmission_losses=False
    )
    n.model.constraints["Generator-ext-p-lower"].dims == [
        "Generator",
        "scenario",
        "timestep",
    ]
