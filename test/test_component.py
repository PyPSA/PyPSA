import pandas as pd
import pytest

from pypsa import Network
from pypsa.components import Component


@pytest.fixture
def sample_network():
    # Create a sample network object
    n = Network()
    return n


@pytest.fixture
def sample_component(sample_network):
    # Create a sample component object
    data = {"active": [True, False, True], "other_attr": [1, 2, 3]}
    static = pd.DataFrame(data, index=["asset1", "asset2", "asset3"])
    dynamic = {"time_series": pd.DataFrame({"value": [0.1, 0.2, 0.3]})}
    attrs = pd.DataFrame({"attr1": ["metadata1"], "attr2": ["metadata2"]})

    component = Component(
        name="Generator",
        list_name="generators",
        investment_periods=sample_network.investment_periods,
        attrs=attrs,
        static=static,
        dynamic=dynamic,
        ind=None,
    )
    return component


def test_component_initialization(sample_component):
    component = sample_component
    assert component.name == "Generator"
    assert component.list_name == "generators"
    assert "attr1" in component.attrs
    assert component.static.shape == (3, 2)
    assert "time_series" in component.dynamic


def test_component_repr(sample_component):
    component = sample_component
    repr_str = repr(component)
    assert "Component(name='Generator'" in repr_str
    assert "list_name='generators'" in repr_str
    assert "static=DataFrame(shape=(3, 2))" in repr_str


def test_active_assets(sample_component):
    component = sample_component
    active_assets = component.static.query("active").index
    assert len(active_assets) == 2
    assert "asset1" in active_assets
    assert "asset3" in active_assets


def test_active_in_investment_period(sample_component):
    component = sample_component
    active_assets = component.get_active_assets()
    assert active_assets.sum() == 2
    assert active_assets["asset1"]
    assert not active_assets["asset2"]
    assert active_assets["asset3"]
