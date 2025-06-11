import pandas as pd
import pytest

from pypsa import Components, Network
from pypsa.components.legacy import Component
from pypsa.components.types import get as get_component_type


def test_components_non_implemented():
    """Test that the components module raises an ImportError if imported directly."""
    ct = get_component_type("Generator")
    with pytest.raises(NotImplementedError):
        Components(ctype=ct, names=["Generator"])
    n = Network()
    with pytest.raises(NotImplementedError):
        Components(ctype=ct, n=n)


@pytest.fixture
def legacy_component():
    n = Network()
    # Create a sample component object
    data = {"active": [True, False, True], "other_attr": [1, 2, 3]}
    static = pd.DataFrame(data, index=["asset1", "asset2", "asset3"])
    dynamic = {"time_series": pd.DataFrame({"value": [0.1, 0.2, 0.3]})}

    component = Component(
        name="Generator",
        n=n,
        static=static,
        dynamic=dynamic,
    )
    return component


def test_deprecated_arguments():
    with pytest.warns(DeprecationWarning):
        Component(name="Generator", list_name="x", attrs=pd.DataFrame())


def test_component_initialization(legacy_component):
    component = legacy_component
    assert component.name == "Generator"
    assert component.list_name == "generators"
    assert component.static.shape == (3, 2)
    assert "time_series" in component.dynamic


def test_active_assets(legacy_component):
    component = legacy_component
    active_assets = component.static.query("active").index
    assert len(active_assets) == 2
    assert "asset1" in active_assets
    assert "asset3" in active_assets


def test_active_in_investment_period(legacy_component):
    component = legacy_component
    active_assets = component.get_active_assets()
    assert active_assets.sum() == 2
    assert active_assets["asset1"]
    assert not active_assets["asset2"]
    assert active_assets["asset3"]


def test_imports():
    with pytest.raises(ImportError):
        from pypsa.components import Network  # noqa: F401
    with pytest.raises(ImportError):
        from pypsa.components import SubNetwork  # noqa: F401
