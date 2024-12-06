import pytest

import pypsa

component_names = [
    "Generator",
    "Line",
    "Link",
    "StorageUnit",
    "Load",
    "Transformer",
]


@pytest.mark.parametrize("component_name", component_names)
def test_alias_properties(ac_dc_network, component_name):
    n = ac_dc_network
    ct = pypsa.components.types.get(component_name)
    c = n.components[component_name]

    assert c.name == ct.name
    assert c.list_name == ct.list_name
    assert c.description == ct.description
    assert c.category == ct.category
    assert c.type == ct.category
    assert c.defaults.equals(ct.defaults)
    assert c.attrs.equals(ct.defaults)
    if c.standard_types is not None:
        assert c.standard_types.equals(ct.standard_types)

    assert c.df is c.static
    assert c.pnl is c.dynamic


def test_network_attachments(ac_dc_network):
    n = ac_dc_network.copy()

    c = n.components.generators

    assert c.n is n
    assert c.n_save is n
    assert c.attached is True
    c.n = None
    c.attached is False
    with pytest.raises(AttributeError):
        c.n_save


def test_components_repr(ac_dc_network):
    n = ac_dc_network
    c = n.components.generators

    assert repr(c).startswith("PyPSA 'Generator' Components")
