import pytest

from pypsa.components.common import as_components


def test_as_components(ac_dc_network):
    n = ac_dc_network

    assert as_components(n, "Generator") == n.components.generators
    assert as_components(n, "generators") == n.components.generators
    assert as_components(n, n.components.generators) == n.components.generators
    with pytest.raises(TypeError):
        assert as_components(n, 10)
