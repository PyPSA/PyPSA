# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest

from pypsa.components.common import as_components


def test_as_components(ac_dc_network):
    n = ac_dc_network

    assert as_components(n, "Generator") == n.c.generators
    assert as_components(n, "generators") == n.c.generators
    assert as_components(n, n.c.generators) == n.c.generators
    with pytest.raises(TypeError):
        assert as_components(n, 10)
