# -*- coding: utf-8 -*-
import pytest

import pypsa

TOLERANCE = 1e-2


@pytest.fixture
def prepared_network(ac_dc_network):
    n = ac_dc_network
    n.optimize.create_model()
    n.lines["carrier"] = n.lines.bus0.map(n.buses.carrier)
    return n


def test_statistics_capex_by_carrier(ac_dc_network):
    # n = ac_dc_network
    n = pypsa.examples.ac_dc_meshed()
    n.optimize.statistic_expressions.capex()
