# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa
from pypsa.constants import DEFAULT_TIMESTAMP

PANDAS_3 = int(pd.__version__.split(".")[0]) >= 3


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Test requires Python 3.12 or higher"
)
@pytest.mark.skipif(PANDAS_3, reason="pandapower does not yet support pandas 3")
@pytest.mark.parametrize("use_pandapower_index", [True, False])
@pytest.mark.parametrize("extra_line_data", [True, False])
def test_pandapower_custom_case(
    pandapower_custom_network, use_pandapower_index, extra_line_data
):
    import pandapower as pp

    net = pandapower_custom_network
    # because of phase angles, need to init with DC
    pp.runpp(net, calculate_voltage_angles=True, init="dc")
    n = pypsa.Network()
    n.import_from_pandapower_net(
        net, use_pandapower_index=use_pandapower_index, extra_line_data=extra_line_data
    )

    # seed PF with LPF solution because of phase angle jumps
    n.lpf()
    n.pf(use_seed=True)

    # use same index for everything
    net.res_bus.index = net.bus.name.values
    net.res_line.index = net.line.name.values

    # compare bus angles
    equal(
        n.c.buses.dynamic.v_ang.loc[DEFAULT_TIMESTAMP] * 180 / np.pi,
        net.res_bus.va_degree,
    )

    # compare bus voltage magnitudes
    equal(n.c.buses.dynamic.v_mag_pu.loc[DEFAULT_TIMESTAMP], net.res_bus.vm_pu)

    # pandapower folds shunt impedance power into the bus totals, while PyPSA
    # reports it on the shunt component (p positive if load, q positive if
    # generation), so add it back in the bus injection convention before comparing
    buses_i = n.c.buses.static.index
    shunt = n.c.shunt_impedances
    by_bus = shunt.static.bus
    p_per_shunt = shunt.dynamic.p.loc[DEFAULT_TIMESTAMP]
    q_per_shunt = shunt.dynamic.q.loc[DEFAULT_TIMESTAMP]
    shunt_p = p_per_shunt.groupby(by_bus).sum().reindex(buses_i, fill_value=0.0)
    shunt_q = q_per_shunt.groupby(by_bus).sum().reindex(buses_i, fill_value=0.0)
    bus_p = n.c.buses.dynamic.p.loc[DEFAULT_TIMESTAMP] - shunt_p
    bus_q = n.c.buses.dynamic.q.loc[DEFAULT_TIMESTAMP] + shunt_q

    # compare bus active power (NB: pandapower uses load signs)
    equal(bus_p, -net.res_bus.p_mw)

    # compare bus reactive power (NB: pandapower uses load signs)
    equal(bus_q, -net.res_bus.q_mvar)

    # compare branch flows
    equal(n.c.lines.dynamic.p0.loc[DEFAULT_TIMESTAMP], net.res_line.p_from_mw)
    equal(n.c.lines.dynamic.p1.loc[DEFAULT_TIMESTAMP], net.res_line.p_to_mw)
    equal(n.c.lines.dynamic.q0.loc[DEFAULT_TIMESTAMP], net.res_line.q_from_mvar)
    equal(n.c.lines.dynamic.q1.loc[DEFAULT_TIMESTAMP], net.res_line.q_to_mvar)

    equal(n.c.transformers.dynamic.p0.loc[DEFAULT_TIMESTAMP], net.res_trafo.p_hv_mw)
    equal(n.c.transformers.dynamic.p1.loc[DEFAULT_TIMESTAMP], net.res_trafo.p_lv_mw)
    equal(n.c.transformers.dynamic.q0.loc[DEFAULT_TIMESTAMP], net.res_trafo.q_hv_mvar)
    equal(n.c.transformers.dynamic.q1.loc[DEFAULT_TIMESTAMP], net.res_trafo.q_lv_mvar)
