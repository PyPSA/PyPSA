# -*- coding: utf-8 -*-
import numpy as np
import pandapower as pp
import pytest
from numpy.testing import assert_array_almost_equal as equal

import pypsa


@pytest.mark.parametrize("use_pandapower_index", [True, False])
@pytest.mark.parametrize("extra_line_data", [True, False])
def test_pandapower_custom_case(
    pandapower_custom_network, use_pandapower_index, extra_line_data
):
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
    equal(n.buses_t.v_ang.loc["now"] * 180 / np.pi, net.res_bus.va_degree)

    # compare bus voltage magnitudes
    equal(n.buses_t.v_mag_pu.loc["now"], net.res_bus.vm_pu)

    # compare bus active power (NB: pandapower uses load signs)
    equal(n.buses_t.p.loc["now"], -net.res_bus.p_mw)

    # compare bus active power (NB: pandapower uses load signs)
    equal(n.buses_t.q.loc["now"], -net.res_bus.q_mvar)

    # compare branch flows
    equal(n.lines_t.p0.loc["now"], net.res_line.p_from_mw)
    equal(n.lines_t.p1.loc["now"], net.res_line.p_to_mw)
    equal(n.lines_t.q0.loc["now"], net.res_line.q_from_mvar)
    equal(n.lines_t.q1.loc["now"], net.res_line.q_to_mvar)

    equal(n.transformers_t.p0.loc["now"], net.res_trafo.p_hv_mw)
    equal(n.transformers_t.p1.loc["now"], net.res_trafo.p_lv_mw)
    equal(n.transformers_t.q0.loc["now"], net.res_trafo.q_hv_mvar)
    equal(n.transformers_t.q1.loc["now"], net.res_trafo.q_lv_mvar)
