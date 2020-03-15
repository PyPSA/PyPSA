import pypsa

import numpy as np

pandapower_compatible = True
try:
    import pandapower as pp
except ImportError:
    pandapower_compatible = False

import six

import pytest


@pytest.mark.skipif(
    six.PY2 and not pandapower_compatible,
    reason="Pandapower 2.0.0 dropped support for Python 2.",
)
def test_pandapower_case():

    # more complicated examples like
    # net = pandapower.networks.example_simple()
    # can be used once the import of e.g. switches is perfected

    # create empty net
    net = pp.create_empty_network()

    # create buses
    b1 = pp.create_bus(net, vn_kv=20.0, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")

    # create bus elements
    pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=b3, p_mw=0.1, q_mvar=0.05, name="Load")

    # create branch elements
    pp.create_transformer(
        net, hv_bus=b1, lv_bus=b2, std_type="0.4 MVA 20/0.4 kV", name="Trafo"
    )
    pp.create_line(
        net, from_bus=b2, to_bus=b3, length_km=0.1, name="Line", std_type="NAYY 4x50 SE"
    )

    # because of phase angles, need to init with DC
    pp.runpp(net, calculate_voltage_angles=True, init="dc")

    n = pypsa.Network()

    n.import_from_pandapower_net(net)

    # seed PF with LPF solution because of phase angle jumps
    n.lpf()
    n.pf(use_seed=True)

    # use same index for everything
    net.res_bus.index = net.bus.name.values
    net.res_line.index = net.line.name.values

    # compare bus angles
    np.testing.assert_array_almost_equal(
        n.buses_t.v_ang.loc["now"] * 180 / np.pi, net.res_bus.va_degree
    )

    # compare bus voltage magnitudes
    np.testing.assert_array_almost_equal(
        n.buses_t.v_mag_pu.loc["now"], net.res_bus.vm_pu
    )

    # compare bus active power (NB: pandapower uses load signs)
    np.testing.assert_array_almost_equal(n.buses_t.p.loc["now"], -net.res_bus.p_mw)

    # compare bus active power (NB: pandapower uses load signs)
    np.testing.assert_array_almost_equal(n.buses_t.q.loc["now"], -net.res_bus.q_mvar)

    # compare branch flows
    np.testing.assert_array_almost_equal(
        n.lines_t.p0.loc["now"], net.res_line.p_from_mw
    )
    np.testing.assert_array_almost_equal(n.lines_t.p1.loc["now"], net.res_line.p_to_mw)
    np.testing.assert_array_almost_equal(
        n.lines_t.q0.loc["now"], net.res_line.q_from_mvar
    )
    np.testing.assert_array_almost_equal(
        n.lines_t.q1.loc["now"], net.res_line.q_to_mvar
    )

    np.testing.assert_array_almost_equal(
        n.transformers_t.p0.loc["now"], net.res_trafo.p_hv_mw
    )
    np.testing.assert_array_almost_equal(
        n.transformers_t.p1.loc["now"], net.res_trafo.p_lv_mw
    )
    np.testing.assert_array_almost_equal(
        n.transformers_t.q0.loc["now"], net.res_trafo.q_hv_mvar
    )
    np.testing.assert_array_almost_equal(
        n.transformers_t.q1.loc["now"], net.res_trafo.q_lv_mvar
    )
