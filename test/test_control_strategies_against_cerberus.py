"""Test power flow (pf) results of inverter control strategies namely; fixed
power factor (fixed_cosphi), power factors as a fucntion of active power
(cosphi_p) and reactive power as a function of voltage (q_v or Q(U)) against
Cerberus. Cerberus is a power flow software which has the above mentioned control
strategies as features. In an example of four generating units (PV or wind..)
first, pf with out contro is run and the results are converted to PyPSA csv files
which are saved in test/networks/results_no_c folder, then controllers are
applied one by one and pf is run for each case and the results are saved in
test/networks/... in seperate folder for each controller respectively, including
their dsn files. Dsn file is the Cerberus file format. And the test are as follow.

In test/networks there are four folders which contain cerberus pf results:

    results_no_c = normal pf with out controller
    results_with_cosphi_p = pf with power_factor as a function of active power
    results_with_fixed_pf = pf with fixed power factor controller
    results_with_q_v = pf with Q(U) controller
"""
# importing important libraries
import os
import pypsa
import numpy as np


def assert_pf_results_are_almost_equal(n1, n2):
    # compare bus voltage magnitudes
    np.testing.assert_array_almost_equal(
            n1.buses_t.v_mag_pu, n2.buses_t.v_mag_pu, verbose=True, decimal=5,
            err_msg='bus v_mag_pu values violate')

    # compare bus voltage angles in degrees
    np.testing.assert_array_almost_equal(
            n1.buses_t.v_ang, n2.buses_t.v_ang,
            verbose=True, decimal=5, err_msg='bus voltage angle violate')

    # Compare bus actibe power
    np.testing.assert_array_almost_equal(
            n1.buses_t.p.sort_index(axis=1), n2.buses_t.p.sort_index(axis=1),
            verbose=True, decimal=5, err_msg='buses p values violates')

    # Compare bus reactive power
    np.testing.assert_array_almost_equal(
            n1.buses_t.q.sort_index(axis=1), n2.buses_t.q.sort_index(axis=1),
            verbose=True, decimal=5,
            err_msg='buses q values violates')

    # Compare pf branch flows
    np.testing.assert_array_almost_equal(
            n1.generators_t.q, n2.generators_t.q, verbose=True, decimal=5,
            err_msg='generators q values violate')

    np.testing.assert_array_almost_equal(
            n1.generators_t.p, n2.generators_t.p, verbose=True, decimal=5,
            err_msg='generators p values violate')

    np.testing.assert_array_almost_equal(
            n1.loads_t.q, n2.loads_t.q, verbose=True, decimal=5,
            err_msg='loads q values violate')

    np.testing.assert_array_almost_equal(
            n1.loads_t.p, n2.loads_t.p, verbose=True, decimal=5,
            err_msg='loads p values violate')

    np.testing.assert_array_almost_equal(
        n1.lines_t.p1, n2.lines_t.p1, verbose=True, decimal=5,
        err_msg='lines p1 values violate')

    np.testing.assert_array_almost_equal(
        n1.lines_t.p0, n2.lines_t.p0, verbose=True, decimal=5,
        err_msg='lines p0 values violate')

    np.testing.assert_array_almost_equal(
        n1.lines_t.q1, n2.lines_t.q1, verbose=True, decimal=5,
        err_msg='lines q1 values violates')

    np.testing.assert_array_almost_equal(
        n1.lines_t.q0, n2.lines_t.q0, verbose=True, decimal=5,
        err_msg='lines q0 values violate')

    np.testing.assert_array_almost_equal(
        n1.transformers_t.p1, n2.transformers_t.p1, verbose=True, decimal=5,
        err_msg='transformers p1 values violate')

    np.testing.assert_array_almost_equal(
        n1.transformers_t.p0, n2.transformers_t.p0, verbose=True, decimal=5,
        err_msg='transformers p0 values violate')

    np.testing.assert_array_almost_equal(
        n1.transformers_t.q1, n2.transformers_t.q1, verbose=True, decimal=5,
        err_msg='transformers q1 values violate')

    np.testing.assert_array_almost_equal(
        n1.transformers_t.q0, n2.transformers_t.q0, verbose=True, decimal=5,
        err_msg='transformers q0 values violate')


def test_PyPSA_pf_results_with_controllers_against_CERBERUS_network():
    """
    test/network contains network results with control and without control from
    cerberus software. This function loads the network with no control applied
    (results_no_c), runs the pf on it in PyPSA and compare the two results.
    The pf results from Cerberus for each controller also exist in  test/network,
    therefore each controller is applied on the network with no control case
    of cerberus network and run PyPSA pf on it and then compared with the cerberus
    results of that controller (for each controller separately).
    """
    # pf results of pf without controller in Cerberus
    cerberus_path = os.path.join(os.path.dirname(__file__), "..", "test",
                                    "networks","results_no_c", "citygrid")

    n1 = pypsa.Network(cerberus_path, ignore_standard_types=True)

    # copy the network and run pf in PyPSA without controllers as n2
    n2 = n1.copy()
    n2.pf()

    # compare the two pf results (n1-cerberus and n2-PyPSA)
    assert_pf_results_are_almost_equal(n1, n2)

    # Fixed power factor controller pf results in Cerberus as n3
    cerberus_path_fixed_cosphi_control = os.path.join(os.path.dirname(__file__), "..", "test",
                                    "networks","results_with_fixed_pf", "citygrid")
    n3 = pypsa.Network(cerberus_path_fixed_cosphi_control, ignore_standard_types=True)

    # copy n1 network, set controller parameters same as Cerberus and run pf in PyPSA as n4
    n4 = n1.copy()
    n4.generators.control_strategy = "fixed_cosphi"
    # in cerberus the option of choosing angle is provided - 20° is chosen
    n4.generators.power_factor = np.cos(20*np.pi/180)  # finding power_factor
    n4.pf(inverter_control=True)

    # compare the two pf results (n3-cerberus and n4-PyPSA)
    assert_pf_results_are_almost_equal(n3, n4)

    # cosphi_p controller pf results in Cerberus as n5
    cerberus_path_cosphi_p_control = os.path.join(os.path.dirname(__file__), "..", "test",
                                    "networks","results_with_cosphi_p", "citygrid")
    n5 = pypsa.Network(cerberus_path_cosphi_p_control, ignore_standard_types=True)

#    # copy n1 network, set controller parameters same as Cerberus and run pf in PyPSA as n6
    n6 = n1.copy()
    n6.generators.control_strategy = "cosphi_p"
    n6.generators.power_factor_min = np.cos(26*np.pi/180)  # in cerberus it is translated as angle
    n6.generators.s_nom = 0.05
    n6.generators.p_ref = 0.03
    n6.pf(inverter_control=True)

    # compare the two pf results (n5-cerberus and n6-PyPSA)
    assert_pf_results_are_almost_equal(n5, n6)

    # Q(U) or q_v controller pf results in Cerberus as n7
    cerberus_path_q_v_control = os.path.join(os.path.dirname(__file__), "..", "test",
                                    "networks","results_with_q_v", "citygrid")
    n7 = pypsa.Network(cerberus_path_q_v_control, ignore_standard_types=True)

    # copy n1 network, set controller parameters same as Cerberus and run pf in PyPSA as n8
    n8 = n1.copy()
    n8.generators.control_strategy = "q_v"
    n8.generators.s_nom = 0.05
    # in Cerberus after setting s_nom and p_set,  q_max and curve droop is required
    # for Q(U), control. In cerberus s_nom = 0.05, p_set = 0.03 and q_max = 40
    # is chosen. power_factor = np.cos(np.arctan(0.04/0.03)) = 0.6
    n8.generators.power_factor = 0.6
    n8.pf(x_tol_outer=1e-6, inverter_control=True)

    # compare the two pf results (n7-cerberus and n8-PyPSA)
    assert_pf_results_are_almost_equal(n7, n8)

if __name__ == "__main__":
    test_PyPSA_pf_results_with_controllers_against_CERBERUS_network()
