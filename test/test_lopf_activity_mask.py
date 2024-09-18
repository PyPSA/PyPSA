import numpy as np
import pytest


def test_optimize(ac_dc_network):
    n = ac_dc_network

    inactive_links = ["DC link"]

    n.links.loc[inactive_links, "active"] = False

    status, _ = n.optimize(snapshots=n.snapshots)

    assert status == "ok"

    assert n.links_t.p0.loc[:, inactive_links].eq(0).all().all()


def test_optimize_with_power_flow(scipy_network):
    """
    Test the functionality of the 'active' attribute in PyPSA components.

    This test function verifies that the 'active' attribute of network components
    (specifically lines in this case) is correctly respected during optimization,
    non-linear power flow, and linear power flow calculations.

    The test performs the following checks:
    1. Optimization respects the 'active' status of lines.
    2. Non-linear power flow calculations adhere to the 'active' attribute.
    3. Linear power flow (LPF) results are consistent with the 'active' status.

    The test is performed for both active (True) and inactive (False) scenarios
    to ensure proper behavior in both cases.
    """

    @pytest.mark.parametrize("line_active", [True, False])
    def test_scenario(line_active):
        n = scipy_network.copy()
        switchable_lines = n.lines.index[100]
        n.lines.loc[switchable_lines, "active"] = line_active

        # Test optimization and non-linear power flow
        res = n.optimize.optimize_and_run_non_linear_powerflow(
            snapshots=n.snapshots[:1]
        )

        assert res["status"] == "ok", f"Optimization failed with status {res['status']}"
        assert res["converged"].all().all(), "Non-linear power flow did not converge"

        expected_flow = (
            not np.isclose(n.lines_t.p0.loc[:, switchable_lines], 0).all().all()
        )
        msg = f"'active' attribute not respected in optimization/non-linear power flow: expected {'non-zero' if line_active else 'zero'} flow"
        assert expected_flow == line_active, msg

        # Test linear power flow
        n.lpf()
        expected_flow = (
            not np.isclose(n.lines_t.p0.loc[:, switchable_lines], 0).all().all()
        )
        msg = f"'active' attribute not respected in linear power flow: expected {'non-zero' if line_active else 'zero'} flow"
        assert expected_flow == line_active, msg

        msg = "Power balance not maintained"
        assert np.isclose(n.buses_t.p.sum().sum(), 0, atol=1e-5), msg

    test_scenario(True)
    test_scenario(False)
