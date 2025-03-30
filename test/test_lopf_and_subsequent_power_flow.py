def test_optimize_with_power_flow(scipy_network):
    n = scipy_network

    res = n.optimize.optimize_and_run_non_linear_powerflow(snapshots=n.snapshots[:4])

    assert res["status"] == "ok"
    assert res["converged"].all().all()
