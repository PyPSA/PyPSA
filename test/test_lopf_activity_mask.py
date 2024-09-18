def test_optimize(ac_dc_network):
    n = ac_dc_network

    inactive_links = ["DC link"]

    n.links.loc[inactive_links, "active"] = False

    status, _ = n.optimize(snapshots=n.snapshots)

    assert status == "ok"

    assert n.links_t.p0.loc[:, inactive_links].eq(0).all().all()


def test_optimize_with_power_flow(scipy_network):
    n = scipy_network.copy()

    switchable_lines = n.lines.index[100]

    res = n.optimize.optimize_and_run_non_linear_powerflow(snapshots=n.snapshots[:1])
    assert res["status"] == "ok"
    assert res["converged"].all().all()
    assert not n.lines_t.p0.loc[:, switchable_lines].eq(0).all().all()

    n = scipy_network.copy()
    n.lines.loc[switchable_lines, "active"] = False

    res = n.optimize.optimize_and_run_non_linear_powerflow(snapshots=n.snapshots[:1])
    assert res["status"] == "ok"
    assert res["converged"].all().all()
    assert n.lines_t.p0.loc[:, switchable_lines].eq(0).all().all()
