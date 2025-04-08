from numpy.testing import assert_array_almost_equal as equal


def test_optimize(ac_dc_network, ac_dc_network_r):
    n = ac_dc_network
    n_r = ac_dc_network_r

    status, _ = n.optimize(snapshots=n.snapshots)

    assert status == "ok"

    equal(
        n.generators_t.p.loc[:, n.generators.index],
        n_r.generators_t.p.loc[:, n.generators.index],
        decimal=2,
    )

    equal(
        n.lines_t.p0.loc[:, n.lines.index],
        n_r.lines_t.p0.loc[:, n.lines.index],
        decimal=2,
    )

    equal(
        n.links_t.p0.loc[:, n.links.index],
        n_r.links_t.p0.loc[:, n.links.index],
        decimal=2,
    )
