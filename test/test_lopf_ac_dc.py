# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from numpy.testing import assert_array_almost_equal as equal


def test_optimize(ac_dc_network, ac_dc_network_r):
    n = ac_dc_network
    n_r = ac_dc_network_r

    status, _ = n.optimize(snapshots=n.snapshots)

    assert status == "ok"

    equal(
        n.c.generators.dynamic.p.loc[:, n.c.generators.static.index],
        n_r.c.generators.dynamic.p.loc[:, n.c.generators.static.index],
        decimal=2,
    )

    equal(
        n.c.lines.dynamic.p0.loc[:, n.c.lines.static.index],
        n_r.c.lines.dynamic.p0.loc[:, n.c.lines.static.index],
        decimal=2,
    )

    equal(
        n.c.links.dynamic.p0.loc[:, n.c.links.static.index],
        n_r.c.links.dynamic.p0.loc[:, n.c.links.static.index],
        decimal=2,
    )
