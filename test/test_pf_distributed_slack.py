# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from numpy.testing import assert_array_almost_equal as equal


def normed(s):
    return s / s.sum()


def test_pf_distributed_slack(scipy_network):
    n = scipy_network
    n.set_snapshots(n.snapshots[:2])

    # There are some infeasibilities without line extensions
    n.c.lines.static.s_max_pu = 0.7
    n.c.lines.static.loc[["316", "527", "602"], "s_nom"] = 1715
    n.c.storage_units.static.state_of_charge_initial = 0.0

    n.optimize(n.snapshots)

    # For the PF, set the P to the optimised P
    n.c.generators.dynamic.p_set = n.c.generators.dynamic.p
    n.c.storage_units.dynamic.p_set = n.c.storage_units.dynamic.p

    # set all buses to PV, since we don't know what Q set points are
    n.c.generators.static.control = "PV"

    # Need some PQ buses so that Jacobian doesn't break
    f = n.c.generators.static[n.c.generators.static.bus == "492"]
    n.c.generators.static.loc[f.index, "control"] = "PQ"
    # by dispatch
    n.pf(distribute_slack=True, slack_weights="p_set")

    equal(
        n.c.generators.dynamic.p_set.apply(normed, axis=1),
        (n.c.generators.dynamic.p - n.c.generators.dynamic.p_set).apply(normed, axis=1),
    )

    # by capacity
    n.pf(distribute_slack=True, slack_weights="p_nom")

    slack_shares_by_capacity = (
        n.c.generators.dynamic.p - n.c.generators.dynamic.p_set
    ).apply(normed, axis=1)

    for _, row in slack_shares_by_capacity.iterrows():
        equal(n.c.generators.static.p_nom.pipe(normed).fillna(0.0), row)

    # by custom weights (mirror 'capacity' via custom slack weights by bus)
    custom_weights = {}
    for sub_network in n.c.sub_networks.static.obj:
        buses_o = sub_network.buses_o
        generators = sub_network.c.generators.static
        custom_weights[sub_network.name] = (
            generators.p_nom.groupby(generators.bus)
            .sum()
            .reindex(buses_o)
            .pipe(normed)
            .fillna(0.0)
        )

    n.pf(distribute_slack=True, slack_weights=custom_weights)

    equal(
        slack_shares_by_capacity,
        (n.c.generators.dynamic.p - n.c.generators.dynamic.p_set).apply(normed, axis=1),
    )

    custom_weights = {
        sub_network.name: sub_network.c.generators.static.p_nom
        for sub_network in n.c.sub_networks.static.obj
    }
    n.pf(distribute_slack=True, slack_weights=custom_weights)

    equal(
        slack_shares_by_capacity,
        (n.c.generators.dynamic.p - n.c.generators.dynamic.p_set).apply(normed, axis=1),
    )
