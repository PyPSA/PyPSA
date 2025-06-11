from numpy.testing import assert_array_almost_equal as equal


def normed(s):
    return s / s.sum()


def test_pf_distributed_slack(scipy_network):
    n = scipy_network
    n.set_snapshots(n.snapshots[:2])

    # There are some infeasibilities without line extensions
    n.lines.s_max_pu = 0.7
    n.lines.loc[["316", "527", "602"], "s_nom"] = 1715
    n.storage_units.state_of_charge_initial = 0.0

    n.optimize(n.snapshots)

    # For the PF, set the P to the optimised P
    n.generators_t.p_set = n.generators_t.p
    n.storage_units_t.p_set = n.storage_units_t.p

    # set all buses to PV, since we don't know what Q set points are
    n.generators.control = "PV"

    # Need some PQ buses so that Jacobian doesn't break
    f = n.generators[n.generators.bus == "492"]
    n.generators.loc[f.index, "control"] = "PQ"
    # by dispatch
    n.pf(distribute_slack=True, slack_weights="p_set")

    equal(
        n.generators_t.p_set.apply(normed, axis=1),
        (n.generators_t.p - n.generators_t.p_set).apply(normed, axis=1),
    )

    # by capacity
    n.pf(distribute_slack=True, slack_weights="p_nom")

    slack_shares_by_capacity = (n.generators_t.p - n.generators_t.p_set).apply(
        normed, axis=1
    )

    for _, row in slack_shares_by_capacity.iterrows():
        equal(n.generators.p_nom.pipe(normed).fillna(0.0), row)

    # by custom weights (mirror 'capacity' via custom slack weights by bus)
    custom_weights = {}
    for sub_network in n.sub_networks.obj:
        buses_o = sub_network.buses_o
        generators = sub_network.generators()
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
        (n.generators_t.p - n.generators_t.p_set).apply(normed, axis=1),
    )

    custom_weights = {
        sub_network.name: sub_network.generators().p_nom
        for sub_network in n.sub_networks.obj
    }
    n.pf(distribute_slack=True, slack_weights=custom_weights)

    equal(
        slack_shares_by_capacity,
        (n.generators_t.p - n.generators_t.p_set).apply(normed, axis=1),
    )
