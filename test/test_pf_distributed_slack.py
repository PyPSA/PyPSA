import os
import pypsa
from numpy.testing import assert_array_almost_equal as equal


def normed(s): return s/s.sum()

def test_pf_distributed_slack():
    csv_folder_name = os.path.join(os.path.dirname(__file__), "..",
                      "examples", "scigrid-de", "scigrid-with-load-gen-trafos")

    network = pypsa.Network(csv_folder_name)
    network.set_snapshots(network.snapshots[:2])

    #There are some infeasibilities without line extensions
    network.lines.s_max_pu = 0.7
    network.lines.loc[["316","527","602"],"s_nom"] = 1715
    network.storage_units.state_of_charge_initial = 0.

    network.lopf(network.snapshots, solver_name='glpk', formulation='kirchhoff')

    #For the PF, set the P to the optimised P
    network.generators_t.p_set = network.generators_t.p
    network.storage_units_t.p_set = network.storage_units_t.p

    #set all buses to PV, since we don't know what Q set points are
    network.generators.control = "PV"

    #Need some PQ buses so that Jacobian doesn't break
    f = network.generators[network.generators.bus == "492"]
    network.generators.loc[f.index,"control"] = "PQ"

    # by dispatch
    network.pf(distribute_slack=True, slack_weights='p_set')

    equal(
        network.generators_t.p_set.apply(normed, axis=1),
        (network.generators_t.p - network.generators_t.p_set).apply(normed, axis=1)
    )

    # by capacity
    network.pf(distribute_slack=True, slack_weights='p_nom')

    slack_shares_by_capacity = (network.generators_t.p - network.generators_t.p_set).apply(normed, axis=1)

    for index, row in slack_shares_by_capacity.iterrows():
        equal(
            network.generators.p_nom.pipe(normed).fillna(0),
            row
        )

    # by custom weights (mirror 'capacity' via custom slack weights by bus)
    custom_weights = {}
    for sub_network in network.sub_networks.obj:
        buses_o = sub_network.buses_o
        custom_weights[sub_network.name] = sub_network.generators().groupby('bus').sum().p_nom.reindex(buses_o).pipe(normed).fillna(0)

    network.pf(distribute_slack=True, slack_weights=custom_weights)

    equal(
        slack_shares_by_capacity,
        (network.generators_t.p - network.generators_t.p_set).apply(normed, axis=1)
    )

    # by custom weights (mirror 'capacity' via custom slack weights by generators)
    custom_weights = {}
    for sub_network in network.sub_networks.obj:
        custom_weights[sub_network.name] = sub_network.generators().p_nom # weights do not sum up to 1

    network.pf(distribute_slack=True, slack_weights=custom_weights)

    equal(
        slack_shares_by_capacity,
        (network.generators_t.p - network.generators_t.p_set).apply(normed, axis=1)
    )

