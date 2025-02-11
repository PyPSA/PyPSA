import pypsa

def test_emissions():
    n = pypsa.examples.ac_dc_meshed()
    n.optimize(n.snapshots[:14])
    n.statistics().round(1).T
    # assert reloaded.meta == scipy_network.meta
    assert n.statistics()['Carbon Emission'].sum() == n.global_constraints.constant['co2_limit']
