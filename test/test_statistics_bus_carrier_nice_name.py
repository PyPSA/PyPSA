import pandas as pd

import pypsa


# Test that filtering by bus_carrier without nice_names works
def test_energy_balance_bus_carrier_filter():
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Carrier", "rural heat")
    n.add("Bus", "heat bus", carrier="rural heat", unit="MW")
    n.add(
        "Load",
        "heat load",
        bus="heat bus",
        carrier="rural heat",
        p_set=[1.0],
    )
    n.c.loads.dynamic.p = n.c.loads.dynamic.p_set.copy()

    result = n.statistics.energy_balance(bus_carrier="rural heat")
    assert not result.empty
    assert isinstance(result.index, pd.MultiIndex)
    assert "bus_carrier" in result.index.names
    assert "rural heat" in result.index.get_level_values("bus_carrier")


# Test that filtering by bus_carrier with nice_names works
def test_energy_balance_bus_carrier_nice_name_filter():
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add("Carrier", "rural heat", nice_name="residential rural heat")
    n.add("Bus", "heat bus", carrier="rural heat", unit="MW")
    n.add(
        "Load",
        "heat load",
        bus="heat bus",
        carrier="rural heat",
        p_set=[1.0],
    )
    n.c.loads.dynamic.p = n.c.loads.dynamic.p_set.copy()

    displayed = n.statistics.energy_balance(nice_names=True)
    assert "residential rural heat" in displayed.index.get_level_values("bus_carrier")

    result = n.statistics.energy_balance(bus_carrier="residential rural heat")
    assert not result.empty
    assert "residential rural heat" in result.index.get_level_values("bus_carrier")
