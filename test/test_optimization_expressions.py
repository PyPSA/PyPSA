import pytest

from pypsa.statistics import (
    get_bus_and_carrier,
    get_bus_and_carrier_and_bus_carrier,
    get_carrier_and_bus_carrier,
    get_name_bus_and_carrier,
)

TOLERANCE = 1e-2


groupers = [
    get_bus_and_carrier,
    get_name_bus_and_carrier,
    get_carrier_and_bus_carrier,
    get_bus_and_carrier_and_bus_carrier,
    False,
    None,
]


@pytest.fixture
def prepared_network(ac_dc_network):
    n = ac_dc_network
    n.optimize.create_model()
    n.lines["carrier"] = n.lines.bus0.map(n.buses.carrier)
    return n


@pytest.mark.parametrize("groupby", groupers)
def test_statistics_capex_by_carrier(prepared_network, groupby):
    n = prepared_network
    df = n.optimize.statistic_expressions.capex(groupby=groupby)
    print(df)


# n = pypsa.examples.ac_dc_meshed()
# n.optimize.create_model()
# s = n.optimize.statistic_expressions
# g = s.groupers

# s.capex(groupby=None)
# s.capex(groupby=False)
# s.capex(groupby=False, comps="Link")
# s.capex(groupby=g.get_bus_and_carrier)
# s.capex(bus_carrier="AC")
# # still broken
# # s.capex(groupby=g.get_name_bus_and_carrier)


# from linopy import merge


# data = {k: d[k].indexes[d[k].dims[0]] for k in d.keys()}
# tuples = [(key, value) for key, idx in data.items() for value in idx]
# multi_index = pd.MultiIndex.from_tuples(tuples, names=["Type", "Carrier"])

# merge(list(d.values()), dim="carrier")
