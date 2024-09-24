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
    n = ac_dc_network.copy()
    n.optimize.create_model()
    n.lines["carrier"] = n.lines.bus0.map(n.buses.carrier)
    n.generators.loc[n.generators.index[0], "p_nom_extendable"] = False
    return n


@pytest.mark.parametrize("groupby", groupers)
@pytest.mark.parametrize("include_non_extendable", [True, False])
def test_statistics_capex(prepared_network, groupby, include_non_extendable):
    n = prepared_network
    n.optimize.statistic_expressions.capex(
        groupby=groupby, include_non_extendable=include_non_extendable
    )


@pytest.mark.parametrize("groupby", groupers)
@pytest.mark.parametrize("include_non_extendable", [True, False])
def test_statistics_capacity(prepared_network, groupby, include_non_extendable):
    n = prepared_network
    n.optimize.statistic_expressions.capacity(
        groupby=groupby, include_non_extendable=include_non_extendable
    )


# @pytest.mark.parametrize("groupby", groupers)
# def test_statistics_opex(prepared_network, groupby):
#     n = prepared_network
#     n.optimize.statistic_expressions.opex(groupby=groupby)
