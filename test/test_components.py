import pytest

import pypsa


@pytest.fixture
def network():
    return pypsa.examples.ac_dc_meshed(from_master=True)


def test_mremove(network):
    """
    GIVEN   the AC DC exemplary pypsa network

    WHEN    two components of Generator are removed with mremove

    THEN    the generator dataframe and the time-dependent generator dataframe
                should not contain the removed elements.
    """
    generators = ['Manchester Wind', 'Frankfurt Wind']

    network.mremove('Generator', generators)

    assert generators not in network.generators.index.tolist()
    assert generators not in network.generators_t.p_max_pu.columns.tolist()


def test_mremove_misspelled_component(network, caplog):
    """
    GIVEN   the AC DC exemplary pypsa network

    WHEN    a misspelled component is removed with mremove

    THEN    the function should not change anything in the Line component
                dataframe and an error should be logged.
    """

    len_lines = len(network.lines.index)

    network.mremove('Liness', ['0', '1'])

    assert len_lines == len(network.lines.index)
    assert caplog.records[-1].levelname == 'ERROR'


def test_mremove_non_component(network, caplog):
    """
    GIVEN   the AC DC exemplary pypsa network

    WHEN    a non-existing component is removed with mremove

    THEN    an error should be logged.
    """

    network = pypsa.examples.ac_dc_meshed(from_master=True)

    network.mremove('ABC', ['0', '1'])

    assert caplog.records[-1].levelname == 'ERROR'
