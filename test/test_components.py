import numpy as np
import pytest

import pypsa


@pytest.fixture
def network():
    return pypsa.examples.ac_dc_meshed(from_master=True)


@pytest.fixture
def empty_network_5_buses():
    # Set up empty network with 5 buses.
    network = pypsa.Network()
    n_buses = 5
    for i in range(n_buses):
        network.add(
            'Bus',
            f'bus_{i}'
        )
    return network


def test_mremove(network):
    """
    GIVEN   the AC DC exemplary pypsa network

    WHEN    two components of Generator are removed with mremove

    THEN    the generator dataframe and the time-dependent generator dataframe
                should not contain the removed elements.
    """
    generators = {'Manchester Wind', 'Frankfurt Wind'}

    network.mremove('Generator', generators)

    assert not generators.issubset(network.generators.index)
    assert not generators.issubset(network.generators_t.p_max_pu.columns)


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


def test_madd_static(empty_network_5_buses):
    """
    GIVEN   an empty PyPSA network with 5 buses

    WHEN    multiple components of Load are added to the network with madd and
                attribute p_set

    THEN    the corresponding load components should be in the index of the
                static load dataframe. Also the column p_set should contain any
                value greater than 0.
    """

    buses = empty_network_5_buses.buses.index

    # Add load components at every bus with attribute p_set.
    load_names = "load_" + buses
    empty_network_5_buses.madd(
        "Load",
        load_names,
        bus=buses,
        p_set=3,
    )

    assert load_names.equals(empty_network_5_buses.loads.index)
    assert (empty_network_5_buses.loads.p_set == 3).all()


def test_madd_t(empty_network_5_buses):
    """
    GIVEN   an empty PyPSA network with 5 buses and 7 snapshots

    WHEN    multiple components of Load are added to the network with madd and
                attribute p_set

    THEN    the corresponding load components should be in the columns of the
                time-dependent load_t dataframe. Also, the shape of the
                dataframe should resemble 7 snapshots x 5 buses.
    """

    # Set up empty network with 5 buses and 7 snapshots.
    snapshots = range(7)
    empty_network_5_buses.set_snapshots(snapshots)
    buses = empty_network_5_buses.buses.index

    # Add load component at every bus with time-dependent attribute p_set.
    load_names = "load_" + buses
    empty_network_5_buses.madd(
        "Load",
        load_names,
        bus=buses,
        p_set=np.random.rand(len(snapshots), len(buses)),
    )

    assert load_names.equals(empty_network_5_buses.loads_t.p_set.columns)
    assert empty_network_5_buses.loads_t.p_set.shape == (
        len(snapshots), len(buses)
    )


def test_madd_misspelled_component(empty_network_5_buses, caplog):
    """
    GIVEN   an empty PyPSA network with 5 buses

    WHEN    multiple components of a misspelled component are added

    THEN    the function should not change anything and an error should be
                logged.
    """

    misspelled_component = 'Generatro'
    empty_network_5_buses.madd(
        misspelled_component,
        ['g_1', 'g_2'],
        bus=['bus_1', 'bus_2'],
    )

    assert empty_network_5_buses.generators.empty
    assert caplog.records[-1].levelname == 'ERROR'
    assert caplog.records[-1].message == (
        f'Component class {misspelled_component} not found'
    )


def test_madd_duplicated_index(empty_network_5_buses, caplog):
    """
    GIVEN   an empty PyPSA network with 5 buses

    WHEN    adding generators with the same name

    THEN    the function should fail and an error should be logged.
    """

    empty_network_5_buses.madd(
        "Generator",
        ['g_1', 'g_1'],
        bus=['bus_1', 'bus_2'],
    )

    assert caplog.records[-1].levelname == 'ERROR'
    assert caplog.records[-1].message == (
        'Error, new components for Generator are not unique'
    )


def test_madd_defaults(empty_network_5_buses):
    """
    GIVEN   an empty PyPSA network with 5 buses

    WHEN    adding multiple components of Generator and Load with madd

    THEN    the defaults should be set correctly according to n.component_attrs.
    """

    gen_names = ['g_1', 'g_2']
    empty_network_5_buses.madd(
        'Generator',
        gen_names,
        bus=['bus_1', 'bus_2'],
    )

    line_names = ['l_1', 'l_2']
    empty_network_5_buses.madd(
        'Load',
        line_names,
        bus=['bus_1', 'bus_2'],
    )

    assert empty_network_5_buses.generators.loc[gen_names[0], 'control'] == (
        empty_network_5_buses.component_attrs.Generator.loc[
            'control',
            'default'
        ]
    )
    assert empty_network_5_buses.loads.loc[line_names[0], 'p_set'] == (
        empty_network_5_buses.component_attrs.Load.loc['p_set', 'default']
    )
