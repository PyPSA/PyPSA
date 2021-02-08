"""Logic for switching
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from six import iteritems


def init_switches(network):
    '''
    Initiate switches.
    '''
    logger.info("Initiating switches.")
    for switch in network.switches.index:
        assert not is_there_a_parallel_switch(network, switch), ("there is a switch that is parallel to another switch. that's prohibitted.")
    determine_logical_topology(network)
    find_only_logical_buses(network)
    find_switches_connections(network)
    switching(network)


def is_there_a_parallel_switch(network, switch):
    switch_b0_b1 = [network.switches.loc[switch, "bus0"], network.switches.loc[switch, "bus1"]]
    for other_switch in network.switches.index.drop(switch):
        if (network.switches.loc[other_switch, 'bus0'] in switch_b0_b1 and
            network.switches.loc[other_switch, 'bus1'] in switch_b0_b1):
            logger.error("switch %s is parllel to switch %s" % (switch, other_switch))
            return True
    return False


def reinit_switches_after_removing_components(network, cls_name, switch_related):
    """
    after removing components that are switch-related we need to init the
    switches again. using seperate function to make things clearer for now.
    see reinit_switches_after_adding_components for potential speed improvements.
    right now this is redetermining everything related to switches like in
    init_switches, but without calling switching()
    """
    # before calling this, we opened all switches and dropped the components
    logger.info("reinitializing switches after removing component(s) of type %s related to switches:\n%s" % (cls_name, switch_related))
    determine_logical_topology(network)
    find_only_logical_buses(network)
    find_switches_connections(network)
    # after calling this we will switch back to the switching status from before calling this


def reinit_switches_after_adding_components(network, status_old_switches, cls_name, switch_related, skip_result_deletion=False):
    """
    after adding components that are switch-related we need to init the
    switches again. the reinit for switches includes:
        1. open closed switches
        1. determine_logical_topology(network)
        2. find_only_logical_buses(network)
        3. find_switches_connections(network)
        4. reclose closed switches

    in this seperate function we imrpove speed for 3. because it is the slowest part and do this instead:
        - update switches_conncetions by only making changes to the switches that are attached to the affected buses
          of the new component(s)
    """
    logger.info("reinitializing switches after adding component(s) of type %s related to switches:\n%s" % (cls_name, switch_related))
    switches_status_before = network.switches.status.copy()  # there might be new ones
    closed_switches_before = switches_status_before.loc[switches_status_before == 1].index.tolist()
    closed_old_switches = status_old_switches.loc[status_old_switches == 1].index.tolist()
    network.open_switches(closed_old_switches, skip_result_deletion=True)
    determine_logical_topology(network)
    find_only_logical_buses(network)
    # find_switches_connections hurts most, because it is slow:
    if cls_name == "Switch":  # TODO: think about this...
        logger.info("adding a switch, consider improving speed... redetermining all switches_connections.")
        find_switches_connections(network)
    else:
        # start section replacing find_switches_connections:
        list_name = network.components[cls_name]["list_name"]
        cls_df = network.df(cls_name).loc[switch_related]
        is_branch = True if "bus0" in cls_df.columns else False
        # collect affected buses:
        affected_buses = []
        for b in ["bus", "bus0", "bus1"]:
            if b in cls_df.columns:
                affected_buses += cls_df[b].tolist()
        affected_buses = list(set(affected_buses))
        # collect affected switches:
        affected_switches = network.switches.loc[(network.switches.bus0.isin(affected_buses))
                                                 | network.switches.bus1.isin(affected_buses)].index
        switches_connections = network.switches_connections
        for i in switches_connections.index:  # lists in dataframe cells are a problem
            switches_connections.loc[i] = switches_connections.loc[i].copy()

        def extend_switches_connections(switches_connections, switch, column_name, connections_list):
            """
            append connections_list to switches_connections.loc[switch, column_name].
            if the column is missing add it filled with [] to switches_connections first.
            """
            try:
                switches_connections.loc[switch, column_name] += (connections_list)
            except KeyError:
                logger.info("reinit_switches_after_adding_components: adding column %s to switches_connections",
                            column_name)
                switches_connections[column_name] = switches_connections.apply(lambda x: [], axis=1)
                switches_connections.loc[switch, column_name] += (connections_list)
            return switches_connections

        for switch in affected_switches:
            if is_branch:
                rename_list_el0_log0 = cls_df.loc[cls_df.bus0 == network.switches.loc[switch, "bus0"]].index.tolist()
                rename_list_el0_log1 = cls_df.loc[cls_df.bus0 == network.switches.loc[switch, "bus1"]].index.tolist()
                rename_list_el1_log0 = cls_df.loc[cls_df.bus1 == network.switches.loc[switch, "bus0"]].index.tolist()
                rename_list_el1_log1 = cls_df.loc[cls_df.bus1 == network.switches.loc[switch, "bus1"]].index.tolist()
                switches_connections = extend_switches_connections(switches_connections, switch,
                                                                   "bus0_" + list_name + "_bus0", rename_list_el0_log0)
                switches_connections = extend_switches_connections(switches_connections, switch,
                                                                   "bus0_" + list_name + "_bus1", rename_list_el0_log1)
                switches_connections = extend_switches_connections(switches_connections, switch,
                                                                   "bus1_" + list_name + "_bus0", rename_list_el1_log0)
                switches_connections = extend_switches_connections(switches_connections, switch,
                                                                   "bus1_" + list_name + "_bus1", rename_list_el1_log1)
            else:
                rename_list_el_log0 = cls_df.loc[cls_df.bus == network.switches.loc[switch, "bus0"]].index.tolist()
                rename_list_el_log1 = cls_df.loc[cls_df.bus == network.switches.loc[switch, "bus1"]].index.tolist()
                switches_connections = extend_switches_connections(switches_connections, switch,
                                                                   "bus_" + list_name + "_bus0", rename_list_el_log0)
                switches_connections = extend_switches_connections(switches_connections, switch,
                                                                   "bus_" + list_name + "_bus1", rename_list_el_log1)
        setattr(network, "switches_connections", switches_connections)
        # end section replacing find_switches_connections(network)
    network.close_switches(closed_switches_before, skip_result_deletion)
    if cls_name == "Switch":  # TODO: think about this...
        network.switches.status = switches_status_before
        network.switching()


def add_switch(network, name, bus0, bus1, status, i_max=np.nan):
    """
    add switch to network. not trivial, we need to initiate switches again
    """
    logger.info("adding switch %s" % name)
    assert name not in network.switches.index, ("name for new switch has to be unique")
    assert not bus0 == bus1, ("do not add a switch with same bus0 and bus1")
    assert not is_switch_connecting_buses(network, bus0, bus1), (
            "do not add a switch that is parallel to another switch")
    if len(network.switches):  # we already have initiated switches
        switches_status_before = network.switches.status.copy()
        logger.info(network.buses)
        open_switches(network, network.switches.index)
    assert(bus0 in network.buses.index or bus0 in network.buses_only_logical.index) & (bus1 in network.buses.index or bus1 in network.buses_only_logical.index), (
           "when adding a switch, make sure to add its buses (%s and %s) to network.buses first:\n %s" % (bus0, bus1, network.buses))
    network.switches.loc[name] = {'i_max': i_max,
                                  'bus0': bus0, 'bus1': bus1,
                                  'status': status,
                                  'bus_connected': np.nan}
    determine_logical_topology(network)
    find_only_logical_buses(network)
    find_switches_connections(network)
    if len(network.switches) > 1:  # we already had initiated switches
        switches_status_before.loc[name] = status
        network.switches.status = switches_status_before
    switching(network)


def is_switch_connecting_buses(network, bus0, bus1):
    checked_buses = [bus0, bus1]
    for switch in network.switches.index:
        if (network.switches.loc[switch, 'bus0'] in checked_buses and
            network.switches.loc[switch, 'bus1'] in checked_buses):
            return True
    return False

def check_for_buses_only_logical_and_add_them_to_buses(network):
    found_and_readded = False
    try:
        if not network.buses_only_logical.index.isin(network.buses.index).all():
            logger.debug("network.buses_only_logical has already been initiated and they "
                         "are not all contained in network.buses so we need to add them to "
                         "network.buses here again.")
            new_df = pd.concat((network.buses, network.buses_only_logical), sort=False)
            if not new_df.index.is_unique:
                raise Exception("something is wrong. not all buses_only_logical have been in" +
                                "network.buses, but adding them leads to duplicated indices")
            setattr(network, network.components["Bus"]["list_name"], new_df)
            found_and_readded = True
    except AttributeError:
        logger.debug("network.buses_only_logical has not been initiated yet, or there are no switches")
    return found_and_readded


def determine_logical_topology(network):
    """
    Build logical_sub_networks from logical topology:
        - Subnetworks of logical elements share one "bus_connected".
          A unique name is built for each bus_connected and the dataframe
          network.buses_connected is created. These buses are used when closing
          switches.
        - In adition the dataframe network.buses_disconnected is created. These
          buses are used when opening switches.

    The attribute connected_bus of switches is assigned here.
    """
    logger.info("determining logical topology")
    # in case of a second call of this function we might need to do this:
    check_for_buses_only_logical_and_add_them_to_buses(network)
    buses_with_switches = (network.buses.loc[network.switches.bus1].index
                           .append(network.buses.loc[network.switches.bus0].index).drop_duplicates())
    adjacency_matrix = network.adjacency_matrix(["Switch"], buses_with_switches)  # TODO: for now only switches, but maybe use fuses or so also
    n_components, labels = csgraph.connected_components(adjacency_matrix, directed=False)
    # add unique name for bus_connected to buses.
    try:  # raises AttributeError when initiating empty network because .str fails
        if network.buses.index.str.contains('bus_connected').any():
            logger.warn("determine_logical_topology: trying to create unique" +
                        "bus names with 'bus_connected' + index here." +
                        "This string is already contained in buses index.")
    except AttributeError:
        logger.info("determine_logical_topology: it seems there are no buses?")
    labels = ['bus_connected' + str(s) for s in labels]
    # add column bus_connected to buses and fill in the unique name for buses in each logical subnetwork
    network.buses.loc[buses_with_switches, "bus_connected"] = labels
    # copy buses with bus_connected and import them with bus_connected as index
    buses_connected = network.buses.loc[buses_with_switches].drop_duplicates(subset="bus_connected")
    network.buses_connected = buses_connected.set_index("bus_connected")
    for c in network.iterate_components(["Switch"]):  # TODO: for now only switches, but maybe use fuses or so also
        c.df["bus_connected"] = c.df.bus0.map(network.buses["bus_connected"])
    # now we dont need the column bus_connected at buses anymore
    network.buses = network.buses.drop(columns="bus_connected")
    network.buses_disconnected = (network.buses.loc[network.buses.loc[network.switches.bus1].index
                                  .append(network.buses.loc[network.switches.bus0].index).drop_duplicates()])
    # TODO: any need for this?
    # map this bus to all other elements
    """
    for c in network.iterate_components(network.branch_components):
        c.df["bus_connected0"] = c.df.bus0.map(network.buses["bus_connected"])
        c.df["bus_connected1"] = c.df.bus1.map(network.buses["bus_connected"])
        c.df["bus_disconnected0"] = c.df.bus0.loc[c.df.bus_connected0.notna()]
        c.df["bus_disconnected1"] = c.df.bus1.loc[c.df.bus_connected1.notna()]
    for c in network.iterate_components(network.one_port_components):
        c.df["bus_connected"] = c.df.bus.map(network.buses["bus_connected"])
        c.df["bus_disconnected"] = c.df.bus.loc[c.df.bus_connected.notna()]
    """


def find_only_logical_buses(network):
    """
    create dataframe of only logical buses and assign it to network.buses_only_logical. can
    be used to avoid sub_networks when opening switches. drop the found buses_only_logical from
    network.buses
    """
    logger.info("find_only_logical_buses: creating network.buses_only_logical and drop them from network.buses")
    # all switches need to be open and in case buses_only_logical already have been dropped they need to be readded
    # in case of a second call of this function we might need to do this:
    check_for_buses_only_logical_and_add_them_to_buses(network)
    buses_with_switches = (network.buses.loc[network.switches.bus1].index
                           .append(network.buses.loc[network.switches.bus0].index).drop_duplicates())
    electrical_buses = []
    for c in network.iterate_components(network.branch_components):
        electrical_in_c = buses_with_switches[buses_with_switches.isin(c.df.bus0) | buses_with_switches.isin(c.df.bus1)]
        electrical_buses += electrical_in_c.tolist()
    for c in network.iterate_components(network.one_port_components):
        electrical_in_c = buses_with_switches[buses_with_switches.isin(c.df.bus)]
        electrical_buses += electrical_in_c.tolist()
    network.buses_only_logical = network.buses.loc[buses_with_switches.drop(electrical_buses)].copy()
    # drop only logical buses from network.buses to avoid subnetworks
    network.buses.drop(network.buses_only_logical.index, inplace=True)


def find_switches_connections(network):
    """
    add the dataframe switches_connections to the network with switches as
    index and columns that indicate for every existent type of component:
        - with wich bus it is connected to the switch
        - at which bus of the switch it is connected
    The columns are named, so that they can be used for accessing the
    relevant dataframes when splitting them by "_". For example, if lines
    are existent in the network the df switches_connections will have these
    columns:
    bus0_lines_bus0, bus0_lines_bus1, bus_1_lines_bus0, bus1_lines_bus1
    Each cell contains a list of indexes of connected electrical elements.
    """
    logger.info("find_switches_connections: creating network.switches_connections")
    n_switches = len(network.switches)  # TODO: for now only switches, but maybe use fuses or so also
    switches_connections = pd.DataFrame(index=network.switches.index)
    for c in network.iterate_components(network.branch_components):
        switches_connections["bus0_" + c.list_name + "_bus0"] = np.empty((n_switches, 0)).tolist()
        switches_connections["bus0_" + c.list_name + "_bus1"] = np.empty((n_switches, 0)).tolist()
        switches_connections["bus1_" + c.list_name + "_bus0"] = np.empty((n_switches, 0)).tolist()
        switches_connections["bus1_" + c.list_name + "_bus1"] = np.empty((n_switches, 0)).tolist()
    for c in network.iterate_components(network.one_port_components):
        switches_connections["bus_" + c.list_name + "_bus0"] = np.empty((n_switches, 0)).tolist()
        switches_connections["bus_" + c.list_name + "_bus1"] = np.empty((n_switches, 0)).tolist()
    for switch in network.switches.index:
        for c in network.iterate_components(network.branch_components):
            rename_list_el0_log0 = c.df.loc[c.df.bus0 == network.switches.loc[switch, "bus0"]].index.tolist()
            rename_list_el0_log1 = c.df.loc[c.df.bus0 == network.switches.loc[switch, "bus1"]].index.tolist()
            rename_list_el1_log0 = c.df.loc[c.df.bus1 == network.switches.loc[switch, "bus0"]].index.tolist()
            rename_list_el1_log1 = c.df.loc[c.df.bus1 == network.switches.loc[switch, "bus1"]].index.tolist()
            switches_connections.loc[switch, "bus0_" + c.list_name + "_bus0"] += (rename_list_el0_log0)
            switches_connections.loc[switch, "bus0_" + c.list_name + "_bus1"] += (rename_list_el0_log1)
            switches_connections.loc[switch, "bus1_" + c.list_name + "_bus0"] += (rename_list_el1_log0)
            switches_connections.loc[switch, "bus1_" + c.list_name + "_bus1"] += (rename_list_el1_log1)
        for c in network.iterate_components(network.one_port_components):
            rename_list_el_log0 = c.df.loc[c.df.bus == network.switches.loc[switch, "bus0"]].index.tolist()
            rename_list_el_log1 = c.df.loc[c.df.bus == network.switches.loc[switch, "bus1"]].index.tolist()
            switches_connections.loc[switch, "bus_" + c.list_name + "_bus0"] += rename_list_el_log0
            switches_connections.loc[switch, "bus_" + c.list_name + "_bus1"] += rename_list_el_log1
    network.switches_connections = switches_connections.copy()


def delete_calculation_results(network):
    """ delete all calculation results """
    to_drop = {'Generator': ['p'],
               'Load': ['p'],
               'StorageUnit': ['p'],
               'Store': ['p'],
               'ShuntImpedance': ['p'],
               'Bus': ['p', 'v_ang', 'v_mag_pu'],
               'Line': ['p0', 'p1'],
               'Transformer': ['p0', 'p1'],
               'Link': ["p" + col[3:] for col in network.links.columns if col[:3] == "bus"]}
    if len(network.buses_t.q):
        for component, attrs in to_drop.items():
            if "p" in attrs:
                attrs.append("q")
            if "p0" in attrs and component != 'Link':
                attrs.extend(["q0", "q1"])

    # reindex and set all values to default
    for component, attributes in iteritems(to_drop):
        df = network.df(component)
        pnl = network.pnl(component)  # list of dfs
        for attr in attributes:
            fill_value = network.components[component]["attrs"].at[attr, "default"]
            pnl[attr] = pnl[attr].reindex(columns=df.index, fill_value=fill_value)
            for col in pnl[attr].columns:
                pnl[attr][col].values[:] = fill_value


def close_switches(network, switches, skip_result_deletion=False):
    """
    In order to close switches we:
        - let bus_disconnected disappear in one_port_components.bus and replace it with bus_connected
        - let bus_disconnected0 and bus_disconnected1 disappear in
          branch_components.bus0 and in branch_components.bus1 and replace it with bus_connected
    """
    logger.info("closing switches")
    # change status of switches:
    network.switches.loc[switches, "status"] = 1
    for col in network.switches_connections.columns:
        # change names in all connected components:
        el_bus_component_log_bus = str.split(col, "_")
        el_bus = el_bus_component_log_bus[0]
        component = el_bus_component_log_bus[1]
        log_bus = el_bus_component_log_bus[2]
        switches_con = network.switches_connections.loc[switches, col]
        for switch in switches:
            switch_con = switches_con.loc[switch]
            if len(switch_con):
                getattr(network, component).loc[switch_con, el_bus] = network.switches.loc[switch, "bus_connected"]
    # add buses
    new_df = pd.concat((network.buses,
                        network.buses_connected.loc[network.switches.loc[switches, "bus_connected"]]), sort=False)
    # the buses might already exist, as the bus_connected is shared:
    if not new_df.index.is_unique:
        logger.debug("New components for buses are not unique, keeping only the first occurance")
        new_df = new_df.loc[~new_df.index.duplicated(keep='first')]
    setattr(network, network.components["Bus"]["list_name"], new_df)
    # remove buses
    # if not existent we ignore the error
    network.buses.drop(network.buses_disconnected.loc[network.switches.loc[switches, "bus0"]].index,
                       errors='ignore', inplace=True)
    network.buses.drop(network.buses_disconnected.loc[network.switches.loc[switches, "bus1"]].index,
                       errors='ignore', inplace=True)
    # TODO: consider adding other elements than buses that have been out of service.
    # Note that the pypsa developpers are planning to add
    # a column "operational" for all assets. (https://github.com/PyPSA/PyPSA/pull/77)
    if not skip_result_deletion:
        delete_calculation_results(network)


def open_switches(network, switches, skip_result_deletion=False, skip_reclosing=False):
    """
    In order to open switches we:
        - let bus_connected disappear in one_port_components.bus and replace it with bus_disconnected
        - let bus_connected0 disappear in branch_components.bus0 and replace it with bus_diconnected0
        - let bus_connected1 disappear in branch_components.bus1 and replace it with bus_diconnected1
    """
    logger.info("opening switches")
    # change status of switches:
    network.switches.loc[switches, "status"] = 0
    for col in network.switches_connections.columns:
        # change names in all connected components:
        el_bus_component_log_bus = str.split(col, "_")
        el_bus = el_bus_component_log_bus[0]
        component = el_bus_component_log_bus[1]
        log_bus = el_bus_component_log_bus[2]
        switches_con = network.switches_connections.loc[switches, col]
        for switch in switches:
            switch_con = switches_con.loc[switch]
            if len(switch_con):
                getattr(network, component).loc[switch_con, el_bus] = network.switches.loc[switch, log_bus]
    # add relvant buses from network.buses_disconnected
    # there are three kinds of buses:
    # only electrical ones. they will never appear here.
    # only logical ones. they should never be added because they will cause subnetworks
    # dual-use buses. they should be added every time a switch is opened
    buses_that_do_not_exist = (network.switches.loc[switches].loc[~network.switches.loc[switches, "bus0"]
                               .isin(network.buses.index), "bus0"].tolist())
    buses_that_do_not_exist += (network.switches.loc[switches].loc[~network.switches.loc[switches, "bus1"]
                                .isin(network.buses.index), "bus1"].tolist())
    # remove only logical buses
    buses_that_do_not_exist = list(set(buses_that_do_not_exist) - set(network.buses_only_logical.index.tolist()))
    logger.debug("Adding these buses, because they represent auxilary " +
                 "buses for switches that are open:\n%s" % buses_that_do_not_exist)
    network.import_components_from_dataframe(network.buses_disconnected.loc[buses_that_do_not_exist], "Bus")
    # remove buses:
    # connected_buses may only be removed when all switches, that share this bus are open
    check_for_unanimity = network.switches.loc[switches, "bus_connected"]
    to_drop = []
    to_reclose = []
    for bus in check_for_unanimity:
        i_closed = network.switches.loc[network.switches.bus_connected == bus].loc[network.switches.status == 1].index.tolist()
        if len(i_closed) == 0:
            to_drop.append(bus)
        else:
            # there is one switch in the logical subnetwork that is closed
            # in case it is connected to the same electrical element, we need
            # to bring back its bus_connected at that electrical element
            to_reclose += i_closed
    if len(to_reclose) and not skip_reclosing:
        to_reclose = list(set(to_reclose))
        logger.info("these switches might need to be closed again:\n%s", to_reclose)
        network.close_switches(to_reclose, skip_result_deletion=True)
    to_drop = list(set(to_drop))
    logger.debug("Removing these buses, because all switches that share those as connected_bus are open:\n%s" % to_drop)
    network.buses.drop(to_drop, errors='ignore', inplace=True)
    # TODO: consider removing other elements than buses. Note that the pypsa developpers are planning to add
    # a column "operational" for all assets. (https://github.com/PyPSA/PyPSA/pull/77)
    # For now, the elements stay and might build subnetworks. Idea for code see below
    """
    # those columns are not initiated in determine_logical_topology() (commented out)
    # after opening a switch there is a chance for elements to be out of service:
    # this is not only dependent on the given switches for branches:
    double_switcheable = network.lines.loc[network.lines["bus_connected0"].notna() &
                                        network.lines["bus_connected1"].notna()]
    for l in double_switcheable.index:
        if ((double_switcheable.loc[l, "bus_disconnected0"] in network.buses.index) &
            (double_switcheable.loc[l, "bus_disconnected1"] in network.buses.index)):
            logger.warn("Line %s has with switches opened at both sides" % l)
            network.os_lines.append(network.lines.loc[l], sort=True)[network.lines.columns.tolist()]
            network.remove("Line", l)
    """
    if not skip_result_deletion:
        delete_calculation_results(network)


def switching(network):
    """
    use switches.status to build the network topology
    """
    logger.info("switching all switches")
    network.open_switches(network.switches.loc[network.switches.status == 0].index, skip_result_deletion=True, skip_reclosing=True)
    network.close_switches(network.switches.loc[network.switches.status == 1].index)
