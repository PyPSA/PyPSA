"""Logic for switching
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy.sparse import csgraph


def init_switches(network):
    '''
    Initiate switches. Save the initial switching combination into
    network.allowed_switching_combos.loc["initial"]
    '''
    logger.info("Initiating switches. adding network.allowed_switching_combos.loc['initial']")
    find_only_logical_buses(network)
    determine_logical_topology(network)
    find_switches_connections(network)
    network.allowed_switching_combos = pd.DataFrame(columns=network.switches.index.tolist())
    network.allowed_switching_combos.loc["initial"] = network.switches.status
    switching(network)


def add_switch(network, name, bus0, bus1, status, i_max=np.nan):
    """
    add switch to network. not trivial, we need to initiate switches again
    """
    assert ~(bus0 in network.buses.index) | ~(bus1 in network.buses.index), ("when adding a switch, make sure to add its buses to network.buses first")
    switches_status_before = network.switches.status.copy()
    open_switches(network, network.switches.index)
    network.switches.loc[name] = {'i_max': i_max,
                                  'bus0': bus0, 'bus1': bus1,
                                  'status': status,
                                  'bus_connected': np.nan}
    determine_logical_topology(network)
    find_switches_connections(network)
    find_only_logical_buses(network)
    switches_status_before.loc[name] = status
    network.switches.status = switches_status_before
    switching(network)


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
    if not network.buses_only_logical.index.isin(network.buses.index).all():
        logger.info("this is not the initial call of determine_logical_topology()" +
                    "and we need to add the only logical buses to buses here again.")
        new_df = pd.concat((network.buses, network.buses_only_logical), sort=False)
        if not new_df.index.is_unique:
            raise Exception ("something is wrong. not all buses_only_logical have been in" + 
                             "the network, but adding them leads to duplicated indices")
        setattr(network, network.components["Bus"]["list_name"], new_df)
    buses_with_switches = network.buses.loc[network.switches.bus1].index.append(network.buses.loc[network.switches.bus0].index).drop_duplicates()
    adjacency_matrix = network.adjacency_matrix(["Switch"], buses_with_switches)  # TODO: for now only switches, but maybe use fuses or so also
    n_components, labels = csgraph.connected_components(adjacency_matrix, directed=False)
    # add unique name for bus_connected to buses.
    if network.buses.index.str.contains('bus_connected').any():
        logger.warn("determine_logical_topology: trying to create unique" +
                    "bus names with 'bus_connected' + index here." +
                    "This string is already contained in buses index.")
    labels = ['bus_connected' + str(s) for s in labels]
    # add column bus_connected to buses and fill in the unique name for buses in each logical subnetwork
    network.buses.loc[buses_with_switches, "bus_connected"] = labels
    # copy buses with bus_connected and import theam with bus_connected as index
    buses_connected = network.buses.loc[buses_with_switches].drop_duplicates(subset="bus_connected")
    network.buses_connected = buses_connected.set_index("bus_connected")
    # network.import_components_from_dataframe(network.buses_connected, "Bus")
    network.buses_disconnected = (network.buses.loc[network.buses.loc[network.switches.bus1].index.append(network.buses.loc[network.switches.bus0].index).drop_duplicates()])
    for c in network.iterate_components(["Switch"]):  # TODO: for now only switches, but maybe use fuses or so also
        c.df["bus_connected"] = c.df.bus0.map(network.buses["bus_connected"])
    # now we dont need the column bus_connected at buses anymore
    network.buses = network.buses.drop(columns="bus_connected")
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
    create list of only logical buses and assign it to network.buses_only_logical. can
    be used to avoid sub_networks when opening switches
    """
    logger.info("find_only_logical_buses: creating network.buses_only_logical")
    buses_with_switches = network.buses.loc[network.switches.bus1].index.append(network.buses.loc[network.switches.bus0].index).drop_duplicates()
    electrical_buses = []
    for c in network.iterate_components(network.branch_components):
        electrical_in_c = buses_with_switches[buses_with_switches.isin(c.df.bus0) | buses_with_switches.isin(c.df.bus1)]
        electrical_buses += electrical_in_c.tolist()
    for c in network.iterate_components(network.one_port_components):
        electrical_in_c = buses_with_switches[buses_with_switches.isin(c.df.bus)]
        electrical_buses += electrical_in_c.tolist()
    network.buses_only_logical = network.buses.loc[buses_with_switches.drop(electrical_buses)].copy()


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
    network.switches_connections = switches_connections


def close_switches(network, switches):
    """
    In order to close switches we:
        - let bus_disconnected disappear in one_port_components.bus and replace it with bus_connected
        - let bus_disconnected0 and bus_disconnected1 disappear in
          branch_components.bus0 and in branch_components.bus1 and replace it with bus_connected
    """
    logger.info("closing switches")
    for switch in switches:
        # change status of switch:
        network.switches.loc[switch, "status"] = 1
        # change names in all connected components:
        for el_bus_component_log_bus in network.switches_connections.loc[switch].index:
            switch_con = network.switches_connections.loc[switch, el_bus_component_log_bus]
            el_bus_component_log_bus = str.split(el_bus_component_log_bus, "_")
            el_bus = el_bus_component_log_bus[0]
            component = el_bus_component_log_bus[1]
            log_bus = el_bus_component_log_bus[2]
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


def open_switches(network, switches):
    """
    In order to open switches we:
        - let bus_connected disappear in one_port_components.bus and replace it with bus_disconnected
        - let bus_connected0 disappear in branch_components.bus0 and replace it with bus_diconnected0
        - let bus_connected1 disappear in branch_components.bus1 and replace it with bus_diconnected1
    """
    logger.info("opening switches")
    for switch in switches:
        # change status of switch:
        network.switches.loc[switch, "status"] = 0
        # change names in all connected components:
        for el_bus_component_log_bus in network.switches_connections.loc[switch].index:
            switch_con = network.switches_connections.loc[switch, el_bus_component_log_bus]
            el_bus_component_log_bus = str.split(el_bus_component_log_bus, "_")
            el_bus = el_bus_component_log_bus[0]
            component = el_bus_component_log_bus[1]
            log_bus = el_bus_component_log_bus[2]
            getattr(network, component).loc[switch_con, el_bus] = network.switches.loc[switch, log_bus]
    # add relvant buses from network.buses_disconnected
    # there are three kinds of buses:
    # only electrical ones. they will never appear here.
    # only logical ones. they should never be added because they will cause subnetworks
    # dual-use buses. they should be added every time a switch is opened
    buses_that_do_not_exist = network.switches.loc[switches].loc[~network.switches.loc[switches, "bus0"].isin(network.buses.index), "bus0"].tolist()
    buses_that_do_not_exist += network.switches.loc[switches].loc[~network.switches.loc[switches, "bus1"].isin(network.buses.index), "bus1"].tolist()
    # remove only logical buses
    buses_that_do_not_exist = list(set(buses_that_do_not_exist) - set(network.buses_only_logical.index.tolist()))
    logger.debug("Adding these buses, because they represent auxilary buses for switches that are open:\n%s" % buses_that_do_not_exist)
    network.import_components_from_dataframe(network.buses_disconnected.loc[buses_that_do_not_exist], "Bus")
    # remove buses:
    # connected_buses may only be removed when all switches, that share this bus are open
    check_for_unanimity = network.switches.loc[switches, "bus_connected"]
    to_drop = []
    for bus in check_for_unanimity:
        n_closed = network.switches.loc[network.switches.bus_connected == bus, "status"].sum()
        if n_closed == 0:
            to_drop.append(bus)
    to_drop=list(set(to_drop))
    logger.debug("Removing these buses, because all switches that share those as connected_bus are open:\n%s" % to_drop)
    network.buses.drop(to_drop, inplace=True)
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


def switching(network, skip_check=False):
    """
    use switches.status to build the network topology
    """
    logger.info("switching all switches")
    network.close_switches(network.switches.loc[network.switches.status==1].index)
    network.open_switches(network.switches.loc[network.switches.status==0].index)