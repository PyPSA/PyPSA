"""importing important libraries."""
import logging
import numpy as np
from copy import deepcopy
logger = logging.getLogger(__name__)


def fixed_cosphi(p_input, now, parameters):
    """
    fix power factor inverter controller.
    reference : https://www.academia.edu/24772355/

    Parameters
    ----------
    p_input : pandas data frame
        Input active power for controller, comming from n.component_t.p.
    now : single snapshot
        Current  element of n.snapshots on which the power flow is run.
    parameters : pandas data frame
        Controller parameters chosen for fix power factor controller for each
        index of elements, i.e. power factor choice for controlling any generator
        , load or storage unit elements.

    Returns
    -------
    q_set : pandas data frame
        Is the new reactive power that will be set as new q_set in each
        controlled component as a result of applying this controller.
    """
    q_set = -p_input.loc[now, parameters['power_factor'].index].mul(
                                 np.tan(np.arccos(parameters['power_factor'])))
    return q_set


def cosphi_p(p_input, now, parameters, p_set_varies_per_snapshot, df, df_t):
    """
    Power factor as a function of active power (cosphi_p) controller.
    reference : https://ieeexplore.ieee.org/document/6096349.

    Parameters
    ----------
    p_input : pandas data frame
        Input active power for controller, comming from n.component_t.p.
    now : single snapshot
        Current  element of n.snapshots on which the power flow is run.
    p_set_varies : bool (True / False)
        It determines if p_set is changing in each snapshot (True) or fixed.
    df : pandas data frame
        Component data frame, i.e. n.loads, n.storage_units...
    df_t : pandas data frame
        Component data frame, i.e. n.loads_t, n.storage_units_t...
    parameters : pandas data frame
        Controller parameters chosen for this controller for each index of the
        elements, i.e. power factor choice for controlling any index of generator
        load,or storage unit elements. It contains the following parameters:
    s_nom : pandas data frame
        Inverter nominal apparent power.
    set_p1 : pandas data frame
        It is a set point  in percentage,  where it tells the controller to work
        with unity power factor if  (injected_power / s_nom)*100  <  set_p1.
    set_p2 : pandas data frame
        It is a set point in percentage,  where it tells controller to work with
        'power_factor_min' if  (injected_power / s_nom)*100   >  set_p2.
    power_factor_min : pandas data frame
        Minimum allowed power factor.
    p_set_per_s_nom : pandas data frame
        Inverter real power injection percentage or (p_set / s_nom)*100.

    Returns
    -------
    q_set : pandas data frame
        Is the new reactive power that will be set as new q_set in each
        controlled component as a result of applying this controller.
    """
    set_p1 = parameters['set_p1']
    set_p2 = parameters['set_p2']
    s_nom = parameters['s_nom']
    power_factor_min = parameters['power_factor_min']
    p_set_per_s_nom = (abs(p_input.loc[now, parameters.index]) / abs(s_nom))*100

    # pf allocation using np.select([condtions...], [choices...]) function.
    power_factor = np.select([(p_set_per_s_nom < set_p1), (
        p_set_per_s_nom >= set_p1) & (p_set_per_s_nom <= set_p2), (
            p_set_per_s_nom > set_p2)], [1 ,(1 - ((1 - power_factor_min) / (
             set_p2 - set_p1) * (p_set_per_s_nom - set_p1))), power_factor_min])

    q_set = np.where(power_factor == 1, 0, -p_input.loc[
        now, parameters.index].mul(np.tan((np.arccos(power_factor)))))
    if p_set_varies_per_snapshot:
        df_t.power_factor.loc[now, parameters.index] = power_factor
    else:
        df.loc[parameters.index, 'power_factor'] = power_factor

    return q_set


def q_v(now, n_trials_max, n_trials, p_input, n, c_list_name, parameters):
    """
    Reactive power as a function of voltage Q(U).
    reference : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6096349

    Parameter:
    ----------
    now : single snaphot
        Current  element of n.snapshots on which the power flow is run.
    n_trials_max : integer
        It is the max number of outer loop (while loop in pf.py) trials until
        the controller converges.
    n_trials : integer
        It is the outer loop (while loop in pf.py) number of trials until
        the controller converges.
    p_input : pandas data frame
        Input active power for controller, comming from n.component_t.p.
    n : pypsa.components.Network
        Network
    c_list_name : string
        Name of component, could be; 'loads', 'generators', storage_units.
    parameters : pandas data frame
        Controller parameters chosen for this controller for each index of the
        elements, i.e. power factor choice for controlling any index of generator
        load,or storage unit elements. It contains the following parameters:
    s_nom : pandas data frame
        Inverter nominal apparent power.
    v1, v2 : pandas data frame
        v_mag_pu set points for the lower portion of the q_v curve, where v1
        is the minimum and v2 is the maximum v_mag_pu respectively.
    v3, v4 : pandas data frame
        v_mag_pu set points for the upper portion of the q_v curve, where
        v3 is the minimum and v4 is the maximum v_mag_pu respectively, in
        terms of maginitude v1 < v2 < v3 < v4.
    damper : pandas data frame
        It can take value (0-1) to controll controller output and help it
               to converge in case it enters to the infinit loop.
    bus : pandas data frame
        The bus where inverter is connected to.
    q_set_per_qmax : pandas data frame
        It is reactive power compensation in % out of maximum reactive
        power capability of inverter (q_max = np.sqrt(s_nom**2 - (p_set)**2).

    Returns
    -------
    q_set_out : pandas data frame
    Is the new reactive power that will be set as new q_set in each
        controlled component as the result of applying q_v controller.
    """
    if n_trials == n_trials_max:
        logger.warning("The voltage difference at snapshot ' %s' , in "
                       "components '%s', with 'q_v' controller exceeds "
                       "x_tol_outer limit, please apply (damper < 1) or"
                       "expand controller parameters range between v1 &"
                       " v2 and or v3 & v4  to avoid the problem." % (
                           now, parameters.index.values))

    v_pu_bus = n.buses_t.v_mag_pu.loc[now, parameters.loc[
                                               parameters.index, 'bus']].values
    # curve parameters
    v1 = parameters['v1']
    v2 = parameters['v2']
    v3 = parameters['v3']
    v4 = parameters['v4']

    # q_set/s_nom selection from choices np.select([conditions], [choices])
    q_set_per_qmax = np.select(
        [(v_pu_bus < v1), (v_pu_bus >= v1) & (v_pu_bus <= v2),
         (v_pu_bus > v2) & (v_pu_bus <= v3), (v_pu_bus > v3) & (v_pu_bus <= v4),
         (v_pu_bus > v4)], [100, 100 - 100 / (v2 - v1) * (v_pu_bus - v1), 0,
                            -100 * (v_pu_bus - v3) / (v4 - v3), -100])

    q_out = ((q_set_per_qmax * (np.sqrt(parameters['s_nom']**2 - (p_input.loc[
        now, parameters.index])**2))) / 100) * parameters['damper']
    q_set = np.where(c_list_name == 'loads', -q_out, q_out)

    return q_set


def apply_controller(n, now, n_trials, n_trials_max, parameter_dict):
    """
    Iterate over storage_units, loads and generators to to apply controller.

    Parameter:
    ----------
    n : pypsa.components.Network
        Network
    now : single snaphot
        Current  element of n.snapshots on which the power flow is run.
    n_trials : integer
        It is the outer loop (while loop in pf.py) number of trials until
        the controller converges.
    n_trials_max : integer
        It is the max number of outer loop (while loop in pf.py) trials until
        the controller converges.
    parameter_dict : dictionary
        It is a dynamic dictionary, meaning that its size and content depends on
        the number controllers chosen on number of components. i.e. if only 'q_v'
        controller on Load component is chosen, then dictionary will look like:
        Parameter_dict{'p_input': {'loads':n.loads_t.p }, controller_parameters:
        {'q_v':{'loads':n.loads, 'loads_t':n.loads}}}, 'v_dep_buses':{array of
        v_dependent bus names}, 'controlled_buses':{array of controlled buses
        names} and copy of the network as reference. Where n.loads is loads df
        that has p_set fixed and loads_t is loads df that has changing p_set in
        each snapshot. pramameter_dict exapands same way for the other
        controllers and components but its structure does not change.

    Returns
    -------
    v_mag_pu_voltage_dependent_controller : pandas data frame v_mag_pu of buses
    having voltage dependent controller attached to one oftheir inverters. This
    is important in order to be able to compare v_mag_pu withthe voltage from
    the next n_trial of the power flow to decide wether to repeat the power flow
    for the next iteration or not (in pf.py file).
    """
    v_mag_pu_voltage_dependent_controller = 0  # if no v_dep. controller is used
    # check if any controlled components exist in the parameter_dict
    if bool('controller_parameters' in parameter_dict):
        # loop over the exisitng controllers in parameter_dict
        for controller in parameter_dict['controller_parameters'].keys():
            # parameter is the controlled indexes dataframe of a components
            for component_name, parameter in parameter_dict[
                                  'controller_parameters'][controller].items():
                # determining the true component_name for each loop
                c_list_name = component_name.strip('_t')
                # input power for that component
                p_input = parameter_dict['P_input'][component_name.strip('_t')]
                # finding n.component.c_list_name and n.component.c_list_name_t
                df = getattr(n, c_list_name)
                df_t = getattr(n, c_list_name + '_t')
                # flag to check if the p_set of the component is static or series
                p_set_varies_per_snapshot = bool('_t' in component_name)

                # call each controller if it exist in parameter_dict
                if controller == 'q_v':
                    q_set = q_v(now, n_trials_max, n_trials, p_input, n,
                                c_list_name, parameter)

                if controller == 'fixed_cosphi':
                    q_set = fixed_cosphi(p_input, now, parameter)

                if controller == 'cosphi_p':
                    q_set = cosphi_p(p_input, now, parameter,
                                     p_set_varies_per_snapshot, df, df_t)
                # add the controller effect to the n or pf
                _set_change_in_q_set_to_n(n, parameter_dict['n_copied'],
                                          p_set_varies_per_snapshot, parameter,
                                          df, df_t, c_list_name, q_set, now,
                                          parameter_dict['controlled_buses'])
        # find the v_mag_pu of buses with v_dependent controller to return
        v_mag_pu_voltage_dependent_controller = n.buses_t.v_mag_pu.loc[
             now, parameter_dict['v_dep_buses']]

    return v_mag_pu_voltage_dependent_controller, n_trials+1


def prepare_dict_values(
               parameter_dict, c_list_name, c_df, c_pnl, ctrl_list, controller):
    """
    Add parameters of the given controlled components to the given parameter_dict.

    Parameters
    ----------
    parameter_dict : dictionary
        Dictonary to add keys and values of the given component and controller.
    c_list_name : string
        Component name, i.e. 'loads', 'storage_units', 'generators'.
    c_df : pandas data frame
        DataFrame of static components for c_list_name, i.e. network.generators
    c_pnl : pandas data frame
        dictionary of DataFrames of varying components for c_list_name,
        i.e. network.generators_t.
    ctrl_list : list
        List of supported controllers ['q_v', 'cosphi_p', 'fixed_cosphi'].
    controller : pandas series
        Type of controller can be any of them in ctrl_list.

    Returns
    -------
    parameter_dict : dictionary
        All needed parameters for the chosen controller.
    """
    # assert error when any of the controllers is not in the controlled list
    assert (controller.isin(ctrl_list)).all(), (
        "Not all given types of controllers are supported. Elements with unknown"
        "controllers are:\n%s\nSupported controllers are : %s." % (c_df.loc[
            (~ c_df['type_of_control_strategy'].isin(ctrl_list)),
            'type_of_control_strategy'], ctrl_list))

    if 'controller_parameters' not in parameter_dict:
        parameter_dict['controller_parameters'] = {}

    if 'P_input' not in parameter_dict:
        parameter_dict['P_input'] = {}
    # storing input power for each component
    parameter_dict['P_input'][c_list_name] = c_pnl.p

    for i in ctrl_list[1:4]:
        # building a dictionary for each controller
        if i not in parameter_dict['controller_parameters']:
            parameter_dict['controller_parameters'][i] = {}

        # storing parameters of the component for indexes with static p_set
        if c_df[c_df.type_of_control_strategy == i].index.isin(c_pnl.p_set).any():
            parameter_dict['controller_parameters'][i][c_list_name + '_t'] = \
                c_df.loc[(c_df.index.isin(c_pnl.p_set) & (controller == i))]

        # storing parameters of the component for indexes with static p_set
        if ~(c_df[c_df.type_of_control_strategy == i].index).isin(c_pnl.p_set).all():
            parameter_dict['controller_parameters'][i][c_list_name] = c_df.loc[
                           (~c_df.index.isin(c_pnl.p_set) & (controller == i))]

    # reindexing n.component_t.power_factor fata frame
    c_pnl.power_factor = c_pnl.power_factor.reindex(c_df.loc[(c_df.index.isin(
                     c_pnl.p_set) & (controller == 'cosphi_p'))].index, axis=1)
    # reindexing n.component_t.q_set
    c_pnl.q_set = c_pnl.q_set.reindex((c_df.loc[(c_df.index.isin(
             c_pnl.p_set) & (controller.isin(ctrl_list[1:4])))]).index, axis=1)

    return parameter_dict


def prepare_controller_parameter_dict(n, sub_network):
    """
    Iterate over components (loads, storage_units, generators) and check if they
    are controlled. For all controlled components, the respective controller
    parameters are stored in the dictionary parameter_dict.
    Returns
    -------
    n_trials_max:
        Maximum number of outer loops, dependent on the existance of a voltage
        dependent controller: 1 if not present or 20 if present
    parameter_dict : dictionary
        All needed controller parameters, (built in prepare_dict_values()).
    """
    # defining initial variable values and status
    controller_present = False
    n_trials_max = 0
    v_dep_buses = controlled_buses = np.array([])
    parameter_dict = {}
    ctrl_list = ['', 'q_v', 'cosphi_p', 'fixed_cosphi']

    # loop through loads, generators, storage_units and stores if they exist
    for c in sub_network.iterate_components(n.controllable_one_port_components):

        if (c.df.type_of_control_strategy != '').any():
            controller_present = True
            controller = c.df['type_of_control_strategy']
            # exclude slack generator to be controlled
            if c.list_name == 'generators':
                c.df.loc[c.df.control == 'Slack', 'type_of_control_strategy'] = ''
            # if voltage dep. controller exist,find the bus name
            if c.df.type_of_control_strategy.isin(['q_v']).any():
                v_dep_buses = np.append(v_dep_buses, np.unique(c.df.loc[(
                                            controller.isin(['q_v'])), 'bus']))
                n_trials_max = 30  # max pf repeatation for controller convergance

            # find all controlled bus names
            controlled_buses = np.append(controlled_buses, np.unique(
                           c.df.loc[(controller.isin(ctrl_list[1:4])), 'bus']))

            # call the function to prepare the controller dictionary
            parameter_dict = prepare_dict_values(
                parameter_dict, c.list_name, c.df, c.pnl, ctrl_list, controller)

            logger.info("We are in %s. That's the parameter dict:\n%s", c.name,
                        parameter_dict)

    parameter_dict['v_dep_buses'] = v_dep_buses  # names of v_dependent buses
    parameter_dict['controlled_buses'] = controlled_buses  # names buses
    parameter_dict['n_copied'] = deepcopy(n)  # deepcopy of n as reference

    return n_trials_max, parameter_dict, controller_present


def _set_change_in_q_set_to_n(n, n_copy, p_set_varies_per_snapshot, parameter,
                              df, df_t, c_list_name, q_set, now, controlled_buses):
    """
    Iterate over storage_units, loads and generators to to apply controller.

    Parameter:
    ----------
    n : pypsa.components.Network
        Network
    n_copy : pypsa.components.Network
        a deepcopy of n, required to add the change in q on it.
    p_set_varies_per_snapshot : bool(True / False)
        True if component has p_set that varies per snapshot, else False.
    parameter : pandas data frame
        Contains the controlled indexes of a component with its all attributes.
    df : pandas data frame
        Data frame of components, i.e. n.loads / generators / storage_units
    df_t : pandas data frame
        Time varying data frames of a component, i.e. n.loads_t / generators_t.
    c_list_name : string
        Component name, i.e. loads, storage_units, generators.
    q_set : pandas data frame
        Data frame of new q_set as controller output.
    now : single snapshot
        Current  element of n.snapshots on which the power flow is run.
    controlled_buses : numpy array
        Names of the buses which controller is applied on, required to add the
        changes in q on them.

    Returns
    -------
    None
    """
    # set change in q_set on n.component_t.q_set or n.component.q_set dataframes
    if p_set_varies_per_snapshot:
        df_t.q_set.loc[now, parameter.index] = q_set
    else:
        df.loc[parameter.index, 'q_set'] = q_set
    # set change in q on n.component_t.q
    df_t.q.loc[now, parameter.index] = getattr(
        n_copy, c_list_name + '_t').q.loc[now, parameter.index] + q_set
    # find change in q for buses
    q_change_in_buses = sum([((df_t['q'].loc[now, parameter.index] * df.loc[
        parameter.index, 'sign']).groupby(df.loc[parameter.index, 'bus']).sum()
                           .reindex(controlled_buses, axis=1, fill_value=0.))])
    # set the change in q to controlled buses in pf
    n.buses_t.q.loc[now, controlled_buses] = n_copy.buses_t.q.loc[
                                     now, controlled_buses] + q_change_in_buses
