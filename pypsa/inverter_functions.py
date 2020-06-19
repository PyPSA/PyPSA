"""importing important libraries."""
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
logger = logging.getLogger(__name__)


def fixed_cosphi(p_input, now, c_attrs):
    """
    fix power factor inverter controller.
    reference : https://www.academia.edu/24772355/

    Parameters
    ----------
    p_input : pandas data frame
        Input active power for controller, comming from n.component_t.p.
    now : single snapshot
        Current  element of n.snapshots on which the power flow is run.
    c_attrs : pandas data frame
        Component attrs including controller required parameters for controlled
        indexes, i.e. power_factor choice for generators, loads...

    Returns
    -------
    q_set : pandas data frame
        Is the new reactive power that will be set as new q_set in each
        controlled component as a result of applying this controller.
    """
    # calculation of q based on provided power_factor
    q_set = -p_input.loc[now, c_attrs['power_factor'].index].mul(np.tan(
        np.arccos(c_attrs['power_factor'], dtype=np.float64), dtype=np.float64))

    return q_set


def cosphi_p(p_input, now, df, df_t, c_attrs, time_varying_p_set):
    """
    Power factor as a function of active power (cosphi_p) controller.
    reference : https://ieeexplore.ieee.org/document/6096349.

    Parameters
    ----------
    p_input : pandas data frame
        Input active power for controller, comming from n.component_t.p.
    now : single snapshot
        Current  element of n.snapshots on which the power flow is run.
    df : pandas data frame
        Component data frame, i.e. n.loads, n.storage_units...
    df_t : pandas data frame
        Component data frame, i.e. n.loads_t, n.storage_units_t...
    time_varying_p_set : bool (True / False)
        It determines if p_set is static (False) or series (True).
    c_attrs : pandas data frame
        Component attrs including controller required parameters for controlled
        indexes, i.e. s_nom, set_p1, set_p2, power_factor_min choice for
        generators, loads...
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
    # parameters needed
    set_p1 = c_attrs['set_p1']
    set_p2 = c_attrs['set_p2']
    s_nom = c_attrs['s_nom']
    power_factor_min = c_attrs['power_factor_min']
    p_set_per_s_nom = (abs(p_input.loc[now, c_attrs.index]) / abs(s_nom))*100

    # pf allocation using np.select([condtions...], [choices...]) function.
    power_factor = np.select([(p_set_per_s_nom < set_p1), (
        p_set_per_s_nom >= set_p1) & (p_set_per_s_nom <= set_p2), (
            p_set_per_s_nom > set_p2)], [1, (1 - ((1 - power_factor_min) / (
             set_p2 - set_p1) * (p_set_per_s_nom - set_p1))), power_factor_min])

    # find q_set and avoid -0 apperance as the output when power_factor = 1
    q_set = np.where(power_factor == 1, 0, -p_input.loc[
            now, c_attrs.index].mul(np.tan((np.arccos(
                          power_factor, dtype=np.float64)), dtype=np.float64)))

    # setting the power factor value to n.components and n.components_t
    if time_varying_p_set:
        df_t.power_factor.loc[now, c_attrs.index] = power_factor
    else:
        df.loc[c_attrs.index, 'power_factor'] = power_factor

    return q_set


def q_v(c_list_name, now, n_trials_max, n_trials, p_input, v_pu_bus, c_attrs):
    """
    Reactive power as a function of voltage Q(U).
    reference : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6096349

    Parameter:
    ----------
    c_list_name : sring
        Component name, i.e. generators, loads...
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
    c_attrs : pandas data frame
        Component attrs including controller required parameters for controlled
        indexes, i.e. s_nom, v1, v2, v3, v4, damper, bus, generators, loads...
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

    Returns
    -------
    q_set_out : pandas data frame
    Is the new reactive power that will be set as new q_set in each
        controlled component as the result of applying q_v controller.
    """
    if n_trials == n_trials_max:
        logger.warning("The voltage difference at snapshot ' %s', in components"
                       " '%s', with 'q_v' controller exceeds x_tol_outer limit,"
                       " please apply (damper < 1) or expand controller"
                       " parameters range between v1 & v2 and or v3 & v4 to"
                       " avoid the problem." % (now, c_attrs.index.values))
    #  curve parameters
    v_pu_bus = v_pu_bus.loc[now, c_attrs.loc[c_attrs.index, 'bus']].values
    v1 = c_attrs['v1']
    v2 = c_attrs['v2']
    v3 = c_attrs['v3']
    v4 = c_attrs['v4']
    s_nom = c_attrs['s_nom']
    p = p_input.loc[now, c_attrs.index]

    # calculation of maximum q compensation in % based on bus v_pu_bus
    curve_q_set_in_percentage = np.select([(v_pu_bus < v1), (v_pu_bus >= v1) & (
            v_pu_bus <= v2), (v_pu_bus > v2) & (v_pu_bus <= v3), (v_pu_bus > v3)
        & (v_pu_bus <= v4), (v_pu_bus > v4)], [100, 100 - 100 / (v2 - v1) * (
                v_pu_bus - v1), 0, -100 * (v_pu_bus - v3) / (v4 - v3), -100])
    # find max q according to power factor and p_set
    Q_max = p.mul(np.tan((np.arccos(c_attrs[
                        'power_factor'], dtype=np.float64)), dtype=np.float64))
    # find inverter q capacity according to power factor provided
    Q_inv_cap = s_nom*np.sin(np.arccos(c_attrs['power_factor'],
                                       dtype=np.float64), dtype=np.float64)
    # find max allowable q that is possible based on s_nom
    Q_allowable = np.where(Q_max <= Q_inv_cap, Q_max, Q_inv_cap)
    # find amount of q_set compensation according to bus v_mag_puz
    q_set = (((curve_q_set_in_percentage * Q_allowable) / 100) * c_attrs[
                                                   'damper'] * c_attrs['sign'])
    # check if there is need to reduce p_set due to q need
    if (Q_max > Q_inv_cap).any().any():
        setting_p_set_required = True
        adjusted_p_set = np.sqrt((s_nom**2 - q_set**2),  dtype=np.float64)
        new_p_set = np.where(p <= adjusted_p_set, p, adjusted_p_set)
        logger.info(" Some p_set in %s component in q_v controller are adjusted"
                    " according to s_nom and power_factor chosen.", c_list_name)
    else:
        new_p_set = None
        setting_p_set_required = False

    return q_set, new_p_set, setting_p_set_required


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
        {'q_v':{'loads':n.loads, 'loads_t':n.loads}}, 'v_dep_buses':{array of
        v_dependent bus names}, 'deepcopy_buses_t': {deepcopy(n.buses_t)}}}.
        Note:In {'q_v':{'loads':n.loads, 'loads_t':n.loads}} the dataframes are
        divided to two parts 1- dataframe which has static p_set (q_v['loads'])
        2- dataframe which has series p_set (q_v['loads_t']),  same for other
        controllers as well if they are chosen.

    Returns
    -------
    v_mag_pu of voltage_dependent_controller : pandas data frame
    Needed to compare v_mag_pu of the controlled buses with the voltage from
    previous iteration to decide for repeation of pf (in pf.py file).
    """
    for controller in parameter_dict['controller_parameters'].keys():
        # parameter is the controlled indexes dataframe of a components
        for component_name, c_attrs in parameter_dict[
                              'controller_parameters'][controller].items():

            # determine component name, p_input and required dataframes
            c_list_name = component_name.strip('_t')
            p_input = parameter_dict['deepcopy_component_t'][c_list_name]
            df = getattr(n, c_list_name)
            df_t = getattr(n, c_list_name + '_t')
            # flag to check if the p_set of the component is static or series
            time_varying_p_set = bool('_t' in component_name)

            # call each controller if it exist in parameter_dict
            setting_p_set_required = False  # initial flag for setting p_set
            p_set = None  # initial when setting_p_set_required is False
            if controller == 'fixed_cosphi':
                q_set = fixed_cosphi(p_input.p, now, c_attrs)

            if controller == 'cosphi_p':
                q_set = cosphi_p(
                        p_input.p, now, df, df_t, c_attrs, time_varying_p_set)

            if controller == 'q_v':
                q_set, p_set, setting_p_set_required = q_v(
                        c_list_name, now, n_trials_max, n_trials, p_input.p,
                        n.buses_t.v_mag_pu, c_attrs)

            # sett the controller output to the network
            _set_controller_outputs_to_n(
                n, parameter_dict, time_varying_p_set, c_attrs, df, df_t, q_set,
                p_set, now, setting_p_set_required, p_input)
    # find the v_mag_pu of buses with v_dependent controller to return
    v_mag_pu_voltage_dependent_controller = n.buses_t.v_mag_pu.loc[
        now, parameter_dict['v_dep_buses']]

    return v_mag_pu_voltage_dependent_controller


def _set_controller_outputs_to_n(n, parameter_dict, time_varying_p_set, c_attrs,
                                 df, df_t, q_set, p_set, now,
                                 setting_p_set_required, p_input):
    """
    Set the controller outputs to the n (network).

    Parameter:
    ----------
    n : pypsa.components.Network
        Network
    parameter_dict : Dictionary
        Dictionary of required controller parameters.
    time_varying_p_set : bool (True / False)
        It determines if p_set is static (False) or series (True).
        the controller converges.
    c_attrs : pandas data frame
        Component attrs including controller required parameters for controlled
        indexes, i.e. p_set, q_set, sign, bus, generators, loads...
    df : pandas data frame
        Component data frame, i.e. n.loads, n.storage_units...
    df_t : pandas data frame
        Component data frame, i.e. n.loads_t, n.storage_units_t...
    q_set : pandas series
        Reactive power componsation output as return value from controllers.
    p_set : pandas series
        Active power new values output from q_v controller, due to more q_set
        need, p_set has values only if setting_p_set_required is True.
    now : single snaphot
        Current  element of n.snapshots on which the power flow is run.
    p_input : pypsa.descriptors.Dict
        deepcopy of n.components_t values as a reference for calcultation.

    Returns
    -------
    None
    """
    p_q_dict = {'q': q_set}
    if setting_p_set_required:
        p_q_dict['p'] = p_set

    # setting p_set, q_set, p and q values to their respective dataframes
    for attr in p_q_dict.keys():
        df_t[attr].loc[now, c_attrs.index] = p_q_dict[attr]

        if time_varying_p_set:
            df_t[attr + '_set'].loc[now, c_attrs.index] = p_q_dict[attr]
        else:
            df.loc[c_attrs.index, attr + '_set'] = p_q_dict[attr]

        # Finding the change in p and q for the connected buses
        if attr == 'q':
            power_change = (df_t.q.loc[now, c_attrs.index] * c_attrs.loc[
                c_attrs.index, 'sign']).groupby(c_attrs.loc[
                        c_attrs.index, 'bus']).sum()
        if attr == 'p':
            power_change = -((p_input.p - df_t.p).loc[now, c_attrs.index] *
                             c_attrs.loc[c_attrs.index, 'sign']).groupby(
                                 c_attrs.loc[c_attrs.index, 'bus']).sum()

        # adding the change to the respective buses
        n.buses_t[attr].loc[now, power_change.index] = parameter_dict[
          'deepcopy_buses_t'][attr].loc[now, power_change.index] + power_change


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

    if 'deep_copy_component_t' not in parameter_dict:
        parameter_dict['deepcopy_component_t'] = {}
    # storing input power for each component
    parameter_dict['deepcopy_component_t'][c_list_name] = deepcopy(c_pnl)
    # storing deepcopy of n.component_t for each component as a reference, is
    # needed because Q(U) controller can also change p_set when more q is
    # needed, and this causes a change in the input power for Q(U) if multiple
    # iteration occurs per load flow which is mostly the case for Q(U).
    for i in ctrl_list[1:4]:
        # building a dictionary for each controller if they exist
        if (c_df.type_of_control_strategy == i).any():
            if i not in parameter_dict['controller_parameters']:
                parameter_dict['controller_parameters'][i] = {}

            # storing parameters of the component for indexes with static p_set
            if c_df[c_df.type_of_control_strategy == i].index.isin(c_pnl.p_set).any():
                parameter_dict['controller_parameters'][i][c_list_name + '_t'] = \
                    c_df.loc[(c_df.index.isin(c_pnl.p_set) & (controller == i))]

            # storing parameters of the component for indexes with static p_set
            if ~(c_df[c_df.type_of_control_strategy == i].index).isin(c_pnl.p_set).all():
                parameter_dict['controller_parameters'][i][c_list_name] = \
                    c_df.loc[(~c_df.index.isin(c_pnl.p_set) & (controller == i))]
    # reindexing n.component_t.power_factor fata frame
    c_pnl.power_factor = c_pnl.power_factor.reindex(c_df.loc[(c_df.index.isin(
                     c_pnl.p_set) & (controller == 'cosphi_p'))].index, axis=1)
    # reindexing n.component_t.q_set
    c_pnl.q_set = c_pnl.q_set.reindex((c_df.loc[(c_df.index.isin(
             c_pnl.p_set) & (controller.isin(ctrl_list[1:4])))]).index, axis=1)

    return parameter_dict


def prepare_controller_parameter_dict(n, sub_network, inverter_control):
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
    # defining initial variable values and status
    n_trials_max = 0
    v_dep_buses = np.array([])
    parameter_dict = {}
    ctrl_list = ['', 'q_v', 'cosphi_p', 'fixed_cosphi']
    if inverter_control:
        # loop through loads, generators, storage_units and stores if they exist
        for c in sub_network.iterate_components(n.controllable_one_port_components):

            if (c.df.type_of_control_strategy != '').any():
                controller = c.df['type_of_control_strategy']
                # exclude slack generator to be controlled
                if c.list_name == 'generators':
                    c.df.loc[c.df.control == 'Slack', 'type_of_control_strategy'] = ''
                # if voltage dep. controller exist,find the bus name
                if c.df.type_of_control_strategy.isin(['q_v']).any():
                    v_dep_buses = np.append(v_dep_buses, np.unique(c.df.loc[(
                            controller.isin(['q_v'])), 'bus']))
                    n_trials_max = 30  # max pf repeatation for controller convergance

                # call the function to prepare the controller dictionary
                parameter_dict = prepare_dict_values(parameter_dict, c.list_name,
                                                     c.df, c.pnl, ctrl_list,
                                                     controller)

                logger.info("We are in %s. That's the parameter dict:\n%s",
                            c.name, parameter_dict)

        parameter_dict['v_dep_buses'] = v_dep_buses  # names of v_dependent buses
        parameter_dict['deepcopy_buses_t'] = deepcopy(n.buses_t)
        # deepcopy of the buses is needed to have the actual status of bus p and q
        # as a reference, i.e. Q(u) can have multiple iterations per load flow
        # until it converges. once it is converged then we say:
        # n.buses_t.q or p = deepcopy(buses_t.p or q) + or - controller output
        # if we dont have the deep copy it will keep on adding or subtracting
        # in each iteration.

    return n_trials_max, parameter_dict
