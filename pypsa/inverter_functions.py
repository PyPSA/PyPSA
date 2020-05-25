"""importing important libraries."""
import numpy as np
import logging
import pandas as pd
logger = logging.getLogger(__name__)


def fixed_cosphi(p_input, now, parameters):
    """
    Fixed Power factor (fixed_cosphi): Sets the new q_set for controlled
    components according to the real power input and power factor given.
    reference : https://www.academia.edu/24772355/
    """

    q_set = -p_input.loc[now, parameters['power_factor'].index].mul(
        np.tan(np.arccos(parameters['power_factor'])))
    return q_set


def cosphi_p(p_input, now, parameters, p_set_varies, df):
    """
    Power factor as a function of real power (cosphi_p) method. It sets new
    power factor and q_set values based on the amount of power injection.

    Parameter:
    ----------

    parameters : pandas data frame
        It contains parameters(s_nom, set_p1, set_p2, power_factor_min).
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
        controlled component as the result of applying cosphi_p controller.
    pf : pandas data frame
    power factor (pf) that will be set as new pf value in each controlled
        component as the result of applying cosphi_p controller.
    ref : https://ieeexplore.ieee.org/document/6096349
    """
    set_p1 = parameters['set_p1']
    set_p2 = parameters['set_p2']
    s_nom = parameters['s_nom']
    power_factor_min = parameters['power_factor_min']
    p_set_per_s_nom = (abs(p_input.loc[now, parameters.index]) / abs(s_nom)) * 100

    # pf allocation using np.select([condtions...], [choices...]) function.
    power_factor = np.select([(p_set_per_s_nom < set_p1),
                              (p_set_per_s_nom >= set_p1) & (p_set_per_s_nom <= set_p2),
                              (p_set_per_s_nom > set_p2)],
                             [1,
                              (1 - ((1 - power_factor_min) / (set_p2 - set_p1) * (p_set_per_s_nom - set_p1))),
                              power_factor_min])

    q_set = np.where(power_factor == 1, 0, -p_input.loc[
        now, parameters.index].mul(np.tan((np.arccos(power_factor)))))
    if p_set_varies:
        df.power_factor.loc[now, parameters.index] = power_factor
    else:
        df.loc[parameters.index, 'power_factor'] = power_factor

    return q_set


def q_v(now, n_trials_max, n_trials, p_input, n, component_name, parameters):
    """
    Reactive power as a function of voltage Q(U): In this strategy controller
    finds the amount of inverter reactive power capability according to the
    amount of power injection and then compensates reactive power based on
    v_mag_pu of the bus where inverter is connected.

    Parameter:
    ----------

    parameters : pandas data frame
        It contains the following parameters ('s_nom', 'v1', 'v2', 'v3', 'v4',
                                                           'damper','bus').
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
    ref : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6096349

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
    # Parameter
    v1 = parameters['v1']
    v2 = parameters['v2']
    v3 = parameters['v3']
    v4 = parameters['v4']

    # q_set/s_nom selection from choices np.select([conditions], [choices])
    q_set_per_qmax = np.select([(v_pu_bus < v1),
                                (v_pu_bus >= v1) & (v_pu_bus <= v2),
                                (v_pu_bus > v2) & (v_pu_bus <= v3),
                                (v_pu_bus > v3) & (v_pu_bus <= v4),
                                (v_pu_bus > v4)],
                               [100, 100 - 100 / (v2 - v1) * (v_pu_bus - v1),
                                0, -100 * (v_pu_bus - v3) / (v4 - v3), -100])

    q_out = ((q_set_per_qmax * (np.sqrt(parameters['s_nom']**2 - (p_input.loc[
        now, parameters.index])**2))) / 100) * parameters['damper']
    q_set = np.where(component_name.strip('_t') == 'loads', -q_out, q_out)

    return q_set


def apply_controller(n, now, n_trials, n_trials_max, parameter_dict):
    """
    Iterate over storage_units, loads and generators to to apply controoler.

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
        the number controllers chosen on number components. Eg; if only 'q_v'
        controller on Load component is chosen, then dictionary will look like:
        Parameter_dict{'p_input': {'loads':n.loads_t.p }, controller_parameters:
        {'q_v':{'loads':n.loads, 'loads_t':n.loads}}}, 'v_dep_buses':{array of
        v_dependent bus names}}. Where n.loads is loads df that has p_set fixed
        and loads_t is loads df that has changing p_set in each snapshot.
        pramameter_dict exapands same way for the other controllers and
        components but its structure does not change.

    Returns
    -------
    v_mag_pu_voltage_dependent_controller : pandas data frame
        v_mag_pu of buses having voltage dependent controller attached to one of
        their inverters. This is important in order to be able to v_mag_pu with
        the voltage from the next n_trial of the power flow to decide wether to
        repeat the power flow  for the next iteration or not (in pf.py file).
    """
    if bool('controller_parameters' in parameter_dict):
        for controller in parameter_dict['controller_parameters'].keys():
            for component_name, parameter in parameter_dict[
                    'controller_parameters'][controller].items():

                p_input = parameter_dict['P_input'][component_name.strip('_t')]
                df = getattr(n, component_name)
                p_set_varies = bool('_t' in component_name)

                if controller == 'q_v':
                    q_set = q_v(now, n_trials_max, n_trials, p_input, n,
                                component_name, parameter)

                if controller == 'fixed_cosphi':
                    q_set = fixed_cosphi(p_input, now, parameter)

                if controller == 'cosphi_p':
                    q_set = cosphi_p(p_input, now, parameter, p_set_varies, df)

            if p_set_varies:
                df.q_set.loc[now, parameter.index] = q_set
            else:
                df.loc[parameter.index, 'q_set'] = q_set

        v_mag_pu_voltage_dependent_controller = n.buses_t.v_mag_pu.loc[
                                            now, parameter_dict['v_dep_buses']]

        return v_mag_pu_voltage_dependent_controller


def prepare_dict_values(parameter_dict, comp, df, df_t, ctrl_list, controller):
    """
    Add parameters of the given controlled components to the given parameter_dict

    Parameters
    ----------
    parameter_dict : dictionary
        Dictonary to add keys and values of the given component and controller.
    comp : string
        Comp is component name, eg: 'loads', 'storage_units', 'generators'.
    df : pandas data frame
        Component data frame, eg: n.loads, n.generators, n.storage_units.
    df_t : pandas data frame
        Component time var data frame, eg; n.loads_t, n.generator_t.
    ctrl_list : list
        List of supported controllers ['q_v', 'cosphi_p', 'fixed_cosphi'].
    controller : pandas data frame
        Type of controller can be any of them in ctrl_list.

    Returns
    -------
    parameter_dict : dictionary
        All needed parameters for the chosen controller.
    """
    assert (controller.isin(ctrl_list)).all(), (
        "Not all given types of controllers are supported. "
        "Elements with unknown controllers are:\n%s\nSupported controllers are"
        ": %s." % (df.loc[(~ df['type_of_control_strategy'].isin(ctrl_list)),
                          'type_of_control_strategy'], ctrl_list))
    if 'controller_parameters' not in parameter_dict:
        parameter_dict['controller_parameters'] = {}
    if 'P_input' not in parameter_dict:
        parameter_dict['P_input'] = {}
    parameter_dict['P_input'][comp] = df_t.p

    for c in ctrl_list[1:4]:
        if (df.type_of_control_strategy == c).any():

            if c not in parameter_dict['controller_parameters']:
                parameter_dict['controller_parameters'][c] = {}

            if df[df.type_of_control_strategy == c].index.isin(df_t.p_set).any():
                parameter_dict['controller_parameters'][c][comp + '_t'] = df.loc[
                                (df.index.isin(df_t.p_set) & (controller == c))]

            if ~(df[df.type_of_control_strategy == c].index).isin(df_t.p_set).all():
                parameter_dict['controller_parameters'][c][comp] = df.loc[
                               (~df.index.isin(df_t.p_set) & (controller == c))]

    df_t.power_factor = df_t.power_factor.reindex(df.loc[
        (df.index.isin(df_t.p_set) & (controller == 'cosphi_p'))].index, axis=1)

    df_t.q_set = df_t.q_set.reindex((df.loc[(df.index.isin(df_t.p_set) & (
                              controller.isin(ctrl_list[1:4])))]).index, axis=1)
    return parameter_dict


def prepare_controller_parameter_dict(n):
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
        All needed parameters for the chosen controller. (built in prepare_dict_values()).
    """
    components = ['loads', 'storage_units', 'generators']
    ctrl_list = ['', 'q_v', 'cosphi_p', 'fixed_cosphi']
    parameter_dict = {}
    parameter_dict['v_dep_buses'] = {}
    n_trials_max = 1

    for comp in components:
        df = getattr(n, comp)
        df_t = getattr(n, comp + '_t')
        controller = df['type_of_control_strategy']
        if (df.type_of_control_strategy != '').any():
            parameter_dict = prepare_dict_values(
                         parameter_dict, comp, df, df_t, ctrl_list, controller)
        if (df.type_of_control_strategy == 'q_v').any():
            # voltage dependent coltroller is present, activate the outer loop
            n_trials_max = 20
        logger.info("We are in %s. That's the parameter dict:\n%s", comp, parameter_dict)

    if n_trials_max > 1:
        parameter_dict['v_dep_buses'] = np.unique(pd.concat(
                 parameter_dict['controller_parameters']['q_v'])['bus'].values)

    return n_trials_max, parameter_dict
