import numpy as np
import logging
logger = logging.getLogger(__name__)


def fixed_cosphi(pf, p_set_input, now):
    '''
    Fixed Power factor (cosphi) control strategy
    Parameters
    ----------
    pf : pandas pandas data frame
        data frame of power factors with their component names.
    p_set_input : pandas dataframe
        Real power input for controller.
    now : single snapshot
        An current (happening) element of network.snapshots on which the power
        flow is run.

    Returns
    -------
    q_out : pandas data frame
        controller output which sets as the new q_set after applying controller.
    reference : https://www.academia.edu/24772355/
    '''
    q_out = -p_set_input.loc[now, pf.index].mul(np.tan((np.arccos(pf))))
    return q_out


def cosphi_p(parameters, p_set_input, now):
    '''
    Power factor as a function of real power control method: In this strategy
    Controller chooses the right power factor for reactive power compendation
    according to the amount of injected power, inverter capacity and the chosen
    set points (set_p1 and set_p2). eg.: if set_p1 = 40% and power injection
    percentage is 20% then controller takes no action and power factor = 1.

    Parameters
    ----------
    parameters : pandas data frame
        It contains the following parameters(sn, set_p1, set_p2, pf_min).
    sn : pandas data frame
        Inverter complex power capacity.
    set_p1 : pandas data frame
        The set point (%) in terms of inverter power injection (p_set/sn)*100
        where until inverter takes no action.
    set_p2 : pandas data frame
        The set point (%) in terms of inverter power injection (p_set/sn)*100
        where inverter works with the minimum allowed power factor (pf_min).
    pf_min : pandas data frame
        Minimum allowed power factor.
    p_set_input : pandas data frame
        Real power input to the controller.
    now : single snapshot
        An current (happening) element of network.snapshots on which the power
        flow is run.
    reference : https://ieeexplore.ieee.org/document/6096349

    Returns
    -------
    q_out : pandas data frmae
        controller output which sets as the new q_set after applying controller.

    '''
    if not parameters.empty:
        parameters['p_per_p_max'] = (abs(p_set_input.loc[now, parameters.index]
                                         ) / abs(parameters['sn']))*100
        # defining curve conditions
        condition1 = parameters['p_per_p_max'] < parameters['set_p1']
        condition2 = (parameters['p_per_p_max'] >= parameters['set_p1']) & (
                      parameters['p_per_p_max'] <= parameters['set_p2'])
        condition3 = parameters['p_per_p_max'] > parameters['set_p2']

        # defining curve choices
        parameters.loc[condition1, 'pf'] = 1
        parameters.loc[condition2, 'pf'] = 1 - ((1-parameters['pf_min']) / (
                 parameters['set_p2'] - parameters['set_p1'])*(parameters[
                    'p_per_p_max'] - parameters['set_p1']))
        parameters.loc[condition3, 'pf'] = parameters['pf_min']
        q_out = -p_set_input.loc[now, parameters.index].mul(
            np.tan((np.arccos(parameters['pf']))))

        return q_out, parameters['pf']
    else:
        pass


def q_v(parameters, p_set_input, now, v_pu_buses, component_type, n_trials):
    '''
    Reactive power as a function of voltage Q(U): In this strategy controller
    finds the amount of inverter reactive power capability according to the
    amount of power injection and then compensates reactive power based on
    v_mag_pu of the base where inverter is connected. eg.: if v_pu_bus = 1.05
    there would be maximum copensation according to inverter capability to
    reduce the voltage.

    Parameters
    ----------
    parameters : pandas data frame
        It contains the following parameters ('sn', 'v1', 'v2', 'v3', 'v4',
                                                              'damper','bus').
    sn : pandas data frame
        Inverter complex power capacity.
    v1, v2 : pandas data frame
        v_mag_pu set points for the lower portion of the controller curve, v1 is
        the minimum and v2 is the maximum v_mag_pu respectively.
    v3, v4 : pandas data frame
        v_mag_pu set points for the upper portion of the controller curve, v3 is
        is the minimum and v4 is the maximum v_mag_pu respectively, controller
        output would be different based on where the bus v_mag_pu is located,
        in terms of maginitude v1 < v2 < v3 < v4.
    damper : pandas data frame
        It can take value (0-1) to controll controller output and help it to
               converge since it is directly multiplied to the output. it can
               be used when the controller does not meet the while loop
               condition (while voltage_difference > 0.001 in pf.py file) and it
               enters into infinite loop. Although the infinite loop is limited
               to 20 iterations, but Adding damper will create fluctuations in
               the controller output and help to converge.
    bus : pandas data frame
        The bus where inverter is connected on.
    p_set_input : pandas data frame
        Real power input to the controller..
    now : single snapshot
        An current (happening) element of network.snapshots on which the power
        flow is run.
    v_pu_buses : pandas data frame
        Voltage per unit of all buses controlled by this control method.
    component_type : string
        The behavior of controller is different for storageUunit & Generator
        and Load components.
    n_trials : integer
        It shows the outer loop (while loop in pf.py) number of trials until
        the controller converges.
    reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6096349

    Returns
    -------
    q_out : pandas data frame
        controller output which sets as the new q_set after applying controller.

    '''
    if n_trials == 20:
        logger.warning("The voltage difference at snapshot ' %s' , in  "
                       "components '%s', with 'q_v' control exceeds x_tol_outer"
                       "limit, please apply (damper < 1) or expand controller"
                       "parameters range between v1 & v2 and v3 & v4  to avoid"
                       "the problem." % (now, parameters.index.values))

    if not parameters.empty:
        parameters['p_set'] = p_set_input.loc[now, parameters.index]
        parameters['v_pu_bus'] = v_pu_buses.loc[now, parameters.loc[
            parameters.index, 'bus']].values

        # defining curve conditions
        condition1 = parameters['v_pu_bus'] < parameters['v1']
        condition2 = (parameters['v_pu_bus'] >= parameters['v1']) & (
            parameters['v_pu_bus'] <= parameters['v2'])
        condition3 = (parameters['v_pu_bus'] > parameters['v2']) & (
            parameters['v_pu_bus'] <= parameters['v3'])
        condition4 = (parameters['v_pu_bus'] > parameters['v3']) & (
            parameters['v_pu_bus'] <= parameters['v4'])
        condition5 = parameters['v_pu_bus'] > parameters['v4']

        # defining curve choices
        parameters.loc[condition1, 'qbyqmax'] = 100
        parameters.loc[condition2, 'qbyqmax'] = (100 - (100-0) / (
            parameters['v2'] - parameters['v1'])*(
                parameters['v_pu_bus'] - parameters['v1']))

        parameters.loc[condition3, 'qbyqmax'] = 0
        parameters.loc[condition4, 'qbyqmax'] = -(100)*(
            parameters['v_pu_bus'] - parameters['v3']) / (
                parameters['v4'] - parameters['v3'])
        parameters.loc[condition5, 'qbyqmax'] = -100

        parameters['Q_max'] = np.sqrt(parameters['sn']**2 - (
            p_set_input.loc[now, parameters.index])**2)
        q_out = ((parameters['qbyqmax']*parameters['Q_max']
                  )/100)*parameters['damper']

        if component_type == 'loads':
            return -q_out
        else:
            return q_out
    else:
        pass


def apply_controller_to_components_df(component_df, component_df_t, v_pu_buses,
                                      now, n_trials, component_type):
    '''
    The function interates in components data frames and apply the right
    controller types to the controlled component.

    Parameters
    ----------
    component_df : pandas data frame
        Can be network.loads/storage_units/generators.
    component_df_t : pandas data frame
        can be network.loads_t/storage_units_t/generators_t
    v_pu_buses : pandas data frame
        v_mag_pu of all buses.
    now : single snapshot
        An current (happening) element of network.snapshots on which the power
        power flow is run.
    n_trials : integer
        It shows the outer loop (while loop in pf.py) number of trials until
        the controller converges.
    component_type : string
        can be 'loads', 'storage_units' or 'generators'.

    Returns
    -------
    bus_names : pandas data frame
        Name of all thoses buses which are controlled by voltage dependent
        controllers such as q_v. This output is needed to calculate their
        v_mag_pu.

    '''
    ctrl_list = ['', 'q_v', 'cosphi_p', 'fixed_cosphi']
    assert (component_df.type_of_control_strategy.any() in ctrl_list), (
        "The type of controllers %s you have typed in components %s, is not"
        " supported.\n Supported controllers are %s." % (component_df.loc[(
            ~ component_df['type_of_control_strategy'].isin(ctrl_list)),
            'type_of_control_strategy'].values, component_df.loc[(~component_df[
                'type_of_control_strategy'].isin(ctrl_list)),
                'type_of_control_strategy'].index.values, ctrl_list[1:4]))

    bus_names = np.unique(component_df.loc[(
        component_df['type_of_control_strategy'].isin(['q_v'])), 'bus'].values)
    p_set_input = component_df_t.p

    # complete df of controlled components, with varying p_set
    component_df_p_set_t = component_df.loc[(component_df.index).isin(
      component_df_t.p_set.columns) & (component_df['type_of_control_strategy']
                                       != ''), :]
    # adding column labels to q_set and power factor (pf) dfs
    component_df_t.q_set = component_df_t.q_set.reindex((
        component_df_p_set_t.loc[(component_df_p_set_t[
            'type_of_control_strategy'].isin(['q_v', 'fixed_cosphi', 'cosphi_p'
                                              ]))].index), axis=1, fill_value=0)
    component_df_t.pf = component_df_t.pf.reindex((component_df_p_set_t.loc[(
        component_df_p_set_t['type_of_control_strategy'].isin(['cosphi_p'])
        )].index), axis=1, fill_value=0)

    # fixed_cosphi method
    component_df_t.q_set.loc[
        now, component_df_p_set_t.type_of_control_strategy == 'fixed_cosphi'
        ] = fixed_cosphi(component_df_p_set_t.loc[(
                    component_df_p_set_t['type_of_control_strategy'] ==
                    'fixed_cosphi'), 'pf'], p_set_input, now)
    components_with_fixed_p_set = component_df[(~(component_df.index).isin(
        component_df_t.p_set.columns)) & (
        component_df.type_of_control_strategy == 'fixed_cosphi')].index
    component_df.loc[components_with_fixed_p_set, 'q_set'] = fixed_cosphi(
        component_df.loc[components_with_fixed_p_set, 'pf'], p_set_input, now)

    #  cosphi_p method
    if not component_df_t.pf.empty:
        component_df_t.q_set.loc[
            now, component_df_p_set_t.type_of_control_strategy == 'cosphi_p'
            ], component_df_t.pf.loc[
                now, component_df_p_set_t.type_of_control_strategy == 'cosphi_p'
                ] = cosphi_p(component_df_p_set_t.loc[(component_df_p_set_t[
                    'type_of_control_strategy'] == 'cosphi_p'), (
                        'pf_min', 'sn', 'set_p1', 'set_p2')], p_set_input, now)

    components_with_fixed_p_set = component_df[(~(component_df.index).isin(
        component_df_t.p_set.columns)) & (component_df.type_of_control_strategy
                                          == 'cosphi_p')].index
    if not components_with_fixed_p_set.empty:
        component_df.loc[components_with_fixed_p_set, 'q_set'],
        component_df.loc[components_with_fixed_p_set, 'pf'] = cosphi_p(
            component_df.loc[components_with_fixed_p_set, (
                'pf_min', 'sn', 'set_p1', 'set_p2')], p_set_input, now)

    # Q(U) method
    component_df_t.q_set.loc[
        now, component_df_p_set_t.type_of_control_strategy == 'q_v'] = q_v(
            component_df_p_set_t.loc[(component_df_p_set_t[
                'type_of_control_strategy'] == 'q_v'), (
                    'sn', 'v1', 'v2', 'v3', 'v4', 'damper', 'bus')
                    ], p_set_input, now, v_pu_buses, component_type, n_trials)
    components_with_fixed_p_set = component_df[(~(component_df.index).isin(
        component_df_t.p_set.columns)) & (
            component_df.type_of_control_strategy == 'q_v')].index
    component_df.loc[components_with_fixed_p_set, 'q_set'] = q_v(
        component_df.loc[components_with_fixed_p_set, (
            'sn', 'v1', 'v2', 'v3', 'v4', 'damper', 'bus')], p_set_input,
        now, v_pu_buses, component_type, n_trials)

    return bus_names


def apply_controller(network, now, n_trials):
    '''
    This function iterates to storage_units, loads and generators to check
    if any controller is chosen to apply it to the component.

    Parameters
    ----------
    now : single snaphot
        An current (happening) element of network.snapshots on which the power
        flow is run.
    n_trials : TYPE
        It shows the outer loop (while loop in pf.py) number of trials until
        the controller converges.

    Returns
    -------
    v_mag_pu_voltage_dependent_controller : pandas data frame
        It is the data frame of v_mag_pu of those buses which they have voltage
        dependent controller attached to one of the inverters. This is important
        to compare the voltages with the voltage from the next n_trial and take
        their difference as a condition for the next iteration of the while
        loop in pf.py file.

    '''
    v_buses = network.buses_t.v_mag_pu
    bus_name_l = bus_name_g = bus_name_s = np.array([])

    if network.loads.loc[network.loads.type_of_control_strategy
                         != '', 'type_of_control_strategy'].any():
        bus_name_l = apply_controller_to_components_df(
            network.loads, network.loads_t,
            v_buses, now, n_trials, 'loads')

    if network.generators.loc[network.generators.type_of_control_strategy
                              != '', 'type_of_control_strategy'].any():
        bus_name_g = apply_controller_to_components_df(
            network.generators, network.generators_t,
            v_buses, now, n_trials, 'generators')

    if network.storage_units.loc[network.storage_units.type_of_control_strategy
                                 != '', 'type_of_control_strategy'].any():
        bus_name_s = apply_controller_to_components_df(
            network.storage_units, network.storage_units_t,
            v_buses, now, n_trials, 'storage_units')

    voltage_dependent_controller_bus_names = np.unique(np.concatenate((
        bus_name_l, bus_name_g, bus_name_s), axis=0))
    v_mag_pu_voltage_dependent_controller = v_buses.loc[
        now, voltage_dependent_controller_bus_names]

    return v_mag_pu_voltage_dependent_controller


def iterate_over_control_strategies(network):
    '''
    This method iterates in componets and checks if any voltage dependent
    controller is chosen or not. If any is chosen it will give an output which
    enables the power flow to repeat until the condition of while loop in pf.py
    is met.

    Parameters
    ----------
    Returns
    -------
    voltage_dependent_controller_present : bool, defaut False
        It will give a bool value (True/False) whether voltage dependent
        controller is present in the grid or not, if present True if not False.
    n_trials_max : integer
        This enables the power flow to repeat again in while loop inside
        pf.py file until either the while loop condition is met or n_trial_max
        is reached.

    '''
    n_trials_max = 20  # can be also an attribute of network
    voltage_dependent_controller_present = np.where(
        network.loads.type_of_control_strategy.isin(['q_v']).any(
        ) or network.generators.type_of_control_strategy.isin(['q_v']).any(
            ) or network.storage_units.type_of_control_strategy.isin(
                ['q_v']).any(), True, False)

    n_trials_max = np.where(voltage_dependent_controller_present,
                            n_trials_max, 1)
    return voltage_dependent_controller_present, n_trials_max
