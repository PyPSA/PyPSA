import numpy as np
import logging
logger = logging.getLogger(__name__)


def fixed_cosphi(p_input, now, component_df_p_set_var, component_df_t,
                 component_df):
    '''
    Fixed Power factor (fixed_cosphi) control strategy: Sets the new q_set
    according to the real power input and power factor given.

    Parameters
    ----------
    p_input : pandas data frame
        Real power input to the controller.
    now : single snapshot
        An instance of n.snapshots on which the power flow is run.
    component_df_p_set_var : pandas data frame
        Data frame of all componets which their p_set varies at each snapshot.
    component_df_t : pandas data frame
        Data frame of varying attrs. eg:- n.loads_t
    component_df : pandas data frame
        Data frame of component. eg:- n.loads
    reference : https://www.academia.edu/24772355/

    Returns
    -------
    None.
    '''
    # power factor (pf) data frame for indexes with fixed p_sets in snapshots
    pf_var_p_set = component_df_p_set_var.loc[(component_df_p_set_var[
        'type_of_control_strategy'] == 'fixed_cosphi'), 'pf']

    # power factor (pf) data frame for indexes with varying p_sets in snapshots
    pf_fixed_p_set = component_df.loc[(~(component_df.index).isin(
        component_df_t.p_set.columns)) & (component_df.type_of_control_strategy
                                          == 'fixed_cosphi'), 'pf']

    # setting the new q_sets for fixed p_sets indexes
    component_df_t.q_set.loc[now, pf_var_p_set.index] = -p_input.loc[
                now, pf_var_p_set.index].mul(np.tan((np.arccos(pf_var_p_set))))

    # setting the new q_sets for varying p_sets indexes
    component_df.loc[pf_fixed_p_set.index, 'q_set'] = -p_input.loc[
        now, pf_fixed_p_set.index].mul(np.tan((np.arccos(pf_fixed_p_set))))


def cosphi_p(p_input, now, component_df_p_set_var, component_df_t,
             component_df):
    '''
    Power factor as a function of real power (cosphi_p): This function sets
    new values to power factor (pf) and q_set based on chosen parameters in two
    conditions. 1. when p_set is chanaging in each snapshot and 2. when p_set is
    is fixed at any snapshot. In both of the cases (1, 2) calculation function
    is called based on the given parameters.

    Parameters
    ----------
    p_input : pandas data frame
        Real power input to the controller.
    now : single snapshot
        An instance of n.snapshots on which the power flow is run.
    component_df_p_set_var : pandas data frame
        Data frame of all componets which their p_set varies at each snapshot.
    component_df_t : pandas data frame
        Data frame of varying attrs. eg:- n.loads_t
    component_df : pandas data frame
        Data frame of component. eg:- n.loads
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

    reference : https://ieeexplore.ieee.org/document/6096349

    Returns
    -------
    None.

    '''
    def calculation(parameters):
        parameters['p_per_p_max'] = (abs(p_input.loc[now, parameters.index]) /
                                     abs(parameters['sn']))*100
        # defining curve conditions for power factor (pf) selection
        adopt_unity_pf = parameters['p_per_p_max'] < parameters['set_p1']
        adopt_non_unity_pf = (parameters['p_per_p_max'] >= parameters[
            'set_p1']) & (parameters['p_per_p_max'] <= parameters['set_p2'])
        adopt_min_pf = parameters['p_per_p_max'] > parameters['set_p2']

        # defining power factor (pf) choices for each conditions
        unity_pf = 1
        non_unity_pf = 1 - ((1-parameters['pf_min']) / (parameters['set_p2'] -
                            parameters['set_p1'])*(parameters['p_per_p_max'] -
                                                   parameters['set_p1']))
        min_pf = parameters['pf_min']

        # power factor (pf) allocation from choices based on conditions
        parameters['pf'] = np.select(
                            [adopt_unity_pf, adopt_non_unity_pf, adopt_min_pf],
                            [unity_pf, non_unity_pf, min_pf])

        q_out = -p_input.loc[now, parameters.index].mul(
            np.tan((np.arccos(parameters['pf']))))

        return q_out, parameters['pf']

    # parameters of indexes with varying p_set at each snapshot
    parameters_var_p_set = component_df_p_set_var.loc[(component_df_p_set_var[
        'type_of_control_strategy'] == 'cosphi_p'), (
            'pf_min', 'sn', 'set_p1', 'set_p2')]

    # parameters of indexes with fixed p_set at any snapshot
    parameters_fixed_p_set = component_df.loc[(~(component_df.index).isin(
        component_df_t.p_set.columns)) & (component_df.type_of_control_strategy
                                          == 'cosphi_p'), ('pf_min', 'sn',
                                                           'set_p1', 'set_p2')]
    # adding lables to n.components.pf dataframe
    component_df_t.pf = component_df_t.pf.reindex(
        parameters_var_p_set.index, axis=1, fill_value=0)

    # setting new q_set and power factor (pf) for varying p_set indexes
    component_df_t.q_set.loc[now, parameters_var_p_set.index
                             ], component_df_t.pf.loc[
        now, parameters_var_p_set.index] = calculation(parameters_var_p_set)

    # setting new q_set and power factor (pf) for fixed p_set indexes
    component_df.loc[parameters_fixed_p_set.index, 'q_set'], component_df.loc[
        parameters_fixed_p_set.index, 'pf'] = calculation(parameters_fixed_p_set)


def q_v(p_input, now, v_pu_buses, component_type, n_trials,
        component_df_p_set_var, component_df_t, component_df):
    '''
    Reactive power as a function of voltage Q(U): In this strategy controller
    finds the amount of inverter reactive power capability according to the
    amount of power injection and then compensates reactive power based on
    v_mag_pu of the base where inverter is connected. eg.: if v_pu_bus = 1.05
    there would be maximum copensation according to inverter capability to
    reduce the voltage.

    Parameters
    ----------
    p_input : pandas data frame
        Real power input to the controller.
    now : single snapshot
        An instance of n.snapshots on which the power flow is run.
    v_pu_buses : pandas data frame
        Voltage per unit of all buses controlled by this control method.
    component_type : string
        The behavior of controller is different for storageUunit & Generator
        and Load components.
    n_trials : integer
        It shows the outer loop (while loop in pf.py) number of trials until
        the controller converges.
    component_df_p_set_var : pandas data frame
        Data frame of all componets which their p_set varies at each snapshot.
    component_df_t : pandas data frame
        Data frame of varying attrs. eg:- n.loads_t
    component_df : pandas data frame
        Data frame of component. eg:- n.loads
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
    qbyqmax : pandas data frame
        It shows the reactive power compensation percentage based on maximum
        reactive power capability of inverter.
    Q_max : pandas data frame
        It is the maximum reactive power capability of the inverter.
    reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6096349
    Returns
    -------
    None.

    '''
    def calculation(parameters):
        if n_trials == 20:
            logger.warning("The voltage difference at snapshot ' %s' , in "
                           "components '%s', with 'q_v' control exceeds"
                           "x_tol_outer limit, please apply (damper < 1) or"
                           "expand controller parameters range between v1 & v2"
                           "and v3 & v4  to avoid the problem." % (
                               now, parameters.index.values))
        if not parameters.empty:
            parameters['p_set'] = p_input.loc[now, parameters.index]
            parameters['v_pu_bus'] = v_pu_buses.loc[now, parameters.loc[
                parameters.index, 'bus']].values

            # defining q_set selection conditions
            under_voltage = parameters['v_pu_bus'] < parameters['v1']
            voltage_is_low = (parameters['v_pu_bus'] >= parameters['v1']) & (
                parameters['v_pu_bus'] <= parameters['v2'])
            noramal_range = (parameters['v_pu_bus'] > parameters['v2']) & (
                parameters['v_pu_bus'] <= parameters['v3'])
            voltage_is_high = (parameters['v_pu_bus'] > parameters['v3']) & (
                parameters['v_pu_bus'] <= parameters['v4'])
            over_voltage = parameters['v_pu_bus'] > parameters['v4']

            # defining q_set choices in %
            for_under_voltage, for_over_voltage = (100, -100)
            for_noramal_range = 0
            for_low_voltage = (100 - (100-0) / (parameters['v2'] - parameters[
                'v1'])*(parameters['v_pu_bus'] - parameters['v1']))
            for_high_voltage = -(100)*(parameters['v_pu_bus'] - parameters['v3']
                                       ) / (parameters['v4'] - parameters['v3'])

            # q_set selection from choices based on conditions
            parameters['qbyqmax'] = np.select(
                [under_voltage, voltage_is_low, noramal_range, voltage_is_high,
                 over_voltage], [for_under_voltage, for_low_voltage,
                                 for_noramal_range, for_high_voltage,
                                 for_over_voltage])
            # determining inverter max reactive power capacity
            parameters['Q_max'] = np.sqrt(parameters['sn']**2 - (
                p_input.loc[now, parameters.index])**2)
            q_out = ((parameters['qbyqmax']*parameters['Q_max']
                      )/100)*parameters['damper']
            q_out = np.where(component_type == 'loads', -q_out, q_out)

            return q_out

    # parameters of indexes with varying p_set at each snapshot
    parameters_var_p_set = component_df_p_set_var.loc[(
        component_df_p_set_var['type_of_control_strategy'] == 'q_v'), (
            'sn', 'v1', 'v2', 'v3', 'v4', 'damper', 'bus')]

    # parameters of indexes with fixed p_set at each snapshot
    parameters_fixed_p_set = component_df.loc[(~(component_df.index).isin(
        component_df_t.p_set.columns)) & (
            component_df.type_of_control_strategy == 'q_v'), (
                'sn', 'v1', 'v2', 'v3', 'v4', 'damper', 'bus')]

    # setting the new q_sets for fixed p_sets indexes
    component_df_t.q_set.loc[now, parameters_var_p_set.index] = calculation(
        parameters_var_p_set)

    # setting the new q_sets for varying p_sets indexes
    component_df.loc[parameters_fixed_p_set.index, 'q_set'] = calculation(
        parameters_fixed_p_set)


def apply_controller_to_components_df(component_df, component_df_t, v_pu_buses,
                                      now, n_trials, component_type):
    '''
    The function interates in components data frames and apply the right
    controller types to the controlled component.

    Parameters
    ----------
    component_df : pandas data frame
        Can be n.loads/storage_units/generators.
    component_df_t : pandas data frame
        can be n.loads_t/storage_units_t/generators_t
    v_pu_buses : pandas data frame
        v_mag_pu of all buses.
    now : single snapshot
        An current (happening) element of n.snapshots on which the power
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
    p_input = component_df_t.p

    # complete df of controlled components, with varying p_set
    component_df_p_set_var = component_df.loc[(component_df.index).isin(
      component_df_t.p_set.columns) & (component_df['type_of_control_strategy']
                                       != ''), :]
    # adding column labels to q_set data frame
    component_df_t.q_set = component_df_t.q_set.reindex((
        component_df_p_set_var.loc[(component_df_p_set_var[
            'type_of_control_strategy'].isin(['q_v', 'fixed_cosphi', 'cosphi_p'
                                              ]))].index), axis=1, fill_value=0)
    # applying control strategies
    fixed_cosphi(p_input, now, component_df_p_set_var, component_df_t,
                 component_df)

    cosphi_p(p_input, now, component_df_p_set_var, component_df_t, component_df)

    q_v(p_input, now, v_pu_buses, component_type, n_trials,
        component_df_p_set_var, component_df_t, component_df)

    return bus_names


def apply_controller(n, now, n_trials):
    '''
    This function iterates to storage_units, loads and generators to check
    if any controller is chosen to apply it to the component.

    Parameters
    ----------
    n : pypsa.components.Network
        Network
    now : single snaphot
        An current (happening) element of n.snapshots on which the power
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
    v_buses = n.buses_t.v_mag_pu
    bus_name_l = bus_name_g = bus_name_s = np.array([])

    if n.loads.loc[n.loads.type_of_control_strategy != '',
                   'type_of_control_strategy'].any():
        bus_name_l = apply_controller_to_components_df(
            n.loads, n.loads_t, v_buses, now, n_trials, 'Load')

    if n.generators.loc[n.generators.type_of_control_strategy != '',
                        'type_of_control_strategy'].any():
        bus_name_g = apply_controller_to_components_df(
            n.generators, n.generators_t, v_buses, now, n_trials, 'generators')

    if n.storage_units.loc[n.storage_units.type_of_control_strategy != '',
                           'type_of_control_strategy'].any():
        bus_name_s = apply_controller_to_components_df(
            n.storage_units, n.storage_units_t, v_buses, now, n_trials,
            'storage_units')

    # combining all bus names from loads, generators and storage_units
    voltage_dependent_controller_bus_names = np.unique(np.concatenate((
        bus_name_l, bus_name_g, bus_name_s), axis=0))
    # finding the bus voltages of buses controlled by voltage dep. controllers
    v_mag_pu_voltage_dependent_controller = v_buses.loc[
        now, voltage_dependent_controller_bus_names]

    return v_mag_pu_voltage_dependent_controller


def iterate_over_control_strategies(n):
    '''
    This method iterates in componets and checks if any voltage dependent
    controller is chosen or not. If any is chosen it will give an output which
    enables the power flow to repeat until the condition of while loop in pf.py
    is met.

    Parameters
    ----------
    n : pypsa.components.Network
        Network
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
        n.loads.type_of_control_strategy.isin(['q_v']).any(
        ) or n.generators.type_of_control_strategy.isin(['q_v']).any(
            ) or n.storage_units.type_of_control_strategy.isin(
                ['q_v']).any(), True, False)

    n_trials_max = np.where(voltage_dependent_controller_present,
                            n_trials_max, 1)
    return voltage_dependent_controller_present, n_trials_max
