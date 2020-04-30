import numpy as np
import logging
logger = logging.getLogger(__name__)


def fixed_cosphi(p_input, now, parameters):
    '''
    Fixed Power factor (pf) (fixed_cosphi): Sets the new q_set for controlled
    components according to the real power input and power factor given.
    reference : https://www.academia.edu/24772355/
    '''
    pf = parameters['pf']
    q_set_out = -p_input.loc[now, pf.index].mul(np.tan(np.arccos(pf)))
    return q_set_out


def cosphi_p(p_input, now, parameters):
    '''
    Power factor as a function of real power (cosphi_p): Sets new values to
    power factor (pf) and q_sets based on the amount of power injection.
    Parameters:
    ----------
    parameters : pandas data frame
        It contains the following parameters(sn, set_p1, set_p2, pf_min).
    sn : pandas data frame
        Inverter complex power capacity.
    set_p1 : pandas data frame
        The set point set_p1 or (p_set/sn)*100 in %, where p_set is the
        amount of injection, its shows inverter injection share out of its
        full capacity (sn). controller takes no action up to set_p1 point.
    set_p2 : pandas data frame
        The set point set_p2 has same definition as set_p1 but set_p2 is
        normally the point where we are expecting the controller to operate
        with full capacity - compensate with minimum allowed power factor.
    pf_min : pandas data frame
        Minimum allowed power factor.
    p_set_per_sn : pandas data frame
        Inverter real power injection percentage.
    Returns
    -------
    q_set : pandas data frame
        Is the new reactive power that will be set as new q_set in each
        controlled component as the result of applying cosphi_p controller.
    pf : pandas data frame
    power factor (pf) that will be set as new pf value in each controlled
        component as the result of applying cosphi_p controller.
    ref : https://ieeexplore.ieee.org/document/6096349
    '''
    # controller parameters
    sn = parameters['sn']
    set_p1 = parameters['set_p1']
    set_p2 = parameters['set_p2']
    pf_min = parameters['pf_min']
    p_set_per_sn = (abs(p_input.loc[now, parameters.index])/abs(sn))*100

    # pf allocation using np.select([condtions...], [choices...]) function.
    pf = np.select([(p_set_per_sn < set_p1), (p_set_per_sn >= set_p1) & (
              p_set_per_sn <= set_p2), (p_set_per_sn > set_p2)], [1, (1-(
               (1-pf_min)/(set_p2-set_p1)*(p_set_per_sn-set_p1))), pf_min])

    q_set = -p_input.loc[now, parameters.index].mul(np.tan((np.arccos(pf))))

    return q_set, pf


def q_v(p_input, now, v_pu_buses, component_type, n_trials, n_trials_max, parameters):
    '''
    Reactive power as a function of voltage Q(U): In this strategy controller
    finds the amount of inverter reactive power capability according to the
    amount of power injection and then compensates reactive power based on
    v_mag_pu of the bus where inverter is connected.
    Parameters:
    ----------
    parameters : pandas data frame
        It contains the following parameters ('sn', 'v1', 'v2', 'v3', 'v4',
                                                           'damper','bus').
    sn : pandas data frame
        Inverter complex power capacity.
    v1, v2 : pandas data frame
        v_mag_pu set points for the lower portion of the q_v curve, wherev1
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
        power capability of inverter (q_max = np.sqrt(sn**2 - (p_set)**2).
    ref : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6096349
    Returns
    -------
    q_set_out : pandas data frame
    Is the new reactive power that will be set as new q_set in each
        controlled component as the result of applying q_v controller.
    '''
    if n_trials == n_trials_max:
        logger.warning("The voltage difference at snapshot ' %s' , in "
                       "components '%s', with 'q_v' controller exceeds"
                       "x_tol_outer limit, please apply (damper < 1) or"
                       "expand controller parameters range between v1 &"
                       " v2 and or v3 & v4  to avoid the problem." % (
                           now, parameters.index.values))

    p_set = p_input.loc[now, parameters.index]
    v_pu_bus = v_pu_buses.loc[now, parameters.loc[
                                   parameters.index, 'bus']].values
    v1 = parameters['v1']
    v2 = parameters['v2']
    v3 = parameters['v3']
    v4 = parameters['v4']
    sn = parameters['sn']
    damper = parameters['damper']

    # q_set selection from choices based on conditions in percentage %
    q_set_per_qmax = np.select([(v_pu_bus < v1), (v_pu_bus >= v1) & (
               v_pu_bus <= v2), (v_pu_bus > v2) & (v_pu_bus <= v3), (
                   v_pu_bus > v3) & (v_pu_bus <= v4), (v_pu_bus > v4)
                   ], [100, (100-(100-0) / (v2-v1)*(v_pu_bus-v1)), 0,
                       -(100)*(v_pu_bus-v3) / (v4-v3), -100])

    q_out = ((q_set_per_qmax*(np.sqrt(sn**2-(p_set)**2))) / 100)*damper
    q_set_out = np.where(component_type == 'loads', -q_out, q_out)
    """
    # TODO: jankaeh: this is the only time component_type is used?
    gets passed along from apply_controller():
    apply_controller_to_df() -> prepare_df_and_call_controllers() -> q_v()
    documentation says: can be 'loads', 'storage_units' or 'generators'.
    Only if it is called with "loads" it has an effect as per line 130. Instead
    it is called with "Load" in apply_controller() in line 267
    """

    return q_set_out


def prepare_df_and_call_controllers(p_input, now, df_t, df, controller,
                                    v_pu_buses, component_type, n_trials,
                                    n_trials_max):
    '''
    This method filters index of components based on type of control and then
    call each controller based on filtered dataframe. df_of_p_set_t and
    df_of_p_set methods filter df for indexes having variable p_set and fixed
    p_set in each snapshots respectively.
    '''
    def df_of_p_set_t(control_type): return df.loc[
                    (df.index.isin(df_t.p_set) & (controller == control_type))]

    def df_of_p_set(control_type): return df.loc[
<<<<<<< HEAD
                    (~df.index.isin(df_t.p_set) & (controller == control_type))]
=======
                           (~df.index.isin(df_t.p_set) & (controller == control_type))]
>>>>>>> 610a41241acb85a797cb3eda7d7c3c77a96dad38
    # cosphi_p controller
    df_t.pf = df_t.pf.reindex(df_of_p_set_t('cosphi_p').index, axis=1)

    df_t.q_set.loc[now, df_of_p_set_t('cosphi_p').index], df_t.pf.loc[
                              now, df_of_p_set_t('cosphi_p').index] = cosphi_p(
                                       p_input, now, df_of_p_set_t('cosphi_p'))

    df.loc[df_of_p_set('cosphi_p').index, 'q_set'], df.loc[df_of_p_set(
     'cosphi_p').index, 'pf'] = cosphi_p(p_input, now, df_of_p_set('cosphi_p'))

    # q_v controller
    df_t.q_set.loc[now, df_of_p_set_t('q_v').index] = q_v(
      p_input, now, v_pu_buses, component_type, n_trials, n_trials_max, df_of_p_set_t('q_v'))

    df.loc[df_of_p_set('q_v').index, 'q_set'] = q_v(
        p_input, now, v_pu_buses, component_type, n_trials, n_trials_max, df_of_p_set('q_v'))

    # fixed_cosphi
    df_t.q_set.loc[now, df_of_p_set_t('fixed_cosphi').index] = fixed_cosphi(
                                   p_input, now, df_of_p_set_t('fixed_cosphi'))

    df.loc[df_of_p_set('fixed_cosphi').index, 'q_set'] = fixed_cosphi(
                                     p_input, now, df_of_p_set('fixed_cosphi'))


def apply_controller_to_df(df, df_t, v_pu_buses, now, n_trials,
                           n_trials_max, component_type):
    '''
    The function interates in components data frames and apply the right
    controller types to the controlled component.
    Parameters:
    ----------
    df : pandas data frame
        component data frame. eg: n.loads/storage_units/generators.
    df_t : pandas data frame
        Variable component data frame eg: n.loads_t/storage_units_t/generators_t
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
    p_input = df_t.p
    controller = df['type_of_control_strategy']
    ctrl_list = ['', 'q_v', 'cosphi_p', 'fixed_cosphi']

    assert (controller.isin(ctrl_list)).all(), (
        "Not all given types of controllers are supported. "
        "Elements with unknown controllers are:\n%s\nSupported controllers are"
        ": %s." % (df.loc[(~ df['type_of_control_strategy'].isin(ctrl_list)),
        'type_of_control_strategy'], ctrl_list))

    # names of buses controlled by voltage dependent controllers
    bus_names = np.unique(df.loc[(df['type_of_control_strategy'
                                     ].isin(['q_v'])), 'bus'].values)

    # adding lables n.df_t.q_set data frame
    df_t.q_set = df_t.q_set.reindex((df.loc[(df.index.isin(df_t.p_set) & (
                                  controller.isin(ctrl_list)))]).index, axis=1)

    prepare_df_and_call_controllers(p_input, now, df_t, df, controller,
                                    v_pu_buses, component_type, n_trials,
                                    n_trials_max)

    return bus_names


def apply_controller(n, now, n_trials, n_trials_max):
    '''
    This function iterates to storage_units, loads and generators to check
    if any controller is chosen to apply it to the component.
    Parameters:
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
        v_mag_pu of those buses which they have voltage dependent controller
        attached to one of the inverters connected to them. This is important
        in or to be able to compare these voltages with the voltage from the
        next n_trial of the power flow and take their difference as a condition
        repeat or not for the next iteration of the while loop in pf.py file.
    '''
    v_buses = n.buses_t.v_mag_pu
    bus_name_l = bus_name_g = bus_name_s = np.array([])

    if n.loads.loc[n.loads.type_of_control_strategy != '',
                   'type_of_control_strategy'].any():
        bus_name_l = apply_controller_to_df(
<<<<<<< HEAD
                            n.loads, n.loads_t, v_buses, now, n_trials, 'loads')
=======
                n.loads, n.loads_t, v_buses, now, n_trials,
                n_trials_max, 'Load')
>>>>>>> 610a41241acb85a797cb3eda7d7c3c77a96dad38

    if n.generators.loc[n.generators.type_of_control_strategy != '',
                        'type_of_control_strategy'].any():
        bus_name_g = apply_controller_to_df(
                n.generators, n.generators_t, v_buses, now, n_trials,
                n_trials_max, 'generators')

    if n.storage_units.loc[n.storage_units.type_of_control_strategy != '',
                           'type_of_control_strategy'].any():
        bus_name_s = apply_controller_to_df(
                n.storage_units, n.storage_units_t, v_buses, now, n_trials,
                n_trials_max, 'storage_units')

    # combining all bus names from loads, generators and storage_units
    voltage_dependent_controller_bus_names = np.unique(np.concatenate((
        bus_name_l, bus_name_g, bus_name_s), axis=0))

    # finding v_mag_pu of buses controlled by voltage dependent controllers
    v_mag_pu_voltage_dependent_controller = v_buses.loc[
        now, voltage_dependent_controller_bus_names]

    return v_mag_pu_voltage_dependent_controller


def iterate_over_control_strategies(n):
    '''
    This method checks if any voltage dependent controller is chosen or not.
    If any is chosen it will return the output such that it enables the power
    flow to repeat until the condition of while loop in pf.py is met.
    Parameters:
    ----------
    n : pypsa.components.Network-
        Network containing all components and elements of the power flow
    Returns
    -------
    voltage_dependent_controller_present : bool, defaut False
        It will give a bool value (True/False) whether voltage dependent
        controller is present in the grid or not, if present True if not False.
    n_trials_max : integer
        This enables the power flow to repeat power flow in while loop inside
        pf.py file until either the while loop condition is met or n_trial_max
        is reached.
    '''
    n_trials_max = 20  # can be also an attribute of network
    voltage_dependent_controller_present = np.where(
        n.loads.type_of_control_strategy.isin(['q_v']).any(
        ) or n.generators.type_of_control_strategy.isin(['q_v']).any(
            ) or n.storage_units.type_of_control_strategy.isin(['q_v']).any(
                ), True, False)

    n_trials_max = np.where(
                         voltage_dependent_controller_present, n_trials_max, 1)
    return voltage_dependent_controller_present, n_trials_max
