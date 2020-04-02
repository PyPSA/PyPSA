import numpy as np
import logging
logger = logging.getLogger(__name__)


def fixed_cosphi(pf, p_set_input, now):
    '''
    Fixed Power factor (cosphi) control strategy

    parameteres
    -----------
    pf:             Power factor dataframe with component indexes
    p_set_input:    Real power input to the controller
    now:            The current snapshot (happening)
    '''
    q_out = -p_set_input.loc[now, pf.index].mul(np.tan((np.arccos(pf))))
    return q_out


def cosphi_p(parameters, p_set_input, now):
    '''
    Power factor as a fucntion of real power (cosphi_p) control strategy

    parameteres
    -----------
    parameters: Data frame of containing the parameters (sn, set_p1, set_p2, pf_min)
    sn:         Inverter complex power capacity
    set_p1:     Set point until wich controller takes no action and
                power factor is 1 up this setpoint.

    set_p2:     Set point where controller compensates with
                power factor equals pf_min. set_p1 and set_p2 are defined
                in terms of %. for example if set_p1 = 50%, this means that
                if p_set/sn*100 is less than set_p1 this strategy assumes no
                reactive power compensation and power factor is 1. The opposite
                case is true for set_p2 and inverter will compensate with pf_min.
    pf_min      Minimum allower power factor
    p_set_input:Real power generation / injection / consumption
                in case of storage_units.
    now         The current snapshot (happening)
    dataframe:  The complete dataframe of (load, storage or generators)
    reference:  https://ieeexplore.ieee.org/document/6096349

    '''
    if not parameters.empty:
        parameters['p_per_p_max'] = (abs(p_set_input.loc[now, parameters.index]) / abs(parameters['sn']))*100
        parameters.loc[parameters['p_per_p_max'] < parameters['set_p1'], 'pf'] = 1

        parameters.loc[(parameters['p_per_p_max'] >= parameters['set_p1']) & (parameters['p_per_p_max'] <=
                       parameters['set_p2']), 'pf'] = 1 - ((1-parameters['pf_min']) / (
                        parameters['set_p2'] - parameters['set_p1'])*(parameters['p_per_p_max'] - parameters['set_p1']))

        parameters.loc[parameters['p_per_p_max'] > parameters['set_p2'], 'pf'] = parameters['pf_min']

        q_out = -p_set_input.loc[now, parameters.index].mul(np.tan((np.arccos(parameters['pf']))))

        return q_out, parameters['pf']
    else:
        pass


def q_v(parameters, p_set_input, now, v_pu_buses, component_type, n_trials):
    '''
    Reactive power as a function of voltage Q(U)

    Parameters
    ----------
    Parameters:Data frame of containing the parameters ('sn', 'v1','v2','v3','v4', 'damper','bus')

    sn:        Inverter complex power capacity
    curves     v1 < v2 < v3 < v4 are the voltage per unit set points where
               different curve limits are defined, starting from v1 as minimum
               allowed power factor and ending with v4 as maximum allowed power
               factor in the grid. Based on these set points and depending on
               where the bus voltage will be located controller compensates
               inductive or capacitive reactive power to regulate the voltage
               close to the desired value as close as possible.

    damper:    Is used for adding damping characteristic to the controller output
               damper parameter is used when the controller is not meeting the
               while loop condition (while voltage_difference > 0.001 in pf.py
               file) and it enters into infinite loop. Although the infinite
               loop is limited to 20 iterations but when it does not meet the
               condition up to 20 iterations the controller q_out might take the
               value from previous iterations which is not consistent with the
               current iteration voltage per unit. This problem can be solved
               using  damper (0 - 1) parameter.
   bus         The bus to which inverter is connected

   p_set_input:Real power generation / injection / consumption
               in case of storage_units.
   now         The current snapshot (happening)
   v_pu_buses  Voltage per unit of all buses controlled by this control method
    type_component: Can be loads, generators, or storage units
    qbyqmax:   Reactive power compensation in percentage
               for each curve limit zone.
    q_out:     Controller reactive power compensation output.
    Q_max:     Inveter maximum reactive power capacity at each itereation, which
               can be different at each snapshot if p_set is changing.
    reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6096349

    '''
    if n_trials == 20:
        logger.warning("The voltage difference at snapshot ' %s' , in components '%s', "
                       "with 'q_v' control exceeds x_tol_outer limit, please apply (damper < 1) or "
                       "expand controller parameters range between v1 & v2 and v3 & v4  to avoid the problem."
                       % (now, parameters.index.values))

    if not parameters.empty:
        parameters['p_set'] = p_set_input.loc[now, parameters.index]
        parameters['v_pu_bus'] = v_pu_buses.loc[now, parameters.loc[parameters.index, 'bus']].values

        # defining curve limits
        parameters.loc[parameters['v_pu_bus'] < parameters['v1'], 'qbyqmax'] = 100

        parameters.loc[(parameters['v_pu_bus'] >= parameters['v1']) & (parameters['v_pu_bus'] <=
                       parameters['v2']), 'qbyqmax'] = (100 - (100-0) / (parameters['v2'] - parameters['v1'])*(
                           parameters['v_pu_bus'] - parameters['v1']))

        parameters.loc[(parameters['v_pu_bus'] > parameters['v2']) & (parameters['v_pu_bus'] <=
                       parameters['v3']), 'qbyqmax'] = 0

        parameters.loc[(parameters['v_pu_bus'] > parameters['v3']) & (parameters['v_pu_bus'] <=
                       parameters['v4']), 'qbyqmax'] = -(100)*(parameters['v_pu_bus'] - parameters['v3']) / (
                           parameters['v4'] - parameters['v3'])

        parameters.loc[(parameters['v_pu_bus'] > parameters['v4']), 'qbyqmax'] = -100

        parameters['Q_max'] = np.sqrt(parameters['sn']**2 - (p_set_input.loc[now, parameters.index])**2)
        q_out = ((parameters['qbyqmax']*parameters['Q_max'])/100)*parameters['damper']

        if component_type == 'loads':
            return -q_out
        else:
            return q_out
    else:
        pass


def apply_controller_to_components_df(component_df, component_df_t, v_pu_buses, now, n_trials, component_type):
    repeat_loop = 0
    component_buses_controlled_by_q_v = np.unique(component_df.loc[(
        component_df['type_of_control_strategy'].isin(['q_v'])), 'bus'].values)
    ctrl_list = ['', 'q_v', 'cosphi_p', 'fixed_cosphi']

    if component_buses_controlled_by_q_v.size != 0:
        repeat_loop = 1
    p_set_input = component_df_t.p
    assert (component_df.type_of_control_strategy.any() in ctrl_list), (
        "The type of controllers %s you have typed in components %s, is not supported.\n Supported controllers are %s."
        % (component_df.loc[(~ component_df['type_of_control_strategy'].isin(
            ctrl_list)), 'type_of_control_strategy'].values,
            component_df.loc[(~component_df['type_of_control_strategy'].isin
                                (ctrl_list)), 'type_of_control_strategy'].index.values, ctrl_list))

    # seperation of varying p_set complete data frame: it will contain data of all the indexes which are having p_set_t
    component_t_df = component_df.loc[(component_df.index).isin(component_df_t.p_set.columns) & (
                         component_df['type_of_control_strategy'] != ''), :]

    # adding column labels to q_set and power factor (pf) dataframes, which are empty initially
    component_df_t.q_set = component_df_t.q_set.reindex((component_t_df.loc[(component_t_df[
        'type_of_control_strategy'].isin(['q_v', 'fixed_cosphi', 'cosphi_p']))].index), axis=1, fill_value=0)
    component_df_t.pf = component_df_t.pf.reindex((component_t_df.loc[(component_t_df[
        'type_of_control_strategy'].isin(['cosphi_p']))].index), axis=1, fill_value=0)

    # fixed_cosphi method
    component_df_t.q_set.loc[now, component_t_df.type_of_control_strategy == 'fixed_cosphi'] = fixed_cosphi(
        component_t_df.loc[(component_t_df['type_of_control_strategy'] == 'fixed_cosphi'), 'pf'], p_set_input, now)
    components_with_fixed_p_set = component_df[(~(component_df.index).isin(component_df_t.p_set.columns)) & (
        component_df.type_of_control_strategy == 'fixed_cosphi')].index
    component_df.loc[components_with_fixed_p_set, 'q_set'] = fixed_cosphi(
        component_df.loc[components_with_fixed_p_set, 'pf'], p_set_input, now)

    # # cosphi_p method
    if not component_df_t.pf.empty:
        component_df_t.q_set.loc[now, component_t_df.type_of_control_strategy == 'cosphi_p'], component_df_t.pf.loc[
          now, component_t_df.type_of_control_strategy == 'cosphi_p'] = cosphi_p(component_t_df.loc[(component_t_df[
          'type_of_control_strategy'] == 'cosphi_p'), ('pf_min', 'sn', 'set_p1', 'set_p2')], p_set_input, now)
    components_with_fixed_p_set = component_df[(~(component_df.index).isin(component_df_t.p_set.columns)) & (
        component_df.type_of_control_strategy == 'cosphi_p')].index
    if not components_with_fixed_p_set.empty:
        component_df.loc[components_with_fixed_p_set, 'q_set'], component_df.loc[
            components_with_fixed_p_set, 'pf'] = cosphi_p(component_df.loc[
                components_with_fixed_p_set, ('pf_min', 'sn', 'set_p1', 'set_p2')], p_set_input, now)

    # Q(U) method
    component_df_t.q_set.loc[now, component_t_df.type_of_control_strategy == 'q_v'] = q_v(component_t_df.loc[(
        component_t_df['type_of_control_strategy'] == 'q_v'), ('sn', 'v1', 'v2', 'v3', 'v4', 'damper', 'bus')],
            p_set_input, now, v_pu_buses, component_type, n_trials)
    components_with_fixed_p_set = component_df[(~(component_df.index).isin(component_df_t.p_set.columns)) & (
        component_df.type_of_control_strategy == 'q_v')].index
    component_df.loc[components_with_fixed_p_set, 'q_set'] = q_v(component_df.loc[components_with_fixed_p_set, (
        'sn', 'v1', 'v2', 'v3', 'v4', 'damper', 'bus')], p_set_input, now, v_pu_buses, component_type, n_trials)

    return repeat_loop, component_buses_controlled_by_q_v


def apply_controller(network, now, n_trials):
    v_buses = network.buses_t.v_mag_pu
    voltage_difference = 0
    load_buses_controlled_by_q_v = gen_buses_controlled_by_q_v = storage_buses_controlled_by_q_v = np.array([])
    repeat_loop_l, repeat_loop_g, repeat_loop_s = 0, 0, 0

    if network.loads.loc[network.loads.type_of_control_strategy != '', 'type_of_control_strategy'].any():
        repeat_loop_l, load_buses_controlled_by_q_v = apply_controller_to_components_df(
            network.loads, network.loads_t, v_buses,
            now, n_trials, 'loads')

    if network.generators.loc[network.generators.type_of_control_strategy != '', 'type_of_control_strategy'].any():
        repeat_loop_g, gen_buses_controlled_by_q_v = apply_controller_to_components_df(
            network.generators, network.generators_t, v_buses,
            now, n_trials, 'generators')

    if network.storage_units.loc[network.storage_units.type_of_control_strategy != '', 'type_of_control_strategy'].any():
        repeat_loop_s, storage_buses_controlled_by_q_v = apply_controller_to_components_df(
            network.storage_units, network.storage_units_t, v_buses,
            now, n_trials, 'storage_units')

    repeat_loop = repeat_loop_l + repeat_loop_g + repeat_loop_s
    bus_names_controlled_by_q_v = np.unique(np.concatenate((
        load_buses_controlled_by_q_v, gen_buses_controlled_by_q_v, storage_buses_controlled_by_q_v), axis=0))
    v_pu_buses_controlled_by_q_v = v_buses.loc[now, bus_names_controlled_by_q_v]

    return v_pu_buses_controlled_by_q_v, repeat_loop, voltage_difference
