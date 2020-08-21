"""importing important libraries."""
from .descriptors import get_switchable_as_dense
import logging
import numpy as np
logger = logging.getLogger(__name__)


def find_allowable_q(p, power_factor, s_nom):
    """
    Some times the reactive power that controller want to compensate using
    (p*tan(arccos(power_factor))) can go higher than what inverter can provide
    based on inverter "s_nom" and the provided "power_factor", in this case:
        - calculate reactive power that the formula gives "q"
        - calcualte reactive power max available capacity that inverter can
          provide based on the power factor given "q_inv_cap".
        - check q if it is less than "q_inv_cap" ok, if not take the value from
          "q_allowable" instead.
        - Return all (q, q_inv_cap, q_allowable) to the controller for further
          calculations and considerations.

    This values are returned to controller in order to check and make sure that
    the inverter equation s_nom = np.sqrt((p**2 + q**2) is not violated.
    """
    # Calculate reactive power that controller want ot compensate initially
    q = p.mul(np.tan((np.arccos(power_factor, dtype=np.float64)),
                     dtype=np.float64))
    # find inverter q capacity according to power factor provided
    q_inv_cap = s_nom*np.sin(np.arccos(power_factor, dtype=np.float64),
                             dtype=np.float64)
    # find max allowable q that is possible for controller to give as output
    q_allowable = np.where(q <= q_inv_cap, q, q_inv_cap)

    return q_inv_cap, q_allowable, q


def adjust_p_set(s_nom, q, p, c, control_strategy):
    """
    when the initial reactive power "q" calculated by controller together with
    the active power "p" violates inverter equation s_nom = np.sqrt((p**2 + q**2),
    in this case controller needs to reduce p in order to fulfil reactive power
    need. In this case p is reduced and calculated here "new_p_set" and return
    it to the controller to consider this as p_out and set it to the network.
    """
    adjusted_p_set = np.sqrt((s_nom**2 - q**2),  dtype=np.float64)
    new_p_set = np.where(abs(p) <= abs(adjusted_p_set), p, adjusted_p_set)
    # info for user that p_set has been changed
    log_info = np.where(
            control_strategy == "fixed_cosphi', '"fixed_cosphi" control is adjusted',
            ' "q_v" control might be adjusted, if needed')

    logger.info(" Some p_set in '%s' component with %s due to reactive power "
                "compensation priority. ", c, log_info)

    return new_p_set


def apply_fixed_cosphi(n, snapshot, c, index):
    """
    fix power factor inverter controller.
    This controller provides a fixed amount of reactive power compensation to the
    grid as a function of the amount of injected power (p_set) and the chosen
    power factor value. 
    Controller will take care of inverter capacity and controlls that the sum
    of active and reactive power does not increase than the inverter capacity.
    When reactive power need is more than what controller calculate based on
    the provided power factor, controller decreases a portion of active power
    to meet reactive power need, in this case controller will have two outputs
    p_out and q_out where q_out is the reactive power output and p_out is the
    reduced p_set and will be updated in buses_t.p and components_t.p.
    
    Finally the controller outpus are passed to "_set_controller_outputs_to_n"
    to update the network.

    reference : https://ieeexplore.ieee.org/document/6096349
    DOI link  : 10.1109/JPHOTOV.2011.2174821

    Parameters
    ----------
    n : pypsa.components.Network
        Network
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    index : index of controlled elements
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...

    Returns
    -------
    None
    """
    # needed parameters
    p_input = n.pnl(c).p.loc[snapshot, index]
    params = n.df(c).loc[index]
    power_factor = params['power_factor']
    s_nom = params['s_nom']

    p_out=None
    ctrl_p_out = False
    q_inv_cap, q_allowable, q = find_allowable_q(p_input, power_factor, s_nom)
    q_out = -q_allowable

    # check if the calculated q is not exceeding the inverter capacity if yes then
    # decrease p_input in order not to violate s_nom = np.sqrt((p**2 + q**2) .
    if (abs(q) > q_inv_cap).any().any():
        ctrl_p_out = True
        p_out = adjust_p_set(s_nom, q_out, p_input, c, 'fixed_cosphi')

    _set_controller_outputs_to_n(n, c, index, snapshot, ctrl_p_out=ctrl_p_out,
                                 ctrl_q_out=True, p_out=p_out, q_out=q_out)


def apply_cosphi_p(n, snapshot, c, index):
    """
    Power factor as a function of active power (cosphi_p) controller.
    This controller provides a variable power factor value based on the chosen
    parameters and the droop curve defined below. And then using the calculated
    power factor an amount of reactive power is calculated for reactive power
    compensation, controller works as follow:
        - calculate: p_set_per_p_ref = (p_set / p_ref)*100, where p_ref is a
          setpoint in MW.
        - Then controller compares "p_set_per_p_ref" with the "set_p1" and
          "set_p2" set points where set_p1 and set_p2 are percentage values.
        - Controller decides the power factor based on the defined droop below
          (power_factor = ...). i.e. if p_set_per_p_ref < set_p1 then power
          factor is 1, since p_set_per_p_ref < set_p1 shows low generation and
          controller think there might not be any need for reactive power
          with this amount of generation, thus power_factor=1 which means q = 0.
          For the other conditions power factor is calculated respectively.
    Finally the controller outpus are passed to "_set_controller_outputs_to_n"
    to update the network.

    reference : https://ieeexplore.ieee.org/document/6096349.
    DOI link  : 10.1109/JPHOTOV.2011.2174821

    Parameters
    ----------
    n : pypsa.components.Network
        Network
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    index : index of controlled elements
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...

    Returns
    -------
    None
    """
    # parameters needed
    params = n.df(c).loc[index]
    p_input = n.pnl(c).p.loc[snapshot, index]

    p_set_per_p_ref = (abs(p_input) / params['p_ref'])*100

    # choice of power_factor according to controller inputs and its droop curve
    power_factor = np.select([(p_set_per_p_ref < params['set_p1']),
         (p_set_per_p_ref >= params['set_p1']) & (p_set_per_p_ref <= params[
            'set_p2']), (p_set_per_p_ref > params['set_p2'])],
        [1, (1 - ((1 - params['power_factor_min']) / (params['set_p2'] - params[
                    'set_p1']) * (p_set_per_p_ref - params['set_p1']))),
            params['power_factor_min']])

    # find q_set and avoid -0 apperance as the output when power_factor = 1
    q_out = np.where(power_factor == 1, 0, -p_input.mul(np.tan((np.arccos(
                          power_factor, dtype=np.float64)), dtype=np.float64)))

    S = np.sqrt((p_input)**2 + q_out**2)
    assert ((S < params['s_nom']).any().any()), (
        "The resulting reactive power (q)  while using 'cosphi'_p control  "
        "with the chosen attr 'power_factor_min' in '%s' component results a  "
        "complex power (S = sqrt(p**2 + q**2))) which is greater than 's_nom') "
        "of the inverter, please choose the right power_factor_min value" % (c))

    _set_controller_outputs_to_n(
        n, c, index, snapshot, ctrl_q_out=True, q_out=q_out)


def apply_q_v(n, snapshot, c, index, n_trials_max, n_trials):
    """
    Reactive power as a function of voltage Q(V).
    This contrller controller provide reactive power compensation based on the
    voltage information of the bus where inverter is connected, for this purpose
    the droop for reactive power calculation is divided in to 5 different reactive
    power calculation zones. Where v1, v2, v3, v4 attrs form the droop and the
    reactive power is calculated based on which zone the bus v_mag_pu is landing.
        - controller finds the zone where bus v_mag_pu lands on
        - Based on the zone and the droop provided it calcualtes "curve_q_set_in_percentage"
        - Using "curve_q_set_in_percentage" it calcualtes reactive power q_out.
    Controller will take care of inverter capacity and controlls that the sum
    of active and reactive power does not increase than the inverter capacity.
    When reactive power need is more than what controller calculate based on
    the provided power factor, controller decreases a portion of active power
    to meet reactive power need, in this case controller will have two outputs
    p_out and q_out.
    Finally the controller outpus are passed to "_set_controller_outputs_to_n"
    to update the network.

    reference : https://ieeexplore.ieee.org/document/6096349
    DOI link  : 10.1109/JPHOTOV.2011.2174821 

    Parameters:
    ----------
    n : pypsa.components.Network
        Network
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    index : index of controlled elements
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...
    n_trials_max : integer
        It is the max number of outer loop (while loop in pf.py) trials until
        the controller converges.
    n_trials : integer
        It is the outer loop (while loop in pf.py) number of trials until
        the controller converges.

    Returns
    -------
    None
    """
    if n_trials == n_trials_max:
        logger.warning("The voltage difference at snapshot ' %s', in components"
                       " '%s', with 'q_v' controller exceeds x_tol_outer limit,"
                       " please apply (damper < 1) or expand controller"
                       " parameters range between v1 & v2 and or v3 & v4 to"
                       " avoid the problem." % (snapshot, index))
    #  curve parameters
    v_pu_bus = n.buses_t.v_mag_pu.loc[snapshot, n.df(c).loc[index, 'bus']].values
    params = n.df(c).loc[index]
    print()
    p_input = n.pnl(c).p.loc[snapshot, index]
    p_out = None
    ctrl_p_out = False
    q_inv_cap, q_allowable, q = find_allowable_q(p_input, params['power_factor'], params['s_nom'])

    # calculation of maximum q compensation in % based on bus v_pu_bus inside
    # np.select([conditions], [choices]) function.
    curve_q_set_in_percentage = np.select([
        (v_pu_bus < params['v1']),(v_pu_bus >= params['v1']) & (
            v_pu_bus <= params['v2']), (v_pu_bus > params['v2']) & (
                v_pu_bus <= params['v3']), (v_pu_bus > params['v3'])
        & (v_pu_bus <= params['v4']), (v_pu_bus > params['v4'])], [
            100, 100 - 100 / (params['v2'] - params['v1']) * (
                v_pu_bus - params['v1']), 0, -100 * (
                v_pu_bus - params['v3']) / (params['v4'] - params['v3']), -100])

    # calculation of q based on the "curve_q_set_in_percentage" output
    q_out = (((curve_q_set_in_percentage * q_allowable) / 100) * params[
            'damper'] * params['sign'])

    # check if there is need to reduce p_set due to q need
    if (q > q_inv_cap).any().any():
        ctrl_p_out = True
        p_out = adjust_p_set(params['s_nom'], q, p_input, c, 'q_v')

    _set_controller_outputs_to_n(n, c, index, snapshot, ctrl_p_out=ctrl_p_out,
                                 ctrl_q_out=True, p_out=p_out, q_out=q_out)


def apply_controller(n, now, n_trials, n_trials_max, dict_controlled_index):
    """
    Iterate over chosen control strategies which exist as keys in dict_controlled_index
    and call each, to be applied to the controlled indexes which are also inside
    each controller (key) as values that contain controlled indexes of controlled
    component with all their attributes. And return the bus names that contain
    "q_v" controller attached for voltage difference comparison purpose in pf.py.

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
    dict_controlled_index : dictionary
        It is a dynamic dictionary that contains each controller as a key (if
        applied in any component) and each controlled component (filtered to
        only controlled indexes) as values.

    Returns
    -------
    v_mag_pu of voltage_dependent_controller : pandas data frame
    Needed to compare v_mag_pu of the controlled buses with the voltage from
    previous iteration to decide for repeation of pf (in pf.py file).
    """
    v_dep_buses = np.array([])

    for controller in dict_controlled_index.keys():
        # parameter is the controlled indexes dataframe of a components
        for c, index in dict_controlled_index[controller].items():

            # call each controller
            if (controller == 'fixed_cosphi') and (n_trials == 1):
                apply_fixed_cosphi(n, now, c, index)

            elif (controller == 'cosphi_p') and (n_trials == 1):
                apply_cosphi_p(n, now, c, index)

            elif controller == 'q_v':
                v_dep_buses = np.append(v_dep_buses, np.unique(n.df(c).loc[index].loc[(
                    n.df(c).loc[index].control_strategy.isin(["q_v", "p_v"])), 'bus']))
                apply_q_v(n, now, c, index, n_trials_max, n_trials)

    # find the v_mag_pu of buses with v_dependent controller to return
    v_mag_pu_voltage_dependent_controller = n.buses_t.v_mag_pu.loc[
            now, v_dep_buses]

    return v_mag_pu_voltage_dependent_controller


def _set_controller_outputs_to_n(n, c, index, snapshot, ctrl_p_out=False,
                                 ctrl_q_out=False, p_out=None, q_out=None):
    """
    Set the controller outputs to the n (network). The controller outputs
    "p_out" and or "q_out" are set to buses_t.p or buses_t.q and component_t.p
    or component_t.q dataframes.

    Parameter:
    ----------
    n : pypsa.components.Network
        Network
    c : string
        Component name, i.e. 'Load', 'StorageUnit'...
    index : indexes of controlled elements
    snapshot : single snapshot
        Current (now)  element of n.snapshots on which the power flow is run.
    ctrl_p_out : bool default to False
        If ``True``, meaning that p_set is changed by controller due to reactive
        need and controller gives the effective p_out which needs to be set in
        power flow outputs.
    ctrl_q_out : bool default to False
        If ``True``, If controller has reactive power output then this flage
        activates in order to set the controller reactvie power output to the
        network.
    p_out : numpy array
        Active power output of the controller. note: "q_v" and "fixed_cosphi"
        have active power outputs only when a portion of active power is converted
        to reactive power due to reactive power need.
    q_out : numpy array defaut to None
        Reactive power output of the controller
        This behavior is in apply_cosphi and apply_q_v methods.

    Returns
    -------
    None
    """
    # input power before applying controller output to the network
    p_input = n.pnl(c).p.loc[snapshot, index]
    q_input = n.pnl(c).q.loc[snapshot, index]

    # empty dictrionary and adding attribute values to it in each snapshot
    p_q_dict = {}
    if ctrl_p_out:
        p_q_dict['p'] = p_out
    if ctrl_q_out:
        p_q_dict['q'] = q_out

    # setting p_out, q_out to component_t.(p or q) dataframes
    for attr in p_q_dict.keys():
        n.pnl(c)[attr].loc[snapshot, index] = p_q_dict[attr]

        # Finding the change in p and q for the connected buses
        if attr == 'q':
            power_change = -((q_input - n.pnl(c).q).loc[
                    snapshot, index] * n.df(c).loc[
                            index, 'sign']).groupby(n.df(c).loc[
                                    index, 'bus']).sum()

        if attr == 'p':
            power_change = -((p_input - n.pnl(c).p).loc[snapshot, index] *
                             n.df(c).loc[index, 'sign']).groupby(
                                 n.df(c).loc[index, 'bus']).sum()

        # adding the p and q change to the controlled buses
        n.buses_t[attr].loc[snapshot, power_change.index] += power_change


def prepare_controlled_index_dict(n, sub_network, inverter_control, snapshots):
    """
    For components of type "Transformer", "Generator", "Load", "Store" and
    "StorageUnit" collect the indices of controlled elements in the dictionary
    of dictionaries dict_controlled_index:
        - Any exisitng control strategy will be set as a key of dict_controlled_index
        - Each of these keys holds a dictionary as value, with:
            - the types of components it is enabled for as Keys
            - and the related indices of the components as values.
    If a "q_v" or 'p_v' controller is present, n_trial_max is set to 30
    which enables the outer loop of the power flow and sets the maximum allowed
    number of iterations.
    The returned dictionary is used in apply_controller().


    Parameter:
    ----------
    n : pypsa.components.Network
        Network
    sub_network : pypsa.components.Network.sub_network
        network.sub_networks.
    inverter_control : bool, default False
        If ``True``, activates outerloop which applies inverter control strategies
        (control strategy chosen in n.components.control_strategy) in the power flow.
    snapshots : list-like|single snapshot
        A subset or an elements of network.snapshots on which to run
        the power flow, defaults to network.snapshots

    Returns
    -------
    n_trials_max : int
        Shows the maximum allowed power flow iteration for convergance of voltage
        dependent controllers.
    dict_controlled_index : dictionary
        dictionary that contains each controller as key and controlled indexes
        as values.
    """
    n_trials_max = 0
    dict_controlled_index = {}
    ctrl_list = ['', 'q_v', 'cosphi_p', 'fixed_cosphi']
    if inverter_control:
        # loop through loads, generators, storage_units and stores if they exist
        for c in sub_network.iterate_components(n.controllable_one_port_components):

            if (c.df.loc[c.ind].control_strategy != '').any():
                assert (c.df.loc[c.ind].control_strategy.isin(ctrl_list)).all(), (
                        "Not all given types of controllers are supported. "
                        "Elements with unknown controllers are:\n%s\nSupported "
                        "controllers are : %s." % (c.df.loc[c.ind].loc[
                            (~ c.df.loc[c.ind]['control_strategy'].isin(ctrl_list)),
                            'control_strategy'], ctrl_list[1:4]))

                # exclude slack generator to be controlled
                if c.list_name == 'generators':
                    c.df.loc[c.ind].loc[c.df.loc[c.ind].control == 'Slack',
                                        'control_strategy'] = ''
                # if voltage dep. controller exist,find the bus name
                n_trials_max = np.where(
                      c.df.loc[c.ind].control_strategy.isin(['q_v']).any(), 30, 0)

                for i in ctrl_list[1:5]:
                    # building a dictionary for each controller if they exist
                    if (c.df.loc[c.ind].control_strategy == i).any():
                        if i not in dict_controlled_index:
                            dict_controlled_index[i] = {}

                        dict_controlled_index[i][c.name] = c.df.loc[c.ind].loc[(
                                c.df.loc[c.ind].control_strategy == i)].index

                logger.info("We are in %s. These indexes are controlled:\n%s",
                            c.name, dict_controlled_index)

        assert (bool(dict_controlled_index)), (
                "inverter_control is activated but no component is controlled,"
                " please choose the control_strategy in the desired "
                " component indexes. Supported type of controllers are:\n%s."
                % (ctrl_list[1:4]))

    return n_trials_max, dict_controlled_index
