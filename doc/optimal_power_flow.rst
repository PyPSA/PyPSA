######################
 Optimal Power Flow
######################


See the module ``pypsa.opf``.


Non-Linear Optimal Power Flow
==============================

Optimisation with the full non-linear power flow equations is not yet
supported.



Linear Optimal Power Flow
=========================

Optimisation with the linearised power flow equations for (mixed) AC
and DC networks is fully supported.

All constraints and variables are listed below.


Overview
--------
Execute:


``network.lopf(snapshots,solver_name)``

where ``snapshots`` is an iterable of snapshots and ``solver_name`` is a
string, e.g. "gurobi" or "glpk".

The linear OPF module can optimises the dispatch of generation and storage
and the capacities of generation, storage and transmission.

It is assumed that the load is inelastic and must be met in every
snapshot (this will be relaxed in future versions).

The optimisation currently uses continuous variables. MILP unit commitment may be
added in the future.

The objective function is the total system cost for the snapshots
optimised.

Each snapshot can be given a weighting :math:`w_t` to represent
e.g. multiple hours.

Each transmission asset has a capital cost.

Each generation and storage asset has a capital cost and a marginal cost.


WARNING: If the transmission capacity is changed in passive networks,
then the impedance will also change (i.e. if parallel lines are
installed). This is NOT reflected in the LOPF, so the network
equations may no longer be valid. Note also that all the expansion is
continuous.


Optimising dispatch only: a market model
----------------------------------------

Capacity optimisation can be turned off so that only the dispatch is
optimised, like a short-run electricity market model.

For simplified transmission representation using Net Transfer
Capacities (NTCs), there is a Link component which does controllable
power flow like a transport model (and can also represent a
point-to-point HVDC link).



Optimising total annual system costs
------------------------------------

To minimise long-run annual system costs for meeting an inelastic electrical
load, capital costs for transmission and generation should be set to
the annualised investment costs in e.g. EUR/MW/a, marginal costs for
dispatch to e.g. EUR/MWh and the weightings (now with units hours per
annum, h/a) are chosen such that


.. math::
   \sum_t w_t = 8760

In this case the objective function gives total system cost in EUR/a
to meet the total load.




Objective function
------------------

See ``pypsa.opf.define_linear_objective(network,snapshots)``.

The objective function is composed of capital costs :math:`c` for each component and operation costs :math:`o` for generators

.. math::
   \sum_{n,s} c_{n,s} \bar{g}_{n,s} + \sum_{n,s} c_{n,s} \bar{h}_{n,s} + \sum_{l} c_{l} F_l \\
   + \sum_{t} w_t \left[\sum_{n,s} o_{n,s,t} g_{n,s,t} + \sum_{n,s} o_{n,s,t} h_{n,s,t} \right]


Additional variables which do not appear in the objective function are
the storage uptake variable, the state of charge and the voltage angle
for each bus.



Generator constraints
---------------------

These are defined in ``pypsa.opf.define_generator_variables_constraints(network,snapshots)``.

Generator nominal power and generator dispatch for each snapshot may be optimised.


Each generator has a dispatch variable :math:`g_{n,s,t}` where
:math:`n` labels the bus, :math:`s` labels the particular generator at
the bus (e.g. it can represent wind/gas/coal generators at the same
bus in an aggregated network) and :math:`t` labels the time.

It obeys the constraints:

.. math::
   \tilde{g}_{n,s,t}*\bar{g}_{n,s} \leq g_{n,s,t} \leq  \bar{g}_{n,s,t}*\bar{g}_{n,s}

where :math:`\bar{g}_{n,s}` is the nominal power (``generator.p_nom``)
and :math:`\tilde{g}_{n,s,t}` and :math:`\bar{g}_{n,s,t}` are
time-dependent restrictions on the dispatch (per unit of nominal
power) due to e.g. wind availability or power plant de-rating.

For generators with ``generator.dispatch == "variable"`` the per unit
availability :math:`\bar{g}_{n,s,t}` is a time series
``generator_t.p_max_pu``.


For generators with ``generator.dispatch == "flexible"`` the per unit
availability is a constant ``generator.p_max_pu_fixed``.


If the generator's nominal power :math:`\bar{g}_{n,s}` is also the
subject of optimisation (``generator.p_nom_extendable == True``) then limits on the nominal power may also be introduced, e.g.



.. math::
   \bar{g}_{n,s} \leq  \hat{g}_{n,s}



Storage Unit constraints
------------------------

These are defined in ``pypsa.opf.define_storage_variables_constraints(network,snapshots)``.


Storage nominal power and dispatch for each snapshot may be optimised.

The maximum state of charge may not be independently optimised at the moment.

The storage nominal power is given by :math:`\bar{h}_{n,s}`.

In contrast to the generator, which has one time-dependent variable, each storage unit has three:

The storage dispatch :math:`h_{n,s,t}` (when it depletes the state of charge):

.. math::
   0 \leq h_{n,s,t} \leq \bar{h}_{n,s}

The storage uptake :math:`f_{n,s,t}` (when it increases the state of charge):

.. math::
   0 \leq f_{n,s,t} \leq  \bar{h}_{n,s}

and the state of charge itself:

.. math::
   0\leq soc_{n,s,t} \leq r_{n,s} \bar{h}_{n,s}

where :math:`r_{n,s}` is the number of hours at nominal power that fill the state of charge.

The variables are related by

.. math::
   soc_{n,s,t} = \eta_{\textrm{stand};n,s}^{w_t} soc_{n,s,t-1} + \eta_{\textrm{store};n,s} w_t f_{n,s,t} -  \eta_{\textrm{dispatch};n,s} w_t h_{n,s,t} + w_t\textrm{inflow}_{n,s,t} - w_t\textrm{spillage}_{n,s,t}

:math:`\eta_{\textrm{stand};n,s}` is the standing losses dues to
e.g. thermal losses for thermal
storage. :math:`\eta_{\textrm{store};n,s}` and
:math:`\eta_{\textrm{dispatch};n,s}` are the efficiency losses for
power going into and out of the storage unit.



There are two options for specifying the initial state of charge :math:`soc_{n,s,t=-1}`: you can set
``storage_unit.cyclic_state_of_charge = False`` (the default) and the value of
``storage_unit.state_of_charge_initial`` in MWh; or you can set
``storage_unit.cyclic_state_of_charge = True`` and then
the optimisation assumes :math:`soc_{n,s,t=-1} = soc_{n,s,t=|T|-1}`.



If in the time series ``storage_unit_t.state_of_charge_set`` there are
values which are not NaNs, then it will be assumed that these are
fixed state of charges desired for that time :math:`t` and these will
be added as extra constraints. (A possible usage case would be a
storage unit where the state of charge must empty every day.)



Passive branch flows
------------------------

See ``pypsa.opf.define_passive_branch_flows(network,snapshots)`` and
``pypsa.opf.define_passive_branch_constraints(network,snapshots)`` and ``pypsa.opf.define_branch_extension_variables(network,snapshots)``.





For lines and transformers, whose power flows according to impedances,
the power flow :math:`f_{l,t}` in AC networks is given by the difference in voltage
angles :math:`\theta_{n,t}` at bus0 and :math:`\theta_{m,t}` at bus1 divided by the series reactance :math:`x_l`


.. math::
   f_{l,t} = \frac{\theta_{n,t} - \theta_{m,t}}{x_l}


(For DC networks, replace the voltage angles by the difference in voltage magnitude :math:`\delta V_{n,t}` and the series reactance by the series resistance :math:`r_l`.)


This flow is the limited by the capacity :math:``F_l`` of the line


.. math::
   |f_{l,t}| \leq F_l

Note that if :math:`F_l` is also subject to optimisation
(``branch.s_nom_extendable == True``), then the impedance :math:`x` of
the line is NOT automatically changed with the capacity (to represent
e.g. parallel lines being added).

There are two choices here:

Iterate the LOPF again with the updated impedances (see e.g. `<http://www.sciencedirect.com/science/article/pii/S0360544214000322#>`_).

Use a different program which can do MINLP to represent the changing
line impedance.




Controllable branch flows
-------------------------

See ``pypsa.opf.define_controllable_branch_flows(network,snapshots)``
and ``pypsa.opf.define_branch_extension_variables(network,snapshots)``.


For links, whose power flow is controllable, there is simply an
optimisation variable for each component which satisfies

.. math::
   |f_{l,t}| \leq F_l



Nodal power balances
--------------------

See ``pypsa.opf.define_nodal_balances(network,snapshots)``.

This is the most important equation, which guarantees that the power
balances at each bus :math:`n` for each time :math:`t`.

.. math::
   \sum_{s} g_{n,s,t} + \sum_{s} h_{n,s,t} - \sum_{s} f_{n,s,t} - \sum_{s} \ell_{n,s,t} + \sum_{l} K_{nl} f_{l,t} = 0

Where :math:`\ell_{n,s,t}` is the exogenous load at each node (``load.p_set``) and the incidence matrix :math:`K_{nl}` for the graph takes values in :math:`\{-1,0,1\}` depending on whether the branch :math:`l` ends or starts at the bus.

CO2 constraint
--------------

See ``pypsa.opf.define_co2_constraint(network,snapshots)``.

This depends on the power plant efficiency and specific CO2 emissions
of the fuel source.


Variables and notation summary
------------------------------

TODO - see objective function.

:math:`n \in N = \{0,\dots |N|-1\}` label the buses

:math:`t \in T = \{0,\dots |T|-1\}` label the snapshots

:math:`l \in L = \{0,\dots |L|-1\}` label the branches

:math:`s \in S = \{0,\dots |S|-1\}` label the different generator/storage types at each bus


Inputs
------

For the linear optimal power flow, the following data for each component
are used. For almost all values, defaults are assumed if not
explicitly set. For the defaults and units, see :doc:`components`.

network{snapshot_weightings,co2_limit}

bus.{v_nom}

load.{p_set}

generator.{dispatch, p_nom, p_nom_extendable, p_nom_min, p_nom_max, p_min/max_pu_fixed (if control is "flexible"), p_min/max_pu (if control is "variable"), marginal_cost, capital_cost, efficiency}

storage_unit.{dispatch, p_nom, p_nom_extendable, p_nom_min, p_nom_max, p_min/max_pu_fixed, marginal_cost, capital_cost, efficiency*, standing_loss, inflow, state_of_charge_set, max_hours, state_of_charge_initial, cyclic_state_of_charge}

line.{x, s_nom, s_nom_extendable, s_nom_min, s_nom_max, capital_cost}

transformer.{x, s_nom, s_nom_extendable, s_nom_min, s_nom_max, capital_cost}

link.{p_set, s_nom, s_nom_extendable, s_nom_min, s_nom_max, capital_cost}

source.{co2_emissions}

Note that for lines and transformers you MUST make sure that
:math:`x` is non-zero, otherwise the bus admittance matrix will be singular.

Outputs
-------

bus.{v_mag_pu, v_ang, p, marginal_price}

load.{p}

generator.{p, p_nom_opt}

storage_unit.{p, p_nom_opt, state_of_charge, spill}

line.{p0, p1, s_nom_opt}

transformer.{p0, p1, s_nom_opt}

link.{p0, p1, s_nom_opt}
