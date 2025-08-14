###################
System Optimization
###################


Overview
--------

PyPSA can optimize the following problems:

1. **Economic Dispatch (ED)** market model with unit commitment and storage operation with perfect foresight or rolling horizon,

2. **Linear Optimal Power Flow (LOPF)** with network constraints for Kirchhoff's Voltage Law (KVL) and Kirchhoff's Current Law (KCL),

3. **Security-Constrained Linear Optimal Power Flow (SCLOPF)** for network contingency analysis,

4. **Capacity Expansion Planning (CEP)** with single or multiple investment periods and system-wide constraints, and

5. **Modelling-to-Generate-Alternatives (MGA)** for near-optimal space exploration.

These problems build on each other, e.g., capacity expansion planning models
include economic dispatch and linear optimal power flow constraints.

Thereby, PyPSA can co-optimise the dispatch of generation, conversion and
storage technologies, and the capacities of generation, storage, conversion and
transmission infrastructure.

In any case, the objective is to minimize the total system cost for the
snapshots selected, where each snapshot is given a weighting :math:`w_t` to
represent e.g. multiple hours.

Depending on the parameters provided (e.g. whether components are extendable or
committable), the optimisation is formulated as a **linear program (LP)**,
**quadratic program (QP)** or **mixed-integer linear program (MILP)**. Most
variables are continuous, but unit commitment constraints and block-sized
investments can be modelled with binary variables. and solved with the solver of
your choice (e.g. Gurobi, CPLEX, or HiGHS) by executing:

.. code:: python

    n.optimize(solver_name="highs", solver_options={"solver": "ipm"})


where ``solver_name`` is a string, e.g. "gurobi" or "highs", ``solver_options``
is a dictionary of solver-specific flags to pass to the solver.


Market model
^^^^^^^^^^^^

Capacity optimisation can be turned off so that only the dispatch is optimised,
like a short-run electricity market model. For simplified transmission
representation using Net Transfer Capacities (NTCs), there is a ``Link``
component which does controllable power flow like a transport model (and can
also represent a point-to-point HVDC link).


Capacity expansion model
^^^^^^^^^^^^^^^^^^^^^^^^

To minimise long-run annual system costs for meeting a perfectly inelastic
demand, capital costs for components should be set to the annualised investment
costs in e.g. EUR/MW/a, marginal costs for dispatch to e.g. EUR/MWh and the
weightings (now with units hours per annum, h/a) are chosen such that


.. math::
   \sum_t w_t = 8760

In this case the objective function gives total system cost in EUR/a
to meet the total demand.

Stochastic optimisation
^^^^^^^^^^^^^^^^^^^^^^^

For the very simplest stochastic optimisation you can use the
weightings ``w_t`` as probabilities for the snapshots, which can
represent different load/weather conditions. More sophisticated
functionality is planned.


Variables
---------

.. csv-table::
  :widths: 20 50
  :delim: ;

  :math:`n \in N = \{0,\dots |N|-1\}`; label the buses
  :math:`t \in T = \{0,\dots |T|-1\}`; label the snapshots
  :math:`l \in L = \{0,\dots |L|-1\}`; label the branches
  :math:`s \in S = \{0,\dots |S|-1\}`; label the different generator/storage types at each bus
  :math:`w_t`; weighting of time :math:`t` in the objective function
  :math:`g_{n,s,t}`; dispatch of generator :math:`s` at bus :math:`n` at time :math:`t`
  :math:`\bar{g}_{n,s}`; nominal power of generator :math:`s` at bus :math:`n`
  :math:`\bar{g}_{n,s,t}`; availability of  generator :math:`s` at bus :math:`n` at time :math:`t` per unit of nominal power
  :math:`u_{n,s,t}`; binary status variable for generator with unit commitment
  :math:`suc_{n,s,t}`; start-up cost if generator with unit commitment is started at time :math:`t`
  :math:`sdc_{n,s,t}`; shut-down cost if generator with unit commitment is shut down at time :math:`t`
  :math:`c_{n,s}`; capital cost of extending generator nominal power by one MW
  :math:`o_{n,s}`; marginal cost of dispatch generator for one MWh
  :math:`f_{l,t}`; flow of power in branch :math:`l` at time :math:`t`
  :math:`F_{l}`; capacity of branch :math:`l`
  :math:`\eta_{n,s}`; efficiency of generator :math:`s` at bus :math:`n`
  :math:`\eta_{l}`; efficiency of controllable link :math:`l`
  :math:`e_s`; CO2-equivalent-tonne-per-MWh of the fuel carrier :math:`s`


Objective function
------------------

The objective function is composed of capital costs :math:`c` for each component and operation costs :math:`o` for most components

.. math::
  :nowrap:

    \begin{gather*}
    \sum_{n,s} c_{n,s} \bar{g}_{n,s} + \sum_{n,s} c_{n,s} \bar{h}_{n,s} + \sum_{l} c_{l} F_l \\
    + \sum_{t} w_t \left[\sum_{n,s} o_{n,s,t} g_{n,s,t} + \sum_{n,s} o_{n,s,t} h_{n,s,t} \right]
    + \sum_{t} \left[suc_{n,s,t} + sdc_{n,s,t} \right]
    \end{gather*}




Additional variables which do not appear in the objective function are
the storage uptake variable and the state of charge.


Generator constraints
---------------------

Generator nominal power and generator dispatch for each snapshot may be optimised.


Each generator has a dispatch variable :math:`g_{n,s,t}` where
:math:`n` labels the bus, :math:`s` labels the particular generator at
the bus (e.g. it can represent wind/gas/coal generators at the same
bus in an aggregated network) and :math:`t` labels the time.

It obeys the constraints:

.. math::
   \tilde{g}_{n,s,t}*\bar{g}_{n,s} \leq g_{n,s,t} \leq  \bar{g}_{n,s,t}*\bar{g}_{n,s}

where :math:`\bar{g}_{n,s}` is the nominal power (``n.generators.p_nom``)
and :math:`\tilde{g}_{n,s,t}` and :math:`\bar{g}_{n,s,t}` are
time-dependent restrictions on the dispatch (per unit of nominal
power) due to e.g. wind availability or power plant de-rating.

For generators with time-varying ``p_max_pu`` in ``n.generators_t`` the per unit
availability :math:`\bar{g}_{n,s,t}` is a time series.


For generators with static ``p_max_pu`` in ``n.generators`` the per unit
availability is a constant.


If the generator's nominal power :math:`\bar{g}_{n,s}` is also the
subject of optimisation (``n.generators.p_nom_extendable == True``) then
limits ``n.generators.p_nom_min`` and ``n.generators.p_nom_max`` on the
installable nominal power may also be introduced, e.g.



.. math::
   \tilde{g}_{n,s} \leq    \bar{g}_{n,s} \leq  \hat{g}_{n,s}


Storage Unit constraints
-------------------------

Storage nominal power and dispatch for each snapshot may be optimised.

With a storage unit the maximum state of charge may not be independently optimised from the maximum power output (they are linked by the maximum hours parameter ``max_hours``) and the maximum power output is linked to the maximum power input.

.. note::
   To optimise these capacities independently, build a storage unit out of the more fundamental ``Store`` and ``Link`` components.

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
   soc_{n,s,t} = \eta_{\textrm{stand};n,s}^{w_t} soc_{n,s,t-1} + \eta_{\textrm{store};n,s} w_t f_{n,s,t} -  \eta^{-1}_{\textrm{dispatch};n,s} w_t h_{n,s,t} + w_t\textrm{inflow}_{n,s,t} - w_t\textrm{spillage}_{n,s,t}

:math:`\eta_{\textrm{stand};n,s}` is the standing losses dues to
e.g. thermal losses for thermal
storage. :math:`\eta_{\textrm{store};n,s}` and
:math:`\eta_{\textrm{dispatch};n,s}` are the efficiency losses for
power going into and out of the storage unit.



There are two options for specifying the initial state of charge :math:`soc_{n,s,t=-1}`: you can set
``n.storage_units.cyclic_state_of_charge = False`` (the default) and the value of
``n.storage_units.state_of_charge_initial`` in MWh; or you can set
``n.storage_units.cyclic_state_of_charge = True`` and then
the optimisation assumes :math:`soc_{n,s,t=-1} = soc_{n,s,t=|T|-1}`.



If in the time series ``n.storage_units_t.state_of_charge_set`` there are
values which are not NaNs, then it will be assumed that these are
fixed state of charges desired for that time :math:`t` and these will
be added as extra constraints. (A possible usage case would be a
storage unit where the state of charge must empty every day.)


Store constraints
------------------

Store nominal energy and dispatch for each snapshot may be optimised.

The store nominal energy is given by :math:`\bar{e}_{n,s}`.

The store has two time-dependent variables:

The store dispatch :math:`h_{n,s,t}`:

.. math::
   -\infty \leq h_{n,s,t} \leq +\infty

and the energy:

.. math::
   \tilde{e}_{n,s} \leq e_{n,s,t} \leq \bar{e}_{n,s}


The variables are related by

.. math::
   e_{n,s,t} = \eta_{\textrm{stand};n,s}^{w_t} e_{n,s,t-1} - w_t h_{n,s,t}

:math:`\eta_{\textrm{stand};n,s}` is the standing losses dues to
e.g. thermal losses for thermal
storage.

There are two options for specifying the initial energy
:math:`e_{n,s,t=-1}`: you can set
``n.stores.e_cyclic = False`` (the default) and the
value of ``n.stores.e_initial`` in MWh; or you can
set ``n.stores.e_cyclic = True`` and then the
optimisation assumes :math:`e_{n,s,t=-1} = e_{n,s,t=|T|-1}`.


.. _opf-links:

Link constraints
----------------

For links, whose flow is controllable, there is an
optimisation variable for each component which satisfies

.. math::
   |f_{l,t}| \leq F_l

If the link flow is positive :math:`f_{l,t} > 0` then it withdraws
:math:`f_{l,t}` from ``bus0`` and feeds in :math:`\eta_l f_{l,t}` to
``bus1``, where :math:`\eta_l` is the link efficiency.

If additional output buses ``bus{i}`` for :math:`i=2,3,\dots` are
defined (i.e. ``bus2``, ``bus3``, etc) and their associated
efficiencies ``efficiency{i}``, i.e. :math:`\eta_{i,l}`, then at
``bus{i}`` the feed-in is :math:`\eta_{i,l} f_{l,t}`. See also
:ref:`components-links-multiple-outputs`.


Line and Transformer constraints
--------------------------------

For lines and transformers, whose power flows according the impedances, the
power flow :math:`f_{l,t}` in AC networks is governed by the cycle-based
formulation of Kirchhoff's Voltage Law (KVL)


.. math::
    \sum_l C_{l,c} x_l f_{l,t} = 0  \hspace{.4cm} \forall\, c

where :math:`C` is a cycle basis matrix of the network graph and :math:`x_l` is
the series reactance.

.. note::
   For DC networks, replace the series reactance :math:`x_l` by the series resistance :math:`r_l`.

While there are different formulations of KVL, the cycle-based formulation was
found to be much faster than other formulations due to its sparsity, as shown in
`Linear Optimal Power Flow Using Cycle Flows
<https://www.sciencedirect.com/science/article/abs/pii/S0378779617305138>`_.

This formulation defines the same feasible space as the standard formulation
based on voltage angles that is commonly found in textbooks (B-Theta) or the
formulation based on Power Transfer Distribution Factors (PTDFs).

This flow is the limited by the capacity :math:``F_l`` of the line


.. math::
   |f_{l,t}| \leq F_l

.. note::
  If :math:`F_l` is also subject to optimisation
  (``branch.s_nom_extendable -- True``), then the impedance :math:`x` of
  the line is NOT automatically changed with the capacity (to represent
  e.g. parallel lines being added).


.. _unit-commitment:

Unit commitment constraints
---------------------------

.. note::
  The unit commitment constraints are implemented for the ``Generator`` and ``Link`` components.

The implementation is a complete implementation of the unit commitment
constraints defined in Chapter 4.3 of `Convex Optimization of Power Systems
<http://www.cambridge.org/de/academic/subjects/engineering/control-systems-and-optimization/convex-optimization-power-systems>`_
by Joshua Adam Taylor (CUP, 2015).


Unit commitment can be turned on for any generator or link by setting
``committable`` to ``True``. This introduces new binary status variables
:math:`u_{n,s,t} \in \{0,1\}`, saved in ``n.generators_t.status``, which
indicates whether the generator/link is running (1) or not (0) in period
:math:`t`. The restrictions on generator/link  output now become:

.. math::
   u_{n,s,t}*\tilde{g}_{n,s,t}*\bar{g}_{n,s} \leq g_{n,s,t} \leq   u_{n,s,t}*\bar{g}_{n,s,t}*\bar{g}_{n,s} \hspace{.5cm} \forall\, n,s,t

so that if :math:`u_{n,s,t} = 0` then also :math:`g_{n,s,t} = 0`.

.. note::
   Note that a generator/link cannot be both extendable (``n.generators.p_nom_extendable == True``) and committable (``n.generators.committable == True``) because of the coupling of the variables :math:`u_{n,s,t}`
   and :math:`\bar{g}_{n,s}` here.

If the minimum up time :math:`T_{\textrm{min_up}}` (``n.generators.min_up_time``) is set then we have for generic times

.. math::
   \sum_{t'=t}^{t+T_\textrm{min_up}} u_{n,s,t'}\geq T_\textrm{min_up} (u_{n,s,t} - u_{n,s,t-1})   \hspace{.5cm} \forall\, n,s,t

i.e. if the generator/link has just started up at time :math:`t` then :math:`u_{n,s,t-1} = 0`, :math:`u_{n,s,t} = 1` and :math:`u_{n,s,t} - u_{n,s,t-1} = 1`, so that it has to run for at least :math:`T_{\textrm{min_up}}` periods.

The generator/link may have been up for some periods before the ``snapshots`` simulation period. If the up-time before ``snapshots`` starts is less than the minimum up-time, then the generator/link  is forced to be up for the difference at the start of ``snapshots``. If the start of ``snapshots`` is the start of ``n.snapshots``, then the up-time before the simulation is read from the input variable ``n.generators.up_time_before``.  If ``snapshots`` falls in the middle of ``n.snapshots``, then PyPSA assumes the statuses for hours before ``snapshots`` have been set by previous simulations, and reads back the previous up-time by examining the previous statuses. If the start of ``snapshots`` is very close to the start of ``n.snapshots``, it will also take account of ``n.generators.up_time_before`` as well as the statuses in between.


At the end of ``snapshots`` the minimum up-time in the constraint is only enforced for the remaining snapshots, if the number of remaining snapshots is less than :math:`T_{\textrm{min_up}}`.


Similarly if the minimum down time :math:`T_{\textrm{min_down}}` (``n.generators.min_down_time``) is set then we have

.. math::
   \sum_{t'=t}^{t+T_\textrm{min_down}} (1-u_{n,s,t'})\geq T_\textrm{min_down} (u_{n,s,t-1} - u_{n,s,t})   \hspace{.5cm} \forall\, n,s,t

You can also define ``n.generators.down_time_before`` for periods before ``n.snapshots``, analogous to the up time.

For non-zero start up costs :math:`suc_{n,s}` a new variable :math:`suc_{n,s,t} \geq 0` is introduced for each time period :math:`t` and added to the objective function.  The variable satisfies

.. math::
   suc_{n,s,t} \geq suc_{n,s} (u_{n,s,t} - u_{n,s,t-1})   \hspace{.5cm} \forall\, n,s,t

so that it is only non-zero if :math:`u_{n,s,t} - u_{n,s,t-1} = 1`, i.e. the generator/link  has just started, in which case the inequality is saturated :math:`suc_{n,s,t} = suc_{n,s}`. Similarly for the shut down costs :math:`sdc_{n,s,t} \geq 0` we have

.. math::
   sdc_{n,s,t} \geq sdc_{n,s} (u_{n,s,t-1} - u_{n,s,t})   \hspace{.5cm} \forall\, n,s,t




.. _ramping:

Ramping constraints
-------------------

.. note::
  The ramping constraints are implemented for the ``Generator`` and ``Link`` components.

The implementation follows Chapter 4.3 of `Convex Optimization of Power Systems <http://www.cambridge.org/de/academic/subjects/engineering/control-systems-and-optimization/convex-optimization-power-systems>`_ by
Joshua Adam Taylor (CUP, 2015).

Ramp rate limits can be defined for generators and links for increasing output
:math:`ru_{n,s}` and decreasing output :math:`rd_{n,s}`. By
default these are null and ignored. They should be given per unit of
the generator nominal rating. The generator dispatch then obeys

.. math::
   -rd_{n,s} * \bar{g}_{n,s} \leq (g_{n,s,t} - g_{n,s,t-1}) \leq ru_{n,s} * \bar{g}_{n,s}

for :math:`t \in \{1,\dots |T|-1\}`.

For generators/links with unit commitment you can also specify ramp limits
at start-up :math:`rusu_{n,s}` and shut-down :math:`rdsd_{n,s}`

.. math::
  :nowrap:

  \begin{gather*}
  \left[ -rd_{n,s}*u_{n,s,t} -rdsd_{n,s}(u_{n,s,t-1} - u_{n,s,t})\right] \bar{g}_{n,s} \\
  \leq (g_{n,s,t} - g_{n,s,t-1}) \leq  \\
  \left[ru_{n,s}*u_{n,s,t-1} +   rusu_{n,s} (u_{n,s,t} - u_{n,s,t-1})\right]\bar{g}_{n,s}
  \end{gather*}


.. _nodal-power-balance:

Energy flow balances
--------------------

The energy balance equations are the most important constraints, which guarantees that the energy
flow balances at each bus :math:`n` for each time :math:`t`.

.. math::
   \sum_{s} g_{n,s,t} + \sum_{s} h_{n,s,t} - \sum_{s} f_{n,s,t} - \sum_{l} K_{nl} f_{l,t} = \sum_{s} d_{n,s,t} \hspace{.4cm} \leftrightarrow  \hspace{.4cm} w_t\lambda_{n,t}

Where :math:`d_{n,s,t}` is the exogenous load at each node (``n.loads.p_set``) and the incidence matrix :math:`K_{nl}` for the graph takes values in :math:`\{-1,0,1\}` depending on whether the branch :math:`l` ends or starts at the bus. :math:`\lambda_{n,t}` is the shadow price of the constraint, i.e. the locational marginal price, stored in ``n.buses_t.marginal_price``.


The bus's role is to enforce energy conservation for all elements
feeding in and out of it (i.e. like Kirchhoff's Current Law).

.. image:: ../img/buses.png


.. _multi-horizon:

Multiple investment periods
---------------------------

In general, there are two different methods of pathway optimisation with perfect
foresight. These differ in the way of accounting the investment costs:

* In the first case (type I), the complete overnight investment costs are applied.
* In the second case (type II), the investment costs are annualised over the years, in which an asset is active (depending on the build year and lifetime).

Type II is used in PyPSA since it allows a separation of the discounting over
different years and the end-of-horizon effects are smaller compared to method I.
For a more detailed comparison of the two methods and a reference to other energy
system models see `<https://nworbmot.org/energy/multihorizon.pdf>`_.

.. note::
 Be aware, that the attribute ``capital_cost`` represents the annualised investment costs
 not the overnight investment costs for the multi-investment.

Multi-year investment can be passed by setting the argument
``multi_investment_periods`` when calling the
``n.optimize(multi_investment_periods=True)``. For the pathway
optimisation ``snapshots`` have to be a pandas.MultiIndex, with the first level
as a subset of the investment periods.

The investment periods are defined in the component ``investment_periods``.
They have to be integer and increasing (e.g. ``[2020, 2030, 2040, 2050]``).
The investment periods can be weighted both in time called ``years``
(e.g. for global constraints such as :math:`\mathrm{CO}_2` emissions) and in the objective function
``objective`` (e.g. for a social discount rate) using the
``investment_period_weightings``.

The objective function is then expressed by

.. math::
   \min \sum_{a \in A} w^o_a [\sum_{s | b_s<=a<b_s+L_s} (c_{s,a} G_s + \sum_t w^\tau_{a,t} o_{s,a,t}g_{s,a,t})]  .

Where :math:`A` are the investment periods, :math:`w^o_a` the objective weighting of the investment period, :math:`b_s` is the build year of an
asset :math:`s` with lifetime :math:`L_s`, :math:`c_{s,a}` the annualised
investment costs, :math:`o_{s,a, t}` the operational costs and :math:`w^\tau_{s,a}`
the temporal weightings (including snapshot objective weightings and investment
period temporal weightings).

The general procedure for modelling multi-investment periods in PyPSA is to add
an asset for each investment period, in which its capacity should be expandable.
For example, if you want to optimise onshore wind development in the period 2025-2040
with investment periods every 5 years, you add a generator with a corresponding
construction year and lifetime for each investment period
(``onwind-2025``, ``onwind-2030``, ``onwind-2035``, ``onwind-2040``).
This allows one to specify different technological assumptions for the respective
investment period (for example, decreasing investment costs, increasing efficiencies,
improved capacity factors due to higher hub heights of wind turbines, extended lifetimes).
The generators are only available for use after the year of construction and before
the end of their lifetime, for example, the onwind-2030 generator built in 2030
cannot contribute to electricity generation in the 2025 investment period.
To ensure that the technical potential for onshore wind in the region is not
exceeded by the 4 onshore wind generators in our example, one has to add an
additional global constraint (``type=tech_capacity_expansion_limit``, see further description above).

Note that the ``capital_cost`` of the assets is now the fixed annual costs, including annuity and FOM.

`Example jupyter notebook for multi-investment
<https://pypsa.readthedocs.io/en/latest/examples/multi-investment-optimisation.html>`_ and python
script ``examples/multi-decade-example.py``.


.. _global-constraints-opf:

Global constraints
------------------

Global constraints apply to more than one component.
They are activated if a
global constraint with the corresponding ``type`` is added to the n.
By default, the constraint applies to all investment periods. For multi-decade
optimisation, a global constraint can be set for one investment period only
(e.g. a :math:`\mathrm{CO}_2` limit for a specific investment year) by specifying this in the
attribute ``investment_period``. The shadow price of each global constraint is
stored in  :math:`\mu` which is an output of the optimisation stored in ``n.global_constraints.mu``.

Primary energy
^^^^^^^^^^^^^^

The primary energy constraints (``type=primary_energy``) depend on the power plant efficiency and carrier-specific attributes such as
specific :math:`\mathrm{CO}_2` emissions.


Suppose there is a global constraint defined for :math:`\mathrm{CO}_2` emissions with
sense ``<=`` and constant :math:`\textrm{CAP}_{CO2}`. Emissions can come
from generators whose energy carriers have :math:`\mathrm{CO}_2` emissions and from
stores and storage units whose storage medium releases or absorbs :math:`\mathrm{CO}_2`
when it is converted. Only stores and storage units with non-cyclic
state of charge that is different at the start and end of the
simulation can contribute.

If the specific emissions of energy carrier :math:`s` is :math:`e_s`
(``carrier.co2_emissions``) :math:`\mathrm{CO}_2`-equivalent-tonne-per-MWh and the
generator with carrier :math:`s` at node :math:`n` has efficiency
:math:`\eta_{n,s}` then the :math:`\mathrm{CO}_2` constraint is

.. math::
   \sum_{n,s,t} \frac{1}{\eta_{n,s}} w_t\cdot g_{n,s,t}\cdot e_{n,s} + \sum_{n,s}\left(e_{n,s,t=-1} - e_{n,s,t=|T|-1}\right) \cdot e_{n,s} \leq  \textrm{CAP}_{CO2}  \hspace{.4cm} \leftrightarrow  \hspace{.4cm} \mu

The first sum is over generators; the second sum is over stores and
storage units. :math:`\mu` is the shadow price of the constraint,
i.e. the :math:`\mathrm{CO}_2` price in this case.

Emission targets for single investment periods when performing multi-horizon
investment planning can be set by specifying the attribute
``investment_period``.

Transmission volume expansion limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This global constraint can limit the maximum line volume expansion in MWkm
(``type=transmission_volume_expansion_limit``). Possible carriers are 'AC' and 'DC'.

Transmission cost expansion limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This global constraint can limit the maximum cost of line expansion
(``type=transmission_expansion_cost_limit``). Possible carriers are 'AC' and 'DC'.


Technology capacity expansion limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This global constraint can limit the maximum summed capacity of active assets
of a carrier (e.g. onshore wind) for an investment period at a chosen node
(``type=tech_capacity_expansion_limit``).
This constraint is mainly used for multi-decade investment planning. It can represent land
resource or building rate restrictions for a technology in a certain region.
Currently, only the capacities of extendable generators have to be below the set limit.

For example, the capacities of all onshore wind generators (``carrier_attribute="onshore wind"``) at a certain bus
(``bus="DE"``) should be smaller (``sense="<="``) than the technical potential for onshore wind
in the specific region (``constant=Limit``). Then the technology capacity expansion constraint is

.. math::
  \sum_{s | b_s<=a<b_s+L_s} \bar{g}_{n,s} \leq  \textrm{Limit} \hspace{.4cm} a \in A.

Where :math:`A` are the investment periods,
:math:`s` are all extendable generators of the specified carrier, :math:`b_s` is the build year of an
asset :math:`s` with lifetime :math:`L_s`.

The constraint can also be formulated with the opposite sense, so that,
a minimum expansion of a certain technology is required on a certain bus.

Operational Limit
^^^^^^^^^^^^^^^^^

This global constraint can limit the net production of a carrier taking into
account generator, storage units and stores (``type=operational_limit``).


Growth limit per carrier
^^^^^^^^^^^^^^^^^^^^^^^^

A growth limit per carrier which constraints new installed capacities for each
investment period can be defined by setting the attribute ``max_growth`` for the
PyPSA component ``carrier``.


Problem extensions
-----------------------------

Through the ``pypsa.optimization.abstract`` module, PyPSA provides a number of problem extensions:

Rolling-horizon optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :py:meth:`pypsa.Network.optimize:optimize_with_rolling_horizon` allows to
optimize system dispatch in a sequential rolling-horizon fashion. This is useful
for chunking large operational problems or considering myopic operational
foresight.

Iterative transmission expansion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the transmission capacity is changed in passive networks, then the impedance
will also change (i.e. if parallel lines are installed). This is not reflected
in the ordinary optimization, however
:py:meth:`pypsa.Network.optimize.optimize_transmission_expansion_iteratively`
covers this through an iterative process as done `Hagspiel et al. (2014)
<http://www.sciencedirect.com/science/article/pii/S0360544214000322#>`_.


Security-Constrained Power Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure that the optimized power system is robust against line failures,
security-constrained optimization through
:py:meth:`pypsa.Network.optimize.optimize_security_constrained` enforces
security margins for power flow on ``Line`` components. See
:doc:`/user-guide/contingency-analysis` for more details.

Modelling-to-Generate-Alternatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :py:meth:`pypsa.Network.optimize.optimize_mga` runs
modelling-to-generate-alternatives (MGA) on network to find near-optimal
solutions.


Custom constraints
------------------


Custom constraints are important because they allow users to tailor optimization problems to specific requirements or scenarios. By adding custom constraints, users can model more complex or realistic situations that may not be captured by the default optimization formulations provided by PyPSA.


To build custom constraints, users can access and modify the Linopy model instance associated with the PyPSA ``n``. This model instance contains all variables, constraints, and the objective function of the optimization problem. Users can directly add, remove, or modify variables and constraints as needed.

Given a network ``n`` and the corresponding model instance ``m``, some key functions used in the code for working with custom constraints include:

* :py:meth:`Network.optimize.create_model`: Creates a Linopy model instance for the PyPSA ``n``.
* :py:meth:`linopy.model.variables`: Accesses the optimization variables of the Linopy model instance.
* :py:meth:`linopy.model.add_variables``: Adds custom variables to the Linopy model instance.
* :py:meth:`linopy.model.add_constraints`: Adds custom constraints to the Linopy model instance.
* :py:meth:`Network.optimize.solve_model`: Solves the optimization problem using the current Linopy model instance and updates the PyPSA network with the solution.


A typical workflow starts with creating a Linopy model instance for a PyPSA network using the ``n.optimize.create_model()`` function. This model instance contains all the optimization variables, constraints, and the objective function, which can be accessed and modified to incorporate custom constraints::

  m = n.optimize.create_model()

This will create a Linopy model instance ``m`` for the PyPSA network ``n`` and is also accessible using the ``n.model`` attribute.
Accessing and combining variables is an essential part of creating custom constraints. You can access variables using the Linopy model instance's `variables` attribute, which provides a dictionary-like structure containing the variables associated with each component in the n. For example, you can access generator active power variables using::

  gen_p = m.variables["Generator-p"]

This will return an array of variables, of class ``linopy.Variable`` which defines a variable reference for each generator and snapshot in the n. The ``Variable`` type is closely related to ``xarray.DataArray`` and ``pandas.DataFrame``, and can be used in similar ways.
To create custom constraints, you may need to combine variables, such as generator output and line flow variables, using mathematical operations like addition, subtraction, multiplication, and division.

When defining a custom constraint, you can create a Linopy expression representing the relationship between the variables involved in the constraint. The expression can be created using standard Python operators like ``==``, ``>=``, and ``<=``. For example, if you want to create a constraint that forces the total generation at a bus to be at least 80% of the total demand, you can create an expression like:

.. code-block:: python

  bus = n.generators.bus.to_xarray()
  total_generation = gen_p.groupby(bus).sum().sum("snapshot")
  total_demand = n.loads_t.p_set.sum().sum()
  constraint_expression = total_generation >= 0.8 * total_demand

Note that in the Linopy formulation variable expressions stand on the left-hand-side of the constraint, while the right-hand-side is a constant value.
After defining the custom constraint expression, add it to the Linopy model using the ``m.add_constraints()`` function, providing a name for the constraint to facilitate further modifications or analysis::

  m.add_constraints(constraint_expression, name="Bus-minimum_generation_share")

Once you have added your custom constraints to the Linopy model, use the ``n.optimize.solve_model()`` function to solve the optimization problem. This function considers your custom constraints while solving the optimization problem and updates the PyPSA network with the resulting solution::

  n.optimize.solve_model()

By following this workflow, you can create and modify optimization problems with custom constraints that better represent your specific requirements and scenarios using PyPSA and Linopy.

Note that alternatively the ``extra_functionality`` argument can be used in the ``optimize`` function to add custom functions to the optimization problem. The function is called after the model is created and before it is solved. It takes the network and the snapshots as arguments. However, for ease of use, we recommend using the workflow described above.

Further examples can be found in the examples section of the PyPSA documentation and in the `Linopy documentation <https://linopy.readthedocs.io/en/latest/>`_.



Fixing variables
----------------

It is possible to fix all variables to specific values. Create a ``pandas.DataFrame`` or a column with the same name as the variable but with suffix '_set'. For all not ``NaN`` values additional constraints will be build to fix the variables.

For example, let's say we want to fix the output of a single generator 'gas1' to 200 MW for all snapshots. Then we can add a dataframe ``p_set`` to ``n.generators_t`` with the according value and index::

  n.generators_t['p_set'] = pd.DataFrame(200, index=n.snapshots, columns=['gas1'])

The optimization will now build extra constraints to fix the ``p`` variables of generator 'gas1' to 200. In the same manner, we can fix the variables only for some specific snapshots. This is applicable to all variables, also ``state_of_charge`` for storage units or ``p`` for links. Static investment variables can be fixed via adding additional columns, e.g. a ``s_nom_set`` column to ``n.lines``.



Inputs
------


For the linear optimal power flow, the following data for each component
are used. For almost all values, defaults are assumed if not
explicitly set. For the defaults and units, see :doc:`/user-guide/components`.

* n.{snapshot_weightings}

* n.buses.{v_nom, carrier}

* n.loads.{p_set}

* n.generators.{p_nom, p_nom_extendable, p_nom_min, p_nom_max, p_min_pu, p_max_pu, marginal_cost, capital_cost, efficiency, carrier}

* n.storage_units.{p_nom, p_nom_extendable, p_nom_min, p_nom_max, p_min_pu, p_max_pu, marginal_cost, capital_cost, efficiency*, standing_loss, inflow, state_of_charge_set, max_hours, state_of_charge_initial, cyclic_state_of_charge}

* n.stores.{e_nom, e_nom_extendable, e_nom_min, e_nom_max, e_min_pu, e_max_pu, e_cyclic, e_initial, capital_cost, marginal_cost, standing_loss}

* n.lines.{x, s_nom, s_nom_extendable, s_nom_min, s_nom_max, capital_cost}

* n.transformers.{x, s_nom, s_nom_extendable, s_nom_min, s_nom_max, capital_cost}

* n.links.{p_min_pu, p_max_pu, p_nom, p_nom_extendable, p_nom_min, p_nom_max, capital_cost}

* n.carriers.{carrier_attribute}

* n.global_constraints.{type, carrier_attribute, sense, constant}

Outputs
-------


* n.buses.{v_mag_pu, v_ang, p, marginal_price}

* n.loads.{p}

* n.generators.{p, p_nom_opt}

* n.storage_units.{p, p_nom_opt, state_of_charge, spill}

* n.stores.{p, e_nom_opt, e}

* n.lines.{p0, p1, s_nom_opt, mu_lower, mu_upper}

* n.transformers.{p0, p1, s_nom_opt, mu_lower, mu_upper}

* n.links.{p0, p1, p_nom_opt, mu_lower, mu_upper}

* n.  global_constraints.{mu}
