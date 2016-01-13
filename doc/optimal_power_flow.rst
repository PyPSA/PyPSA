######################
 Optimal Power Flow
######################


See pypsa.opf.

Linear Optimal Power Flow
=========================

network.lopf(snapshots,solver_name)

where snapshots is an iterable of snapshots and solver_name is a
string, e.g. "gurobi" or "glpk".

The linear OPF module can optimises the dispatch of generation and storage
and the capacities of generation, storage and transmission.

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
optimised, like an electricity market model.

For simplified transmission representation using Net Transfer
Capacities (NTCs), there is a TransportLink component which does
controllable power flow like a transport model (and can also represent
a point-to-point HVDC link).



Optimising total annual system costs
------------------------------------

To minimise annual system costs for meeting an inelastic electrical
load, capital costs for transmission and generation should be set to
the annualised investment costs in e.g. EUR/MW/a, marginal costs for
dispatch to e.g. EUR/MWh and the weightings chosen such that


.. math::
   \sum_t w_t = 8760

In this case the objective function gives total system cost in EUR/a
to meet the total load.
