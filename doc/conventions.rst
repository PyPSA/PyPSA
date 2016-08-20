################
 Conventions
################

.. _unit-conventions:

Unit Conventions
=================

The units for physical quantities are chosen for easy user input.

The units follow the general rules:

Power: MW/MVA/MVar (unless per unit of nominal power,
e.g. generator.p_max_pu for variable generators is per unit of
generator.p_nom)

Time: h

Energy: MWh

Voltage: kV phase-phase for bus.v_nom; per unit for v_mag_pu, v_mag_pu_set, v_mag_pu_min etc.

Angles: radians, except transformer.phase_shift which is in degrees for easy input

Impedance: Ohm, except transformers which are pu, using transformer.s_nom for the base power

CO2-equivalent emissions: tonnes of CO2-equivalent per MWh_thermal of energy carrier


Sign Conventions
================


The sign convention in PyPSA follows other major software packages,
such as MATPOWER, PYPOWER and DIgSILENT PowerFactory.

* The power (p,q) of generators or storage units is positive if the
  asset is injecting power into the bus, negative if withdrawing power
  from bus.
* The power (p,q) of loads is positive if withdrawing power from bus, negative if injecting power into bus.
* The power (p0,q0) at bus0 of a branch is positive if the branch is
  withdrawing power from bus0, i.e. bus0 is injecting into branch
* Similarly the power (p1,q1) at bus1 of a branch is positive if the
  branch is withdrawing power from bus1, negative if the branch is
  injecting into bus1
* If p0 > 0 and p1 < 0 for a branch then active power flows from bus0
  to bus1; p0+p1 > 0 is the active power losses for this direction of
  power flow.

AC/DC Terminology
=================

AC stands for Alternating Current and DC stands for Direct Current.

Some people refer to the linearised power flow equations for AC
networks as "DC load flow" for historical reasons, but we find this
confusing when there are actual direct current elements in the network
(which also have a linearised power flow, which would then be DC DC load
flow).

Therefore for us AC means AC and DC means DC. We distinguish between
the full non-linear network equations (with no approximations) and the
linearised network equations (with certain approximations to make the
equations linear).

All equations are listed in the section :doc:`power_flow`.
