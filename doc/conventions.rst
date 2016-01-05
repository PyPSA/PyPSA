################
 Conventions
################

Unit Conventions
=================

The units for physical quantities are chosen for easy user input.

The units follow the general rules:

Power: MW/MVA/MVar (unless per unit of nominal power,
e.g. generator.p_max_pu for variable generators is per unit of
generator.p_nom)

Voltage: kV phase-phase for bus.v_nom; per unit for v_mag, v_mag_set, v_mag_min etc.

Angles: radians, except transformer.phase_shift which is in degrees for easy input

Impedance: Ohm, except transformers which are pu, using transformer.s_nom for the base power


Sign Conventions
========


The sign convention in PyPSA follows other major software packages,
such as MATPOWER, PYPOWER and DIgSILENT PowerFactory.

* The power (p,q) of generators or storage units is positive if the asset is injecting
power into the bus, negative if withdrawing power from bus.
* The power (p,q) of loads is positive if withdrawing power from bus, negative if injecting power into bus.
* The power (p0,q0) at bus0 of a branch is positive if the branch is withdrawing power from
bus0, i.e. bus0 is injecting into branch
* Similarly the power (p1,q1) at bus1 of a branch  is positive if the branch is withdrawing power from bus1, negative if the branch is injecting into bus1
* If p0 > 0 and p1 < 0 for a branch then active power flows from bus0 to bus1; p0+p1 > 0 is the active power losses for this direction of power flow.
