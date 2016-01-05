######################
Power Flow
######################


See pypsa.pf.


Full non-linear power flow
=====================

Currently uses scipy.optimize.fsolve; should use Newton-Raphson instead.

Linear power flow
=============

Assume decoupling of reactive power, no voltage magnitude variations,
angles are small.

For AC network, load flow calculated using small voltage angles and series reactance alone.

For DC network, load flow calculated using small voltage magnitude differences and resistance alone.
