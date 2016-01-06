######################
Power Flow
######################


See pypsa.pf.


Full non-linear power flow
=====================

Currently uses scipy.optimize.fsolve; should use Newton-Raphson instead.

If the series impedance is given by

.. math::
   z = r+jx

and the shunt admittance is given by

.. math::
   y = g + jb

then the currents and voltages at buses 0 and 1 are related by

.. math::
  \left( \begin{array}{c}
    i_0 \\ i_1
  \end{array}
  \right) =   \left( \begin{array}{cc} \left(\frac{1}{z} + \frac{y}{2} \right) \frac{1}{\tau^2} &      -\frac{1}{z}\frac{1}{\tau e^{-j\theta}}  \\
   -\frac{1}{z}\frac{1}{\tau e^{j\theta}} & \frac{1}{z} + \frac{y}{2}
   \end{array}
   \right)  \left( \begin{array}{c}
    v_0 \\ v_1
  \end{array}
    \right)

where :math:`\tau` is the tap ratio between the per unit voltages bus0:bus1 and :math:`\theta` is the phase shift in the transformer.


Linear power flow
=============

Assume decoupling of reactive power, no voltage magnitude variations,
angles are small.

For AC network, load flow calculated using small voltage angles and series reactance alone.

For DC network, load flow calculated using small voltage magnitude differences and resistance alone.
