######################
Power Flow
######################


See the module ``pypsa.pf``.


Full non-linear power flow
==========================


The non-linear power flow ``network.pf()`` works for AC networks and
by extension for DC networks too (with a work-around described below).

Non-linear power flow for AC networks
-------------------------------------

The power flow ensures for given inputs (load and power plant
dispatch) that the following equation is satisfied for each bus
:math:`i`:

.. math::
   S_i = P_i + j Q_i = V_i I_i^* = V_i \left(\sum_j Y_{ij} V_j\right)^*

where :math:`V_i = |V_i|e^{j\theta_i}` is the complex voltage, whose
rotating angle is taken relative to the slack bus.

:math:`Y_{ij}` is the bus admittance matrix, based on the branch
impedances and any shunt admittances attached to the buses.



For the slack bus it is assumed :math:`V_0 = 1`; P and Q are to be found.

For the PV buses, P and :math:`|V|` are given; Q and :math:`\theta` are to be found.

For the PQ buses, P and Q are given; :math:`|V|` and :math:`\theta` are to be found.

If PV and PQ are the sets of buses, then there are :math:`|PV| + 2|PQ|` real equations to solve:


.. math::
   \textrm{Re}\left[ V_i \left(\sum_j Y_{ij} V_j\right)^* \right] - P_i & = 0 \hspace{.7cm}\forall\hspace{.1cm} i \in PV \cup PQ \\
   \textrm{Im}\left[ V_i \left(\sum_j Y_{ij} V_j\right)^* \right] - Q_i & = 0 \hspace{.7cm}\forall\hspace{.1cm} i \in PQ

To be found: :math:`\theta_i \forall i \in PV \cup PQ` and :math:`|V_i| \forall i PQ`.

These equations :math:`f(x) = 0` are solved using the `Newton-Raphson method <https://en.wikipedia.org/wiki/Newton%27s_method#k_variables.2C_k_functions>`_, with the Jacobian:


.. math::
   \frac{\partial f}{\partial x} = \left( \begin{array}{cc}
                                 \frac{\partial P}{\partial \theta} & \frac{\partial P}{\partial |V|} \\
				 \frac{\partial Q}{\partial \theta} & \frac{\partial Q}{\partial |V|}
				 \end{array} \right)

and the initial "flat" guess of :math:`\theta_i = 0` and :math:`|V_i| = 1` for unknown quantities.


Branch model
------------

The branches are modelled with the standard PI model.

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




Non-linear power flow for DC networks
-------------------------------------

For meshed DC networks the equations are a special case of those for
AC networks, with the difference that all quantities are real.

To solve the non-linear equations for a DC network, ensure that the
series reactance :math:`x` and shunt susceptance :math:`b` are zero
for all branches, pick a Slack bus (where :math:`V_0 = 1` and set all
other buses to be 'PQ' buses. Then execute ``network.pf()``.

The voltage magnitudes then satisfy at each bus :math:`i`:

.. math::
   P_i  = V_i I_i = V_i \sum_j G_{ij} V_j

where all quantities are real.

:math:`G_{ij}` is based only on the branch resistances and any shunt
conductances attached to the buses.


Linear power flow
=================

For AC networks, it is assumed for the linear power flow that reactive
power decouples, there are no voltage magnitude variations, voltage
angles differences across branches are small and branch resistances
are much smaller than branch reactances (i.e. it is good for overhead
transmission lines).

For AC networks, the load flow is calculated using small voltage
angles and the series reactances alone.

For DC networks, it is assumed for the linear power flow that voltage
magnitude variations are all small.

For DC networks, the load flow is calculated using small voltage
magnitude differences and series resistances alone.
