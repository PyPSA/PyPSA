######################
Power Flow
######################


See the module ``pypsa.pf``.


Full non-linear power flow
==========================


The non-linear power flow ``network.pf()`` works for AC networks and
by extension for DC networks too (with a work-around described below).

It can be called for a particular ``snapshot`` as
``network.pf(snapshot)``, otherwise ``network.pf()`` will default to the snapshot
``network.now``.



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



For the slack bus :math:`i=0` it is assumed :math:`|V_0|` is given and that :math:`\theta_0 = 0`; P and Q are to be found.

For the PV buses, P and :math:`|V|` are given; Q and :math:`\theta` are to be found.

For the PQ buses, P and Q are given; :math:`|V|` and :math:`\theta` are to be found.

If PV and PQ are the sets of buses, then there are :math:`|PV| + 2|PQ|` real equations to solve:


.. math::
   \textrm{Re}\left[ V_i \left(\sum_j Y_{ij} V_j\right)^* \right] - P_i & = 0 \hspace{.7cm}\forall\hspace{.1cm} i \in PV \cup PQ \\
   \textrm{Im}\left[ V_i \left(\sum_j Y_{ij} V_j\right)^* \right] - Q_i & = 0 \hspace{.7cm}\forall\hspace{.1cm} i \in PQ

To be found: :math:`\theta_i \forall i \in PV \cup PQ` and :math:`|V_i| \forall i \in PQ`.

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
for all branches, pick a Slack bus (where :math:`V_0 = 1`) and set all
other buses to be 'PQ' buses. Then execute ``network.pf()``.

The voltage magnitudes then satisfy at each bus :math:`i`:

.. math::
   P_i  = V_i I_i = V_i \sum_j G_{ij} V_j

where all quantities are real.

:math:`G_{ij}` is based only on the branch resistances and any shunt
conductances attached to the buses.

Inputs
------

For the non-linear power flow, the following data for each component
are used. For almost all values, defaults are assumed if not
explicitly set. For the defaults and units, see :doc:`components`.

bus.{v_nom, v_mag_pu_set (if PV generators are attached)}

load.{p_set, q_set}

generator.{control, p_set, q_set (for control PQ)}

storage_unit.{control, p_set, q_set (for control PQ)}

shunt_impedance.{b, g}

line.{x, r, b, g}

transformer.{x, r, b, g}

converter.{p_set}

transport_link.{p_set}



Note that the control strategy for active and reactive power
PQ/PV/Slack is set on the generators NOT on the buses. Buses then
inherit the control strategy from the generators attached at the bus
(defaulting to PQ if there is no generator attached). Any PV generator
will make the whole bus a PV bus. For PV buses, the voltage magnitude
set point is set on the bus, not the generator, with bus.v_mag_pu_set
since it is a bus property.


Note that for lines and transformers you MUST make sure that
:math:`r+jx` is non-zero, otherwise the bus admittance matrix will be
singular.

Outputs
-------

bus.{v_mag_pu, v_ang, p, q}

load.{p, q}

generator.{p, q}

storage_unit.{p, q}

shunt_impedance.{p, q}

line.{p0, q0, p1, q1}

transformer.{p0, q0, p1, q1}

converter.{p0, q0, p1, q1}

transport_link.{p0, q0, p1, q1}


Linear power flow
=================

The linear power flow ``network.lpf()`` can be called for a particular
``snapshot`` as ``network.lpf(snapshot)``, otherwise ``network.lpf()``
will default to ``network.now``. It can also be called
``network.lpf(snapshots)`` on an iterable of ``snapshots``
to calculate the linear power flow on a selection of snapshots at once
(which is more performant than calling ``network.lpf`` on each
snapshot separately).


For AC networks, it is assumed for the linear power flow that reactive
power decouples, there are no voltage magnitude variations, voltage
angles differences across branches are small and branch resistances
are much smaller than branch reactances (i.e. it is good for overhead
transmission lines).

For AC networks, the linear load flow is calculated using small voltage
angle differences and the series reactances alone.

It is assumed that the active powers :math:`P_i` are given for all buses except the slack bus and the task is to find the voltage angles :math:`\theta_i` at all buses except the slack bus, where it is assumed :math:`\theta_0 = 0`.

To find the voltage angles, the following linear set of equations are solved

.. math::
   P_i = \sum_j (KBK^T)_{ij} \theta_j

where :math:`K` is the incidence matrix of the network and :math:`B`
is the diagonal matrix of inverse line series reactances. The matrix
:math:`KBK^T` is singular with a single zero eigenvalue for a
connected network, therefore the row and column corresponding to the
slack bus is deleted before inverting.

The flow in the network can then be found by multiplying by the transpose incidence matrix and inverse series reactances:

.. math::
   F_l = \sum_i (BK^T)_{li} \theta_i



For DC networks, it is assumed for the linear power flow that voltage
magnitude differences across branches are all small.

For DC networks, the linear load flow is calculated using small voltage
magnitude differences and series resistances alone.

The linear load flow for DC networks follows the same calculation as for AC networks, but replacing the voltage angles by the difference in voltage magnitude :math:`\delta V_{n,t}` and the series reactance by the series resistance :math:`r_l`.


Inputs
------

For the linear power flow, the following data for each component
are used. For almost all values, defaults are assumed if not
explicitly set. For the defaults and units, see :doc:`components`.

bus.{v_nom}

load.{p_set}

generator.{p_set}

storage_unit.{p_set}

shunt_impedance.{g}

line.{x}

transformer.{x}

converter.{p_set}

transport_link.{p_set}

Note that for lines and transformers you MUST make sure that
:math:`x` is non-zero, otherwise the bus admittance matrix will be singular.

Outputs
-------

bus.{v_mag_pu, v_ang, p}

load.{p}

generator.{p}

storage_unit.{p}

shunt_impedance.{p}

line.{p0, p1}

transformer.{p0, p1}

converter.{p0, p1}

transport_link.{p0, p1}
