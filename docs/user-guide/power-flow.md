######################
Power Flow
######################



Non-linear power flow
==========================


The non-linear power flow ``n.pf()`` works for AC networks and
by extension for DC networks too.

The non-linear power flow ``n.pf()`` can be called for a
particular ``snapshot`` as ``n.pf(snapshot)`` or on an iterable
of ``snapshots`` as ``n.pf(snapshots)`` to calculate the
non-linear power flow on a selection of snapshots at once (which is
more performant than calling ``n.pf`` on each snapshot
separately). If no argument is passed, it will be called on all
``n.snapshots``, see :py:meth:`pypsa.Network.pf` for details.




AC networks (single slack)
--------------------------

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

AC networks (distributed slack)
-------------------------------

If the slack is to be distributed to all generators in proportion
to their dispatch (``distribute_slack=True``), instead of being
allocated fully to the slack bus, the active power balance is altered to

.. math::
   \textrm{Re}\left[ V_i \left(\sum_j Y_{ij} V_j\right)^* \right] - P_i - P_{slack}\gamma_i = 0 \hspace{.7cm}\forall\hspace{.1cm} i \in PV \cup PQ \cup slack

where :math:`P_{slack}` is the total slack power and :math:`\gamma_{i}`
is the share of bus :math:`i` of the total generation that is used to
distribute the slack power. Note that also an additional active power
balance is included for the slack bus since it is now part of the
distribution scheme.

This adds an additional **row** to the Jacobian for the derivatives
of the slack bus active power balance and an additional **column**
for the partial derivatives with respect to :math:`\gamma_i`.

DC networks
-----------

For meshed DC networks the equations are a special case of those for
AC networks, with the difference that all quantities are real.

To solve the non-linear equations for a DC network, ensure that the
series reactance :math:`x` and shunt susceptance :math:`b` are zero
for all branches, pick a Slack bus (where :math:`V_0 = 1`) and set all
other buses to be 'PQ' buses. Then execute ``n.pf()``.

The voltage magnitudes then satisfy at each bus :math:`i`:

.. math::
   P_i  = V_i I_i = V_i \sum_j G_{ij} V_j

where all quantities are real.

:math:`G_{ij}` is based only on the branch resistances and any shunt
conductances attached to the buses.

.. _line-model:

Line model
----------

Lines are modelled with the standard equivalent PI model.



If the series impedance is given by

.. math::
   z = r+jx

and the shunt admittance is given by

.. math::
   y = g + jb

then the currents and voltages at buses 0 and 1 for a line:


.. image:: ../img/line-equivalent.png

are related by

.. math::
  \left( \begin{array}{c}
    i_0 \\ i_1
  \end{array}
  \right) =   \left( \begin{array}{cc} \frac{1}{z} + \frac{y}{2} &      -\frac{1}{z}  \\
   -\frac{1}{z} & \frac{1}{z} + \frac{y}{2}
   \end{array}
   \right)  \left( \begin{array}{c}
    v_0 \\ v_1
  \end{array}
    \right)


.. _transformer-model:

Transformer model
-----------------

The transformer models here are largely based on the implementation in
`pandapower <https://github.com/panda-power/pandapower>`__, which is
loosely based on `DIgSILENT PowerFactory
<http://www.digsilent.de/index.php/products-powerfactory.html>`_.

Transformers are modelled either with the equivalent T model (the
default, since this represents the physics better) or with the
equivalent PI model. The can be controlled by setting transformer
attribute ``model`` to either ``t`` or ``pi``.

The tap changer can either be modelled on the primary, high voltage
side 0 (the default) or on the secondary, low voltage side 1. This is set with attribute ``tap_side``.

If the transformer ``type`` is not given, then ``tap_ratio`` is
defined by the user, defaulting to ``1.``. If the ``type`` is given,
then the user can specify the ``tap_position`` which results in a
``tap ratio`` :math:`\tau` given by:

.. math::
  \tau = 1 + (\textrm{tap_position} - \textrm{tap_neutral})\cdot \frac{\textrm{tap_step}}{100}


For a transformer with tap ratio :math:`\tau` on the primary side
``tap_side = 0`` and phase shift :math:`\theta_{\textrm{shift}}`, the
equivalent T model is given by:


.. image:: ../img/transformer-t-equivalent-tap-hv.png

For a transformer with tap ratio :math:`\tau` on the secondary side
``tap_side = 1`` and phase shift :math:`\theta_{\textrm{shift}}`, the
equivalent T model is given by:


.. image:: ../img/transformer-t-equivalent-tap-lv.png



For the admittance matrix, the T model is transformed into a PI model
with the wye-delta transformation.

For a transformer with tap ratio :math:`\tau` on the primary side
``tap_side = 0`` and phase shift :math:`\theta_{\textrm{shift}}`, the
equivalent PI model is given by:


.. image:: ../img/transformer-pi-equivalent-tap-hv.png

for which the currents and voltages are related by:

.. math::
  \left( \begin{array}{c}
    i_0 \\ i_1
  \end{array}
  \right) =   \left( \begin{array}{cc}  \frac{1}{z} + \frac{y}{2} &      -\frac{1}{z}\frac{1}{\tau e^{-j\theta}}  \\
   -\frac{1}{z}\frac{1}{\tau e^{j\theta}} & \left(\frac{1}{z} + \frac{y}{2} \right) \frac{1}{\tau^2}
   \end{array}
   \right)  \left( \begin{array}{c}
    v_0 \\ v_1
  \end{array}
    \right)




For a transformer with tap ratio :math:`\tau` on the secondary side
``tap_side = 1`` and phase shift :math:`\theta_{\textrm{shift}}`, the
equivalent PI model is given by:


.. image:: ../img/transformer-pi-equivalent-tap-lv.png

for which the currents and voltages are related by:

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


Inputs
------

For the non-linear power flow, the following data for each component
are used. For almost all values, defaults are assumed if not
explicitly set. For the defaults and units, see :doc:`/user-guide/components`.

- ``n.buses.{v_nom, v_mag_pu_set}``
- ``n.loads.{p_set, q_set}``
- ``n.generators.{control, p_set, q_set}``
- ``n.storage_units.{control, p_set, q_set}``
- ``n.stores.{p_set, q_set}``
- ``n.shunt_impedances.{b, g}``
- ``n.lines.{x, r, b, g}``
- ``n.transformers.{x, r, b, g}``
- ``n.links.{p_set}``

.. note:: Note that the control strategy for active and reactive power PQ/PV/Slack is set on the generators not on the buses. Buses then inherit the  control strategy from the generators attached at the bus (defaulting to PQ if there is no generator attached). Any PV generator will make the whole bus a PV bus. For PV buses, the voltage magnitude set point is set on the bus, not the generator, with bus.v_mag_pu_set since it is a bus property.


.. note:: Note that for lines and transformers you MUST make sure that :math:`r+jx` is non-zero, otherwise the bus admittance matrix will be singular.

Outputs
-------

- ``n.buses.{v_mag_pu, v_ang, p, q}``
- ``n.loads.{p, q}``
- ``n.generators.{p, q}``
- ``n.storage_units.{p, q}``
- ``n.stores.{p, q}``
- ``n.shunt_impedances.{p, q}``
- ``n.lines.{p0, q0, p1, q1}``
- ``n.transformers.{p0, q0, p1, q1}``
- ``n.links.{p0, p1}``


Linear power flow
=================

The linear power flow ``n.lpf()`` can be called for a
particular ``snapshot`` as ``n.lpf(snapshot)`` or on an iterable
of ``snapshots`` as ``n.lpf(snapshots)`` to calculate the
linear power flow on a selection of snapshots at once (which is
more performant than calling ``n.lpf`` on each snapshot
separately). If no argument is passed, it will be called on all
``n.snapshots``, , see :py:meth:`pypsa.Network.lpf` for details.

AC networks
-----------

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
   P_i = \sum_j (KBK^T)_{ij} \theta_j - \sum_l K_{il} b_l \theta_l^{\textrm{shift}}

where :math:`K` is the incidence matrix of the network, :math:`B` is
the diagonal matrix of inverse branch series reactances :math:`x_l`
multiplied by the tap ratio :math:`\tau_l`, i.e. :math:`B_{ll} = b_l =
\frac{1}{x_l\tau_l}` and :math:`\theta_l^{\textrm{shift}}` is the
phase shift for a transformer. The matrix :math:`KBK^T` is singular
with a single zero eigenvalue for a connected network, therefore the
row and column corresponding to the slack bus is deleted before
inverting.

The flows ``p0`` in the network branches at ``bus0`` can then be found by multiplying by the transpose incidence matrix and inverse series reactances:

.. math::
   F_l = \sum_i (BK^T)_{li} \theta_i - b_l \theta_l^{\textrm{shift}}

DC networks
-----------

For DC networks, it is assumed for the linear power flow that voltage
magnitude differences across branches are all small.

For DC networks, the linear load flow is calculated using small voltage
magnitude differences and series resistances alone.

The linear load flow for DC networks follows the same calculation as for AC networks, but replacing the voltage angles by the difference in voltage magnitude :math:`\delta V_{n,t}` and the series reactance by the series resistance :math:`r_l`.


Inputs
------

For the linear power flow, the following data for each component
are used. For almost all values, defaults are assumed if not
explicitly set. For the defaults and units, see :doc:`/user-guide/components`.

- ``n.buses.{v_nom}``
- ``n.loads.{p_set}``
- ``n.generators.{p_set}``
- ``n.storage_units.{p_set}``
- ``n.stores.{p_set}``
- ``n.shunt_impedances.{g}``
- ``n.lines.{x}``
- ``n.transformers.{x}``
- ``n.links.{p_set}``

.. note:: Note that for lines and transformers you must make sure that :math:`x` is non-zero, otherwise the bus admittance matrix will be singular.

Outputs
-------

- ``n.buses.{v_mag_pu, v_ang, p}``
- ``n.loads.{p}``
- ``n.generators.{p}``
- ``n.storage_units.{p}``
- ``n.stores.{p}``
- ``n.shunt_impedances.{p}``
- ``n.lines.{p0, p1}``
- ``n.transformers.{p0, p1}``
- ``n.links.{p0, p1}``
