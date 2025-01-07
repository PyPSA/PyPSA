#######
 Design
#######


Networks
========

The ``pypsa.Network`` is an overall container for all network components.
Components cannot exist without a network.

It also holds functions to run compute power flows, optimal power flow and
capacity expansion planning problems, as well as functions to retrieve
statistics and plot the network.

Buses
=====

The bus is the fundamental node to which all loads, generators, storage units,
lines, transformers and links attach. You can have as many components attached
to a bus as you want. The bus's role is to enforce energy conservation for all
elements feeding in and out of it (i.e. like Kirchhoff's Current Law).


.. image:: ../img/buses.png


Energy flows
============

Energy enters the model in generators, storage units or stores with
higher energy before than after the simulation, and any components
with efficiency greater than 1 (e.g. heat pumps).

Energy leaves the model in loads, storage units or stores with higher
energy after than before the simulation, and in lines, links or
storage units with efficiency less than 1.



Data storage
============

To enable efficient calculations on the different dimensions of the
data, data is stored in memory using ``pandas.DataFrame`` objects.

To see which data is stored for each component, see :doc:`/user-guide/components`.


Static data
-----------

For each component type (line, transformer, generator, etc.), which must be
uniquely named for each network, its basic static data is stored in a
``pandas.DataFrame``, which is an attribute of the ``pypsa.Network``, with names
that follow the component names:

* ``n.buses``
* ``n.generators``
* ``n.loads``
* ``n.lines``
* ``n.links``
* ``n.storage_units``
* ``n.stores``
* ``n.transformers``

The columns contain data such as impedance, capacity, costs, efficiencies and
the buses to which components are attached. All attributes for each component
type are listed with their properties (defaults, etc.) in
:doc:`/user-guide/components` and are accessible from the network object, e.g.
in ``n.components["Bus"]["attrs"]``.

.. _time-varying:

Time-varying data
-----------------

Some quantities, such as generator ``p_max_pu`` (generator availability),
generator ``p`` (generator calculated active power), line ``p0`` (line active
power at ``bus0``) and line ``p1`` (line active power at ``bus1``) may vary over
time, so different values of these attributes for the different snapshots
(``n.snapshots``) are stored in the following attributes of the network object:

* ``n.buses_t``
* ``n.generators_t``
* ``n.loads_t``
* ``n.lines_t``
* ``n.links_t``
* ``n.storage_units_t``
* ``n.stores_t``
* ``n.transformers_t``

These are dictionaries of ``pandas.DataFrame`` objects, so that for example
``n.generators_t["p_set"]`` is a ``pandas.DataFrame`` with columns
corresponding to generator names and index corresponding to
``n.snapshots``. You can also access the dictionary like an
attribute ``n.generators_t.p_set``.

Time-varying data are marked as ``series`` in the listings in  :doc:`/user-guide/components`.


For **input data** such as ``p_max_pu`` of a generator you can store the
value statically in ``n.generators`` if the value does not
change over ``n.snapshots`` **or** you can define it to be
time-varying by adding a column to ``n.generators_t.p_max_pu``. If
the name of the generator is in the columns of
``n.generators_t.p_max_pu``, then the static value in
``n.generators`` will be ignored. Some example definitions of
input data:


.. code:: python

    import pypsa

    n = pypsa.Network()

    #four snapshots are defined by integers
    n.set_snapshots(range(4))

    n.add("Bus", "my bus")

    #add a generator whose output does not change over time
    n.add("Generator", "Coal", bus="my bus", p_set=100)

    #add a generator whose output does change over time
    n.add("Generator", "Wind", bus="my bus", p_set=[10,50,20,30])

In this case only the generator "Wind" will appear in the columns of
``n.generators_t.p_set``.

For **output data**, all time-varying data affecting generators is stored in the
``n.generators_t`` dictionaries, but it is only defined once a computation has
been run.

.. _unit-conventions:

Unit Conventions
=================

The units for physical quantities are chosen for easy user input.

The units follow the general rules:

.. list-table:: Title
   :widths: 25 75
   :header-rows: 1

   * - Quantity
     - Units
   * - Power
     - MW/MVA/MVar (unless per unit of nominal power, e.g. generator.p_max_pu
       for variable generators is per unit of generator.p_nom)
   * - Time
     - h
   * - Energy
     - MWh
   * - Voltage
     - kV phase-phase for bus.v_nom; per unit for v_mag_pu, v_mag_pu_set, v_mag_pu_min etc.
   * - Angles
     - radians, except transformer.phase_shift which is in degrees for easy input
   * - Impedance
     - Ohm, except transformers which are pu, using transformer.s_nom for the base power
   * - CO2-equivalent emissions
     - tonnes of CO2-equivalent per MWh_thermal of energy carrier

**Per unit values** of voltage and impedance are used internally for
network calculations. It is assumed internally that the base power is
1 MVA. The base voltage depends on the component.

.. _sign-conventions:

Sign Conventions
================


The sign convention in PyPSA follows other major software packages,
such as MATPOWER, PYPOWER and DIgSILENT PowerFactory.

* The power (p,q) of generators or storage units is positive if the
  asset is injecting power into the bus, negative if withdrawing power
  from bus.
* The power (p,q) of loads is positive if withdrawing power from bus, negative if injecting power into bus.
* The power (p0,q0) at bus0 of a branch (line, link, or transformer) is positive if the branch is
  withdrawing power from bus0, i.e. bus0 is injecting into branch
* Similarly the power (p1,q1) at bus1 of a branch is positive if the
  branch is withdrawing power from bus1, negative if the branch is
  injecting into bus1
* If p0 > 0 and p1 < 0 for a branch then active power flows from bus0
  to bus1; p0+p1 > 0 is the active power losses for this direction of
  power flow.


Input and output data
=====================

Input and output data is strictly separated in PyPSA, such that inputs are not
overwritten by outputs from computations. Therefore, set points are stored
separately from actual dispatch points.

For instance, dispatchable generators have a ``p_set`` series which is separate
from the calculated active power series ``p``, since the operators's intention
may be different from what is calculated (e.g. when using distributed slack for
the active power).
