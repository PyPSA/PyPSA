###########
 Design
###########


Python 2 and Python 3 compatible
================================

PyPSA is written and tested to be compatible with both Python 2.7 and
Python 3.4.



Network object is the overall container
=======================================

The ``pypsa.components.Network`` is an overall container for all
network components; components cannot exist without a network.

It is also the object on which calculations, such as power flow and
optimal power flow, are performed.


Buses are the fundamental nodes
===============================

The ``pypsa.components.Bus`` is the fundamental node to which all
loads, generators, storage units, lines, transformers, converters and
links attach.

You can have as many components attached to a bus as you want.


Energy flow in the model
========================

Energy enters the model in generators, storage units or stores with
higher energy before than after the simulation, and any components
with efficiency greater than 1 (e.g. heat pumps).

Energy leaves the model in loads, storage units or stores with higher
energy after than before the simulation, and in lines, links or
storage units with efficiency less than 1.



Data storage uses pandas DataFrames and Panels
==============================================

To enable efficient calculations on the different dimensions of the
data, data is stored in memory using pandas DataFrames and Panels
(three-dimensional DataFrames).

Other power system toolboxes use databases for data storage; given
modern RAM availability and speed considerations, pandas DataFrames
were felt to be preferable and simpler.


To see which data is stored for each component, see :doc:`components`.


Static component data is stored in pandas DataFrames
====================================================

For each component type (line, transformer, generator, etc.), which
must be uniquely named for each network, its basic static data is
stored in a pandas DataFrame, which is an attribute of the network
object, e.g.

* network.lines
* network.transformers
* network.generators

These are all pandas DataFrames, indexed by the unique name of the
component.

The columns contain data such as impedance, capacity, etc.

Network components cannot exist without a network to hold them.



Time-varying data are stored in pandas Panels
=================================================

Some quantities, such as generator.p_set (generator active power set
point), generator.p (generator calculated active power), line.p0 (line
active power at bus0) and line.p1 (line active power at bus1) vary
over time and therefore are stored as pandas Series. They are stored
together in a three-dimensional pandas Panel, indexed by the attribute
("p_set" or "p"), the component names and the network's time steps in
``network.snapshots``.

They all have names like ``network.generators_t`` and the atttributes
are accessed like:

* network.generators_t.p_set
* network.generators_t.p
* network.lines_t.p0
* network.lines_t.p1



Object model with descriptor properties point to DataFrames
===========================================================

Sometimes it is useful to access the components as objects instead of
using the pandas DataFrames and Panels.

For this each component DataFrame has a column "obj" containing
objects, which have the various component data as attributes, e.g.

bus.v_nom

is a descriptor which points at network.buses.loc["bus_name","v_nom"].


No GUI: Use Jupyter notebooks
=============================

PyPSA has no Graphical User Interface (GUI). However it has features
for plotting time series and networks (e.g. ``network.plot()``), which
works especially well in combination with `Jupyter notebooks
<http://jupyter.org/>`_.

Internal use of per unit
===========================

Per unit values of voltage and impedance are used internally for
network calculations. It is assumed internally that the base power is
1 MVA. The base voltage depends on the component.

See also :ref:`unit-conventions`.


Set points are stored separately from actual dispatch points
============================================================

Dispatchable generators have a p_set series which is separate from the
calculated active power series p, since the operators's intention may
be different from what is calculated (e.g. when using distributed
slack for the active power).


Pyomo for the optimisation framework
====================================

To enable portability between solvers, the OPF is formulated using the
Python optimisation modelling package `pyomo <http://www.pyomo.org/>`_
(which can be thought of as a Python version of `GAMS
<http://www.gams.de/>`_).

Pyomo also has useful features such as index sets, etc.
