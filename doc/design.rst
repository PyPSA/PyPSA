###########
 Design
###########


Network object is the overall container
=======================================

The pypsa.components.Network is an overall container for all network
components.

It is also the object on which calculations, such as power flow and
optimal power flow, are performed.

Data storage uses pandas DataFrames
============

To enable efficient calculations on the different dimensions of the
data, data is stored in memory using pandas DataFrames.

Other power system toolboxes use databases for data storage; given
modern RAM availability and speed considerations, pandas DataFrames
were felt to be preferable and simpler.


To see which data is stored for each component, look in the code at
the appropriate class definition in pypsa.components.


Basic component data is stored in pandas DataFrames
================================

For each component type (line, transformer, generator, etc.), which
must be uniquely named for each network, its basic data is stored in a
pandas DataFrame, which is an attribute of the network object, e.g.

* network.lines
* network.transformers
* network.generators

are all pandas DataFrames, indexed by the unique name of the component.

The columns contain data such as impedance, capacity, etc.


Network components cannot exist without a network to hold them.



Time-varying data are stored in pandas DataFrames
=================================================

Some quantities, such as generator.p_set (generator active power set
point), generator.p (generator actual active power), line.p0 (line
active power at bus0) and line.p1 (line active power at bus1) vary
over time and therefore are stored as pandas Series. They are stored
together per component in DataFrames which are accessed as attributes
of the component DataFrame, e.g.

* network.generators.p_set
* network.generators.p
* network.lines.p0
* network.lines.p1

These DataFrames are index by the network's snapshot list
network.snapshots (set using network.set_snapshots(snapshots)). The
columns are the component names.



Object model with descriptor properties point to DataFrames
===========================================================

Sometimes it is useful to access the components as objects instead of
using the pandas DataFrames.

For this each component DataFrame has a column "obj" containing
objects, which have the various component data as attributes, e.g.

bus.v_nom

is a descriptor which points at network.buses.loc["bus_name","v_nom"].


Internal use of per unit
===============

Per unit values of voltage and impedance are used interally for
network calculations. It is assumed internally that the base power is
1 MVA. The base voltage depends on the component.

See also :ref:`unit-conventions`.


Set points are stored separately from actual dispatch points
==================================

Dispatchable generators have a p_set series which is separate from the
calculated active power series p, since the operators's intention may
be different from what is calculated (e.g. when using distributed
slack for the active power).


Pyomo for the optimisation framework
=================

To enable portability between solvers, the OPF is formulated using the
Python package `pyomo <http://www.pyomo.org/>`_ (which can be thought
of as a Python version of `GAMS <http://www.gams.de/>`_.

Pyomo also has useful features such as index sets, etc.
