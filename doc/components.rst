#################
 Components
#################


Power system components can be found in ``pypsa.components.py``.

The attributes of each component can be accessed either from the
objects, e.g. ``bus.v_nom``, or from the pandas DataFrame of all
components, e.g. ``network.buses.v_nom`` for static attributes or
``network.buses_t.p_set`` for time-dependent series attributes.

All attributes are listed below for each component.

Their status is either "Input" for those which the user specifies or
"Output" for those results which PyPSA calculates.

The inputs can be either "required", if the user *must* give the
input, or "optional", if PyPSA will use a sensible default if the user
gives no input.


The components and their attributes can also be read from the code in
``pypsa.components.py``.


Network
==========

The ``Network`` is the overall container for all components. It also
has the major functions as methods, such as ``network.lopf()`` and
``network.pf()``.

.. csv-table::
   :header-rows: 1
   :file: network.csv



Sub-Network
=============

Sub-networks are determined by PyPSA and should not be entered by the
user.

Sub-networks are subsets of buses and passive branches (i.e. lines and
transformers) that are connected.

They are either "DC" or "AC". In the case of "AC" sub-networks, these
correspond to synchronous areas.

The power flow in sub-networks is determined by the passive flow
through passive branches due to the impedances of the passive branches.

Sub-Network are determined by calling
``network.determine_network_topology()``.


.. csv-table::
   :header-rows: 1
   :file: sub_network.csv


Bus
=======

Fundamental electrical node of system.



.. csv-table::
   :header-rows: 1
   :file: buses.csv


One-Ports: Generators, Storage Units, Loads, Shunt Impedances
============================================================

These components share the property that they all connect to a single
bus.

They have attributes:


.. csv-table::
   :header-rows: 1
   :file: one_ports.csv




Generator
---------

Can have generator.dispatch in ["variable","flexible"], which dictates
how they behave in the OPF.

"flexible" generators can dispatch
anywhere between gen.p_nom*(gen.p_nom_min_pu_fixed) and
gen.p_nom*(gen.p_nom_max_pu_fixed) at all times.

"variable" generators have time series gen.p_max_pu which dictates the
active power availability for each snapshot.



.. csv-table::
   :header-rows: 1
   :file: generators.csv



Storage Unit
------------

Has a time-varying state of charge and various efficiencies.


.. csv-table::
   :header-rows: 1
   :file: storage_units.csv


Load
-----

PQ load.


.. csv-table::
   :header-rows: 1
   :file: loads.csv


Shunt Impedance
---------------

Has voltage-dependent admittance.

.. csv-table::
   :header-rows: 1
   :file: shunt_impedances.csv


Branches: Lines, Transformers, Converters, Transport Links
===========================================================

Have bus0 and bus1 to which they attached.

Power flow at bus recorded in p0, p1, q0, q1.



.. csv-table::
   :header-rows: 1
   :file: branches.csv


Line
------

A transmission line connected line.bus0 to line.bus1. Can be DC or AC.


.. csv-table::
   :header-rows: 1
   :file: lines.csv


Transformer
------------

Converts from one AC voltage level to another.


.. csv-table::
   :header-rows: 1
   :file: transformers.csv


Converter
----------

Converts AC to DC power.


.. csv-table::
   :header-rows: 1
   :file: converters.csv


Transport Link
--------------

Like a controllable point-to-point HVDC connector; equivalent to
converter-(DC line)-converter.


.. csv-table::
   :header-rows: 1
   :file: transport_links.csv


Source
======

For storing information about fuel sources, e.g. $CO_2$ emissions of gas or coal or wind.


.. csv-table::
   :header-rows: 1
   :file: sources.csv
