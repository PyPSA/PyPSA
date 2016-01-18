#################
 Components
#################

The components and their attributes can also be read from the code in
pypsa.components.


Network
==========

Overall container. Can contain disconnected sub-networks.

Sub-Network
=============

Connected subset of network buses and branches.

Sub-Network are determined by calling:

network.determine_network_topology()



Bus
=======

Fundamental electrical node of system.


One-Ports: Generators, Storage Units, Loads, Shunt Impedances
============================================================

These components share the property that they all connect to a single
bus.

They have attributes:


+------------+------------+-----------+---------------------------------------+
| Name       | Type       | Unit      |Description                            |
+============+============+===========+=======================================+
|    bus     |   string   |           |name of bus                            |
|            |            |           |to which                               |
|            |            |           |one-port is                            |
|            |            |           |attached                               |
|            |            |           |                                       |
+------------+------------+-----------+---------------------------------------+
|  p         |series      |MW         | active power (as calculated by PyPSA) |
+------------+------------+-----------+---------------------------------------+
| q          |series      |MVar       |reactive power (as calculated by PyPSA |
+------------+------------+-----------+---------------------------------------+



Generator
---------

Can have generator.dispatch in ["variable","flexible"], which dictates
how they behave in the OPF.

"flexible" generators can dispatch
anywhere between gen.p_nom*(gen.p_nom_min_pu_fixed) and
gen.p_nom*(gen.p_nom_max_pu_fixed) at all times.

"variable" generators have time series gen.p_max_pu which dictates the
active power availability for each snapshot.




+------------+------------+-----------+---------------------------------------+
| Name       | Type       | Unit      |Description                            |
+============+============+===========+=======================================+
| dispatch   |   string   |must be    |Controllability of active power        |
|            |            |"flexible" |dispatch                               |
|            |            |or         |                                       |
|            |            |"variable" |                                       |
|            |            |           |                                       |
+------------+------------+-----------+---------------------------------------+
| control    |string      |must be    |        P,Q,V control strategy         |
|            |            |"PQ", "PV" |                                       |
|            |            |or "Slack" |                                       |
+------------+------------+-----------+---------------------------------------+
| p_set      |series      |MW         |active power set point (for PF)        |
|            |            |           |                                       |
+------------+------------+-----------+---------------------------------------+
| q_set      |series      |MVar       |reactive power set point (for PF)      |
|            |            |           |                                       |
+------------+------------+-----------+---------------------------------------+



Storage Unit
------------

Has a time-varying state of charge and various efficiencies.

Load
-----

PQ load.

Shunt Impedance
---------------

Has voltage-dependent admittance.


Branches: Lines, Transformers, Converters, Transport Links
===========================================================

Have bus0 and bus1 to which they attached.

Power flow at bus recorded in p0, p1, q0, q1.



Line
------

A transmission line connected line.bus0 to line.bus1. Can be DC or AC.


Transformer
------------

Converts from one AC voltage level to another.

Converter
----------

Converts AC to DC power.

Transport Link
--------------

Like a controllable point-to-point HVDC connector; equivalent to
converter-(DC line)-converter.

Source
======

For storing information about fuel sources, e.g. $CO_2$ emissions of gas or coal or wind.
