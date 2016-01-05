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

Generator
============

Can have  generator.dispatch in ["variable","flexible"]

Storage Unit
============

Has a time-varying state of charge and various efficiencies.

Load
======

PQ load.

Shunt Impedance
======

Has voltage-dependent admittance.



Line
=====

A transmission line connected line.bus0 to line.bus1. Can be DC or AC.


Transformer
==========

Converts from one AC voltage level to another.

Converter
==========

Converts AC to DC power.

Transport Link
==============

Like a controllable point-to-point HVDC connector; equivalent to
converter-(DC line)-converter.

Source
======

For storing information about fuel sources, e.g. $CO_2$ emissions of gas or coal or wind.
