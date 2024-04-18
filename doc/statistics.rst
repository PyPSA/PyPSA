#######
 Statistics Module
#######


Overview
=======================================

The ``pypsa.Network.statistics`` module is an accessor of the network object which allows the user to quickly calculate metrics of the network. This comes in handy if someone does not know where all the information of the network are stored or wants to skip the process of doing cumbersome calculation by themself. The module is designed to be user friendly and easy to use.

The current available metrics one can calculate are:

* ``Installed Capacity``: The total installed capacity of the components in the network.
* ``Expanded Capacity``: The total expanded capacity of the components in the network.
* ``Optimal Capacity``: The total optimal capacity of the components in the network.
