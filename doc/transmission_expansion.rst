.. _tep:

###############################
Transmission Expansion Planning
###############################

Main Function
-------------

See the module ``pypsa.teplopf``. This module implements a MILP version of
``network.lopf`` for transmission expansion planning and properly
takes account of changing impedances as lines are expanded and binary investment choices
when formulating Kirchhoff's voltage law for candidate lines using a disjunctive relaxation.

.. automethod:: pypsa.Network.teplopf

Utility Functions
-----------------

.. automodule:: pypsa.tepopf