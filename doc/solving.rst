##########################################
Solving Networks
##########################################

After generating and simplifying the networks they can be solved through the rule ``solve_network``  by using the collection rule ``solve_all_elec_networks``. Moreover, networks can be solved for another focus with the derivative rules ``trace_solve_network`` to log changes during iterations and ``solve_operations_network`` for dispatch-only analyses on an already solved network.

.. _solve:

Solve Network
=============

.. automodule:: solve_network

.. _trace_solve:

Trace Solve Network
===================

.. automodule:: trace_solve_network

.. _solve_operations:

Solve Operations Network
========================

.. automodule:: solve_operations_network
