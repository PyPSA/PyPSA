..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

##########################################
Solving Networks
##########################################

After generating and simplifying the networks they can be solved through the rule :mod:`solve_network`  by using the collection rule :mod:`solve_all_networks`. Moreover, networks can be solved for another focus with the derivative rules :mod:`solve_network`  by using the collection rule :mod:`solve_operations_network` for dispatch-only analyses on an already solved network.

.. toctree::
   :caption: Overview

   solving/solve_network
   solving/solve_operations_network
