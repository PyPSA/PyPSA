..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

##################
Cost Assumptions
##################

The database of cost assumptions is stored in ``data/costs.csv``.

It includes cost assumptions for all included technologies for specific
years from various sources, namely for

- discount rate,
- lifetime,
- investment (CAPEX),
- fixed operation and maintenance (FOM),
- variable operation and maintenance (VOM),
- fuel costs,
- efficiency, and
- carbon-dioxide intensity.

The given overnight capital costs are annualised to net present costs
with a discount rate of :math:`r` over the economic lifetime :math:`n` using the annuity factor

.. math::

    a = \frac{1-(1+r)^{-n}}{r}.

Based on the parameters above the ``marginal_cost`` and ``capital_cost`` of the system components are calculated.

.. note::

    Another great resource for cost assumptions is the `cost database from the Danish Energy Agency <https://ens.dk/en/our-services/projections-and-models/technology-data>`_.

Modifying Cost Assumptions
==========================

Some cost assumptions (e.g. marginal cost and capital cost) can be directly overwritten in the ``config.yaml`` (cf. Section  :ref:`costs_cf`  in :ref:`config`).

To change cost assumptions in more detail, modify cost assumptions directly in ``data/costs.csv`` as this is not yet supported through the config file.

You can also build multiple different cost databases. Make a renamed copy of ``data/costs.csv`` (e.g. ``data/costs-optimistic.csv``) and set the variable ``COSTS=data/costs-optimistic.csv`` in the ``Snakefile``.


Default Cost Assumptions
========================

.. csv-table::
   :header-rows: 1
   :widths: 10,3,5,4,6,8
   :file: ../data/costs.csv
