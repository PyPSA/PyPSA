..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _wildcards:

#########
Wildcards
#########

It is easy to run PyPSA-Eur for multiple scenarios using the wildcards feature of ``snakemake``.
Wildcards allow to generalise a rule to produce all files that follow a regular expression pattern
which e.g. defines one particular scenario. One can think of a wildcard as a parameter that shows
up in the input/output file names of the ``Snakefile`` and thereby determines which rules to run,
what data to retrieve and what files to produce.

Detailed explanations of how wildcards work in ``snakemake`` can be found in the
`relevant section of the documentation <https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards>`_.

.. _simpl:

The ``{simpl}`` wildcard
========================

The ``{simpl}`` wildcard specifies number of buses a detailed
network model should be pre-clustered to in the rule
:mod:`simplify_network` (before :mod:`cluster_network`).

.. _clusters:

The ``{clusters}`` wildcard
===========================

The ``{clusters}`` wildcard specifies the number of buses a detailed
network model should be reduced to in the rule :mod:`cluster_network`.
The number of clusters must be lower than the total number of nodes
and higher than the number of countries. However, a country counts twice if
it has two asynchronous subnetworks (e.g. Denmark or Italy).

If an `m` is placed behind the number of clusters (e.g. ``100m``),
generators are only moved to the clustered buses but not aggregated
by carrier; i.e. the clustered bus may have more than one e.g. wind generator.

.. _ll:

The ``{ll}`` wildcard
=====================

The ``{ll}`` wildcard specifies what limits on
line expansion are set for the optimisation model.
It is handled in the rule :mod:`prepare_network`.

The wildcard, in general, consists of two parts:

    1. The first part can be
       ``v`` (for setting a limit on line volume) or
       ``c`` (for setting a limit on line cost)

    2. The second part can be
       ``opt`` or a float bigger than one (e.g. 1.25).

       (a) If ``opt`` is chosen line expansion is optimised
           according to its capital cost
           (where the choice ``v`` only considers overhead costs for HVDC transmission lines, while
           ``c`` uses more accurate costs distinguishing between
           overhead and underwater sections and including inverter pairs).

       (b) ``v1.25`` will limit the total volume of line expansion
           to 25 % of currently installed capacities weighted by
           individual line lengths; investment costs are neglected.

       (c) ``c1.25`` will allow to build a transmission network that
           costs no more than 25 % more than the current system.

.. _opts:

The ``{opts}`` wildcard
=======================

The ``{opts}`` wildcard triggers optional constraints, which are activated in either
:mod:`prepare_network` or the :mod:`solve_network` step.
It may hold multiple triggers separated by ``-``, i.e. ``Co2L-3H`` contains the
``Co2L`` trigger and the ``3H`` switch. There are currently:


.. csv-table::
   :header-rows: 1
   :widths: 10,20,10,10
   :file: configtables/opts.csv

.. _country:

The ``{country}`` wildcard
==========================

The rules :mod:`make_summary` and :mod:`plot_summary` (generating summaries of all or a subselection
of the solved networks) as well as :mod:`plot_p_nom_map` (for plotting the cumulative
generation potentials for renewable technologies) can be narrowed to
individual countries using the ``{country}`` wildcard.

If ``country=all``, then the rule acts on the network for all countries
defined in ``config.yaml``. If otherwise ``country=DE`` or another 2-letter
country code, then the network is narrowed to buses of this country
for the rule. For example to get a summary of the energy generated
in Germany (in the solution for Europe) use:

.. code:: bash

    snakemake -j 1 results/summaries/elec_s_all_lall_Co2L-3H_DE

.. _cutout_wc:

The ``{cutout}`` wildcard
=========================

The ``{cutout}`` wildcard facilitates running the rule :mod:`build_cutout`
for all cutout configurations specified under ``atlite: cutouts:``.
These cutouts will be stored in a folder specified by ``{cutout}``.

.. _technology:

The ``{technology}`` wildcard
=============================

The ``{technology}`` wildcard specifies for which renewable energy technology to produce availablity time
series and potentials using the rule :mod:`build_renewable_profiles`.
It can take the values ``onwind``, ``offwind-ac``, ``offwind-dc``, and ``solar`` but **not** ``hydro``
(since hydroelectric plant profiles are created by a different rule).

The wildcard can moreover be used to create technology specific figures and summaries.
For instance ``{technology}`` can be used to plot regionally disaggregated potentials
with the rule :mod:`plot_p_nom_max`.

.. _attr:

The ``{attr}`` wildcard
=======================

The ``{attr}`` wildcard specifies which attribute is used for size
representations of network components on a map plot produced by the rule
:mod:`plot_network`. While it might be extended in the future, ``{attr}``
currently only supports plotting of ``p_nom``.

.. _ext:

The ``{ext}`` wildcard
======================

The ``{ext}`` wildcard specifies the file type of the figures the
rule :mod:`plot_network`, :mod:`plot_summary`, and :mod:`plot_p_nom_max` produce.
Typical examples are ``pdf`` and ``png``. The list of supported file
formats depends on the used backend. To query the supported file types on your system, issue:

.. code:: python

    import matplotlib.pyplot as plt
    plt.gcf().canvas.get_supported_filetypes()
