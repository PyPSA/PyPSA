#########
Wildcards
#########

Detailed explanations of how wildcards work in ``snakemake`` can be found in the `relevant section of the documentation <https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards>`_.

The ``simpl`` wildcard
======================

The ``ll`` wildcard
===================

The ``clusters`` wildcard
=========================

.. warning::
    The number of clusters must be lower than the total number of nodes
    and higher than the number of countries. A country counts twice if
    it has two asynchronous subnetworks (e.g. Denmark).

The ``network`` wildcard
========================


The ``opts`` wildcard
=====================

The ``opts`` wildcard triggers optional constraints, which are activated in either ``prepare_network`` or the ``solve_network`` step. It may hold multiple triggers separated by ``-``, i.e. ``Co2L-3H`` contains the ``Co2L`` trigger and the ``3H`` switch. There are currently:


.. csv-table::
   :header-rows: 1
   :widths: 10,20,10,10
   :file: configtables/opts.csv


The ``country`` wildcard
========================

The rules ``make_summary`` (generating summaries of all or a subselection of the solved networks) and ``plot_p_nom_max`` (for plotting the cumulative generation potentials for renewable technologies) can be narrowed to individual countries using the ``country`` wildcard.

If ``country = all``, then the rule acts on the network for all countries defined in ``config.yaml``. If otherwise ``country = DE`` or another country code, then the network is narrowed to buses of this country for the rule. For example to get a summary of the energy generated in Germany (in the solution for Europe) use:


.. code:: bash

    snakemake results/summaries/elec_s_all_lall_Co2L-3H_DE

The ``cutout`` wildcard
=======================

The ``technology`` wildcard
===========================

The ``attr`` wildcard
=====================

The ``ext`` wildcard
====================
