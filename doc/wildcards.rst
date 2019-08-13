.. _wildcards:

#########
Wildcards
#########

Detailed explanations of how wildcards work in ``snakemake`` can be found in the `relevant section of the documentation <https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards>`_.

.. _network:

The ``{network}`` wildcard
==========================

The ``{network}`` wildcard specifies the considered energy sector(s)
and, as currently only ``elec`` (for electricity) is included,
it currently represents rather a placeholder wildcard to facilitate
future extensions including multiple energy sectors at once.

.. _simpl:

The ``{simpl}`` wildcard
========================

The ``{simpl}`` wildcard specifies number of buses a detailed network model should be pre-clustered to in the rule ``simplify_network`` (before ``cluster_network``).

.. seealso::
    :mod:`simplify_network`

.. _clusters:

The ``{clusters}`` wildcard
===========================

The ``{clusters}`` wildcard specifies the number of buses a detailed network model should be reduced to in the rule ``cluster_network``.
The number of clusters must be lower than the total number of nodes
and higher than the number of countries. However, a country counts twice if
it has two asynchronous subnetworks (e.g. Denmark or Italy).

If an `m` is placed behind the number of clusters (e.g. ``100m``), generators are only moved to the clustered buses but not aggregated by carrier; i.e. the clustered bus may have more than one e.g. wind generator.

.. seealso::
    :mod:`cluster_network`

.. _ll:

The ``{ll}`` wildcard
=====================

``v`` (volume) or ``c`` (cost)

``opt`` or a float bigger than one (e.g. 1.25)

.. seealso::
    :mod:`prepare_network`

.. _opts:

The ``{opts}`` wildcard
=======================

The ``{opts}`` wildcard triggers optional constraints, which are activated in either
``prepare_network`` or the ``solve_network`` step.
It may hold multiple triggers separated by ``-``, i.e. ``Co2L-3H`` contains the
``Co2L`` trigger and the ``3H`` switch. There are currently:


.. csv-table::
   :header-rows: 1
   :widths: 10,20,10,10
   :file: configtables/opts.csv

.. seealso::
    :mod:`prepare_network`, :mod:`solve_network`

.. _country:

The ``{country}`` wildcard
==========================

The rules ``make_summary`` and ``plot_summary`` (generating summaries of all or a subselection
of the solved networks) as well as ``plot_p_nom_max`` (for plotting the cumulative
generation potentials for renewable technologies) can be narrowed to
individual countries using the ``{country}`` wildcard.

If ``country = all``, then the rule acts on the network for all countries
defined in ``config.yaml``. If otherwise ``country = DE`` or another 2-letter
country code, then the network is narrowed to buses of this country
for the rule. For example to get a summary of the energy generated
in Germany (in the solution for Europe) use:

.. code:: bash

    snakemake results/summaries/elec_s_all_lall_Co2L-3H_DE

.. seealso::
    :mod:`make_summary`, :mod:`plot_summary`, :mod:`plot_p_nom_max`

.. _cutout_wc:

The ``{cutout}`` wildcard
=========================

.. seealso::
    :mod:`build_cutout`

.. _technology:

The ``{technology}`` wildcard
=============================

.. seealso::
    :mod:`build_renewable_profiles`, :mod:`plot_p_nom_max`, :mod:`build_country_flh`

.. _attr:

The ``{attr}`` wildcard
=======================

The ``{attr}`` wildcard specifies which attribute are used for size representations of network components on a map plot produced by the rule ``plot_network``. While it might be extended in the future, ``{attr}`` currently only supports plotting of ``p_nom``.

.. seealso::
    :mod:`plot_network`

.. _ext:

The ``{ext}`` wildcard
======================

The ``{ext}`` wildcard specifies the file type of the figures the rule ``plot_network``, ``plot_summary``, and ``plot_p_nom_max`` produce. Typical examples are ``pdf`` and ``png``. The list of supported file formats depends on the used backend. To query the supported file types on your system, issue:

.. code:: python

    import matplotlib.pyplot as plt
    plt.gcf().canvas.get_supported_filetypes()

.. seealso::
    :mod:`plot_network`, :mod:`plot_summary`, :mod:`plot_p_nom_max`