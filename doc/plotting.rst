##########################################
Plotting and Summary
##########################################

.. _summary:

Make Summary
============

.. todo:: move to ``make_summary`` autodoc

The following rule can be used to summarize the results in seperate .csv files:

.. code::
    snakemake results/summaries/elec_s_all_lall_Co2L-3H_all
                                        ^ clusters
                                            ^ line volume or cost cap
                                                ^- options
                                                        ^- all countries

the line volume/cost cap field can be set to one of the following:
* ``lv1.25`` for a particular line volume extension by 25%
* ``lc1.25`` for a line cost extension by 25 %
* ``lall`` for all evalutated caps
* ``lvall`` for all line volume caps
* ``lcall`` for all line cost caps

Replacing '/summaries/' with '/plots/' creates nice colored maps of the results.

.. automodule:: make_summary

.. _summary_plot:

Plot Summary
============

.. automodule:: plot_summary

.. _map_plot:

Plot Network
============

.. automodule:: plot_network

.. image:: img/tech-colors.png
    :align: center
