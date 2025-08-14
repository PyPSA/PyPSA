##########################################
 Introduction
##########################################

Functionality
=============

**PyPSA can calculate:**

* static power flow (using both the full non-linear network equations and
  the linearised network equations)
* linear optimal power flow (least-cost optimisation of power plant and storage
  dispatch within network constraints, using the linear network
  equations, over several snapshots)
* security-constrained linear optimal power flow
* total electricity/energy system least-cost investment optimisation (using linear
  network equations, over several snapshots simultaneously for
  optimisation of generation and storage dispatch and investment in
  the capacities of generation, storage, transmission and other infrastructure)

**It has models for:**

* meshed multiply-connected AC and DC networks, with controllable
  converters between AC and DC networks
* standard types for lines and transformers following the implementation in `pandapower <https://www.pandapower.org>`_
* conventional dispatchable generators and links with unit commitment
* generators with time-varying power availability, such as
  wind and solar generators
* storage units with efficiency losses
* simple hydroelectricity with inflow and spillage
* coupling with other energy carriers
* basic components out of which more complicated assets can be built,
  such as Combined Heat and Power (CHP) units, heat pumps, resistive
  Power-to-Heat (P2H), Power-to-Gas (P2G), battery electric vehicles
  (BEVs), Fischer-Tropsch, direct air capture (DAC), etc.; each of
  these is demonstrated in the `examples
  <https://pypsa.readthedocs.io/en/latest/examples-basic.html>`_


Target users
============

PyPSA is intended for researchers, planners and utilities who need a
fast, easy-to-use and transparent tool for power and energy system
analysis. PyPSA is free software and can be arbitrarily extended.


Screenshots
===========


* `PyPSA-Eur <https://github.com/PyPSA/pypsa-eur>`_ optimising capacities of generation, storage and transmission lines (9% line volume expansion allowed) for a 95% reduction in CO2 emissions in Europe compared to 1990 levels

.. image:: ../img/elec_s_256_lv1.09_Co2L-3H.png
    :align: center
    :width: 700px


*  `SciGRID model <https://power.scigrid.de/>`_ simulating the German power system for 2015. Interactive plots also be generated with the `plotly <https://plot.ly/python/>`_ library, as shown in this `Notebook <https://pypsa.readthedocs.io/en/latest/examples/scigrid-lopf-then-pf.html>`_

.. image:: ../img/stacked-gen_and_storage-scigrid.png
    :align: center

.. image:: ../img/lmp_and_line-loading.png
    :align: right


.. image:: ../img/reactive-power.png
    :align: center
    :width: 600px


* Small meshed AC-DC toy model

.. image:: ../img/ac_dc_meshed.png
    :align: center
    :width: 400px



Dependencies
============

PyPSA is written and tested to be compatible with Python 3.10 and
above.


It leans heavily on the following Python packages:

* `pandas <http://pandas.pydata.org/>`_ for storing data about components and time series
* `numpy <http://www.numpy.org/>`_ and `scipy <http://scipy.org/>`_ for calculations, such as
  linear algebra and sparse matrix calculations
* `matplotlib <https://matplotlib.org/>`_ for static plotting
* `cartopy <https://scitools.org.uk/cartopy>`_ for plotting the baselayer map
* `networkx <https://networkx.github.io/>`_ for some network calculations
* `linopy <https://github.com/PyPSA/linopy>`_ for preparing optimisation problems (currently only linear and mixed-integer linear)
* `pytest <http://pytest.org/>`_ for unit testing
* `logging <https://docs.python.org/3/library/logging.html>`_ for managing messages


The optimisation uses solver interfaces that are independent of the preferred
solver. You can use e.g. one of the free solvers `HiGHS <https://highs.dev/>`_,
`GLPK <https://www.gnu.org/software/glpk/>`_ and `CLP/CBC
<https://github.com/coin-or/Cbc/>`_ or commercial solvers like `Gurobi
<http://www.gurobi.com/>`_ or `CPLEX
<https://www.ibm.com/de-de/analytics/cplex-optimizer>`_ for which free academic
licenses are available.

Licence
=======

Copyright 2015-2025 :doc:`/references/developers`

PyPSA is licensed under the open source `MIT License <https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt>`_.
