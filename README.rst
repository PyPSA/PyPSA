################################
Python for Power System Analysis
################################

.. contents::

.. section-numbering::


About
=====

PyPSA stands for "Python for Power System Analysis". It is pronounced "pipes-ah".

PyPSA is a `free software
<http://www.gnu.org/philosophy/free-sw.en.html>`_ toolbox for
simulating and optimising modern power systems that include features
such as conventional generators with unit commitment, variable wind
and solar generation, storage units, sector coupling and mixed
alternating and direct current networks. PyPSA is designed to scale
well with large networks and long time series.

As of 2017 PyPSA is under heavy development and therefore it is
recommended to use caution when using it in a production environment.
Some APIs may change - the changes in each PyPSA version are listed in
the `doc/release_notes.rst <doc/reslease_notes.rst>`_.



PyPSA was initially developed by the `Renewable Energy Group
<https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations
for the `CoNDyNet project <http://condynet.de/>`_, financed by the
`German Federal Ministry for Education and Research (BMBF) <https://www.bmbf.de/en/index.html>`_ as part of the `Stromnetze Research Initiative <http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/>`_.


Documentation
=============

`Documentation as a website <http://www.pypsa.org/doc/index.html>`_

`Documentation as a PDF <http://www.pypsa.org/doc/PyPSA.pdf>`_

`Quick start <http://www.pypsa.org/doc/quick_start.html>`_

`Examples <http://www.pypsa.org/examples/>`_

Documentation is in `sphinx
<http://www.sphinx-doc.org/en/stable/>`_ reStructuredText format in
`doc/ <doc/>`_.


What PyPSA does and does not do (yet)
=======================================

PyPSA can calculate:

* static power flow (using both the full non-linear network equations and
  the linearised network equations)
* linear optimal power flow (optimisation of power plant and storage
  dispatch within network constraints, using the linear network
  equations, over several snapshots)
* security-constrained linear optimal power flow
* total electricity system investment optimisation (using linear
  network equations, over several snapshots simultaneously for
  optimisation of generation and storage dispatch and investment in
  the capacities of generation, storage and transmission)

It has models for:

* meshed multiply-connected AC and DC networks, with controllable
  converters between AC and DC networks
* standard types for lines and transformers following the implementation in `pandapower <https://www.uni-kassel.de/eecs/fachgebiete/e2n/software/pandapower.html>`_
* conventional dispatchable generators with unit commitment
* generators with time-varying power availability, such as
  wind and solar generators
* storage units with efficiency losses
* simple hydroelectricity with inflow and spillage
* coupling with other energy carriers
* basic components out of which more complicated assets can be built,
  such as Combined Heat and Power (CHP) units, heat pumps, resistive
  Power-to-Heat (P2H), Power-to-Gas (P2G), battery electric vehicles
  (BEVs), etc.; each of these is demonstrated in the `examples
  <http://www.pypsa.org/examples/>`_


Functionality that will definitely be added soon:

* Multi-year investment optimisation
* Simple RMS simulations with the swing equation
* Distributed active power slack
* Non-linear power flow solution using `analytic continuation
  <https://en.wikipedia.org/wiki/Holomorphic_embedding_load_flow_method>`_
  in the complex plane following `GridCal
  <https://github.com/SanPen/GridCal>`_

Functionality that may be added in the future:

* Short-circuit current calculations
* Dynamic RMS simulations
* Small signal stability analysis
* Interactive web-based GUI with SVG
* OPF with the full non-linear network equations
* Dynamic EMT simulations
* Unbalanced load flow
* Port to `Julia <http://julialang.org/>`_


Example scripts as Jupyter/iPython notebooks
============================================

There are `extensive examples <http://www.pypsa.org/examples/>`_ available as Jupyter/iPython notebooks. They are also described in the `doc/examples.rst <doc/examples.rst>`_ and are available as Python scripts in `examples/ <examples/>`_.

Screenshots
===========

.. image:: http://www.pypsa.org/img/line-loading.png

.. image:: http://www.pypsa.org/img/lmp.png

.. image:: http://www.pypsa.org/img/reactive-power.png

.. image:: http://www.pypsa.org/img/stacked-gen.png

.. image:: http://www.pypsa.org/img/storage-scigrid.png

.. image:: http://www.pypsa.org/img/scigrid-curtailment.png

.. image:: http://www.pypsa.org/img/meshed-ac-dc.png

.. image:: http://www.pypsa.org/img/euro-pie-pre-7-branch_limit-1-256.png

Optimised capacities of generation and storage for a 95% reduction in CO2 emissions in Europe compare to 1990 levels:

.. image:: http://www.pypsa.org/img/euro-pie-pre-7-branch_limit-1-256.png
.. image:: http://www.pypsa.org/img/legend-flat.png



What PyPSA uses under the hood
===============================

PyPSA is written and tested to be compatible with both Python 2.7 and
Python 3.5.

It leans heavily on the following Python packages:

* `pandas <http://ipython.org/>`_ for storing data about components and time series
* `numpy <http://www.numpy.org/>`_ and `scipy <http://scipy.org/>`_ for calculations, such as
  linear algebra and sparse matrix calculations
* `pyomo <http://www.pyomo.org/>`_ for preparing optimisation problems (currently only linear)
* `networkx <https://networkx.github.io/>`_ for some network calculations
* `py.test <http://pytest.org/>`_ for unit testing
* `logging <https://docs.python.org/3/library/logging.html>`_ for managing messages

The optimisation uses pyomo so that it is independent of the preferred
solver (you can use e.g. the free software GLPK or the commercial
software Gurobi).

The time-expensive calculations, such as solving sparse linear
equations, are carried out using the scipy.sparse libraries.



Mailing list
============

PyPSA has a Google Group `forum / mailing list
<https://groups.google.com/group/pypsa>`_.


Citing PyPSA
============



If you use PyPSA for your research, we would appreciate it if you
would cite the following preprint paper (which has not yet been
through peer review):

* T. Brown, J. Hörsch, D. Schlachtberger, `PyPSA: Python for Power
  System Analysis <https://arxiv.org/abs/1707.09913>`_, 2017,
  `preprint arXiv:1707.09913 <https://arxiv.org/abs/1707.09913>`_

If you want to cite a specific PyPSA version, you can cite the Zenodo
DOI for each release, e.g. for the latest release:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.582307.svg
   :target: https://doi.org/10.5281/zenodo.582307


Licence
=======

Copyright 2015-2017 Tom Brown (FIAS), Jonas Hörsch (FIAS), David
Schlachtberger (FIAS)

This program is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either `version 3 of the
License <LICENSE.txt>`_, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
`GNU General Public License <LICENSE.txt>`_ for more details.
