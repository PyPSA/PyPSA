##########################################
 Introduction
##########################################

PyPSA stands for "Python for Power System Analysis". It is pronounced "pipes-ah".

PyPSA is a `free software
<http://www.gnu.org/philosophy/free-sw.en.html>`_ toolbox for
simulating and optimising modern electric power systems that include
features such as variable wind and solar generation, storage units and
mixed alternating and direct current networks.  PyPSA is designed to
scale well with large networks and long time series.


As of 2016 PyPSA is under heavy development and therefore it
is recommended to use caution when using it in a production
environment. Some APIs may change - those liable to be updated are
listed in the :doc:`todo`.

PyPSA was initially developed by the `Renewable Energy Group
<https://fias.uni-frankfurt.de/physics/schramm/complex-renewable-energy-networks/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations
for the `CoNDyNet project <http://condynet.de/>`_, financed by the
`German Federal Ministry for Education and Research (BMBF) <https://www.bmbf.de/en/index.html>`_ as part of the `Stromnetze <http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/>`_ Research Initiative.


What PyPSA does and does not do (yet)
=======================================

PyPSA can calculate:

* static power flow (using both the full non-linear network equations and
  the linearised network equations)
* optimal power flow (optimisation of power plant dispatch within
  network constraints, using the linear network equations)
* total electricity system optimisation (using linear network
  equations, over several snapshots simultaneously for optimisation of
  generation and storage dispatch and investment in the capacities of generation,
  storage and transmission)

It has models for:

* meshed multiply-connected AC and DC networks, with controllable
  converters between AC and DC networks
* conventional dispatchable generators
* generators with time-varying power availability, such as
  wind and solar generators
* storage units with efficiency losses
* simple hydroelectricity with inflow and spillage



Functionality that will definitely by added soon (see also :doc:`todo`):

* Graphical plotting of networks with power flow
* Integration of heating sector (CHPs, heat pumps, etc.)
* Security Constrained Linear OPF
* Simple RMS simulations with the swing equation
* Distributed active power slack
* Non-linear power flow solution using `analytic continuation <https://en.wikipedia.org/wiki/Holomorphic_embedding_load_flow_method>`_ in the complex plane

Functionality that may be added in the future:

* Unit Commitment using MILP
* Short-circuit current calculations
* Dynamic RMS simulations
* Small signal stability analysis
* Interactive web-based GUI
* OPF with the full non-linear network equations
* Dynamic EMT simulations
* Unbalanced load flow
* Port to `Julia <http://julialang.org/>`_


Screenshots and example Jupyter/iPython notebooks
=================================================

There are `extensive examples <http://www.pypsa.org/examples/>`_ available as Jupyter/iPython notebooks. They are also described in the :doc:`examples` and are available as Python scripts in ``examples/``.

Some screenshots:


.. image:: img/line-loading.png

.. image:: img/lmp.png

.. image:: img/reactive-power.png

.. image:: img/stacked-gen.png

.. image:: img/storage-scigrid.png

.. image:: img/scigrid-curtailment.png

.. image:: img/meshed-ac-dc.png

.. image:: img/europe-map.png


Target user group
=================

PyPSA is intended for researchers, planners and utilities who need a
fast, easy-to-use and transparent tool for power system
analysis. PyPSA is free software and can be arbitrarily extended.



Other comparable software
=========================

For a full list see :doc:`comparable_software`.

PyPSA is not as fully featured as other power system simulation tools
such as the Matlab-based free software `PSAT
<http://faraday1.ucd.ie/psat.html>`_ or the commercial package
`DIgSILENT PowerFactory
<http://www.digsilent.de/index.php/products-powerfactory.html>`_.

However for power flow and optimal power flow over several time
snapshots with variable renewable energy sources and/or storage and/or
mixed AC-DC systems, it offers the flexibility of Python and the
transparency of free software.

Another Python power system tool is `PYPOWER
<https://github.com/rwl/PYPOWER/>`_, which is based on the
Matlab-based `MATPOWER <http://www.pserc.cornell.edu//matpower/>`_. In
contrast to PYPOWER, PyPSA has an easier-to-use data model (objects
and pandas DataFrames instead of numpy arrays), support for
time-varying data inputs and support for multiply-connected networks
using both AC and DC. PyPSA uses some of the sparse-matrix constructs
from PYPOWER.



What PyPSA uses under the hood
===============================

PyPSA is written and tested to be compatible with Python 2.7 and
Python 3.4.

It leans heavily on the following Python packages:

* `pandas <http://ipython.org/>`_ for storing data about components and time series
* `numpy <http://www.numpy.org/>`_ and `scipy <http://scipy.org/>`_ for calculations, such as
  linear algebra and sparse matrix calculations
* `pyomo <http://www.pyomo.org/>`_ for preparing optimisation problems (currently only linear)
* `networkx <https://networkx.github.io/>`_ for some network calculations (such as discovering connected networks)
* `py.test <http://pytest.org/>`_ for unit testing

The optimisation uses pyomo so that it is independent of the preferred
solver (you can use e.g. the free software GLPK or the commercial
software Gurobi).

The time-expensive calculations, such as solving sparse linear
equations, are carried out using the scipy.sparse libraries.

Mailing list
============

PyPSA has a Google Group `forum / mailing list
<https://groups.google.com/group/pypsa>`_.




Licence
==========

PyPSA is released under the `GPLv3 <http://www.gnu.org/licenses/gpl-3.0.en.html>`_.
