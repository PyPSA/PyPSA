##########################################
 Introduction
##########################################

PyPSA stands for "Python for Power System Analysis".

PyPSA is a `free software
<http://www.gnu.org/philosophy/free-sw.en.html>`_ toolbox for
simulating and optimising modern electric power systems that include
features such as variable wind and solar generation, storage units and
mixed alternating and direct current networks.

As of 2016 PyPSA is under heavy development and therefore it
is recommended to use caution when using it in a production
environment. Some APIs may change - those liable to be updated are
listed in the :doc:`todo`.

PyPSA was initially developed by the `Renewable Energy Group
<https://fias.uni-frankfurt.de/physics/schramm/complex-renewable-energy-networks/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations
for the `CoNDyNet project <http://condynet.de/>`_, financed by the
German Federal Ministry for Education and Research (BMBF).


What PyPSA does and does not do (yet)
=======================================

PyPSA can calculate:

* static power flow (using both the full non-linear network equations and
  the linearised network equations)
* optimal power flow with the linear network equations (over several
  snapshots simultaneously for optimisation of generation and storage
  dispatch and the capacities of generation, storage and transmission)

It has models for:

* meshed multiply-connected AC and DC networks, with controllable
  converters between AC and DC networks
* conventional dispatchable generators
* generators with time-varying power availability, such as
  wind and solar generators
* storage units with efficiency losses



Functionality that will definitely by added soon (see also :doc:`todo`):

* Graphical plotting of networks with power flow
* Better modelling of hydroelectricity
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


Target user group
=================

PyPSA is intended for researchers, planners and utilities who need a
fast, easy-to-use and transparent tool for power system
analysis. PyPSA is free software and can be arbitrarily extended.



Other comparable software
=========================

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

PyPSA is written and tested with Python 2, but has included Python 3
forward-compatibility (for e.g. printing and integer division) so that
minimal effort should be required to run it in Python 3.

It leans heavily on the following Python packages:

* `pandas <http://ipython.org/>`_ for storing data about components and time series
* numpy and `scipy <http://scipy.org/>`_ for calculations, such as
  linear algebra and sparse matrix calculations
* `pyomo <http://www.pyomo.org/>`_ for preparing optimisation problems (currently only linear)
* networkx for some network calculations (such as discovering connected networks)
* py.test for unit testing

The optimisation uses pyomo so that it is independent of the preferred
solver (you can use e.g. the free software GLPK or the commercial
software Gurobi).

The time-expensive calculations, such as solving sparse linear
equations, are carried out using the scipy.sparse libraries.

Licence
==========

PyPSA is released under the `GPLv3 <http://www.gnu.org/licenses/gpl-3.0.en.html>`_.
