##########################################
 Introduction
##########################################

PyPSA stands for "Python for Power System Analysis".

PyPSA is intended to be a toolbox for simulating modern electric power
systems that include features such as variable wind and solar
generation, storage units and mixed alternating and direct current
networks.

As of 2016 PyPSA is under heavy development and therefore it
is recommended to use caution when using it in a production
environment. Some APIs may change - those liable to be updated are
listed in XXX.

PyPSA was initially developed by the `Renewable Energy Group
<https://fias.uni-frankfurt.de/physics/schramm/complex-renewable-energy-networks/>`_
at `FIAS <https://fias.uni-frankfurt.de/>`_ to carry out simulations
financed by the CoNDyNet project.

What PyPSA does and does not do (yet)
===================================

PyPSA can calculate:

* static power flow (using both the full non-linear equations and
  the linearised equations)
* linear optimal power flow (over several snapshots
  simultaneously for capacity and storage optimisation)

It has models for:

* meshed multiply-connected AC and DC networks, with converters between AC and DC
* conventional dispatchable generators
* generators with time-varying power availability, such as
  wind and solar generators
* storage units with efficiency losses



Functionality that will definitely by added soon (see also TODO xxxx):

* Plotting of networks with power flow
* Better modelling of hydroelectricity

Functionality that may be added in the future:

* Unit Commitment using MILP
* Short-circuit current calculations
* Dynamic RMS simulations
* Interactive web-based GUI
* AC OPF


Target user group
=================

PyPSA is intended for researchers, planners and utlities who need a
fast and transparent tool for power system analysis, which
can be arbitrarily extended.



Other comparable software
=========================

PyPSA is not as fully featured as other power system simulation tools
such as the Matlab-based free software `PSAT
<http://faraday1.ucd.ie/psat.html>`_ or the commercial package
`DIgSILENT PowerFactory
<http://www.digsilent.de/index.php/products-powerfactory.html>`_.

However for power flow and optimal power flow over several time
snapshots, it offers the flexibility of Python and the transparency of
free software.

Another Python power system tool is `PYPOWER
<https://github.com/rwl/PYPOWER/>`_, which is based on the
Matlab-based `MATPOWER <http://www.pserc.cornell.edu//matpower/>`_. In
contrast to PYPOWER, PyPSA has an easier-to-use data model (objects
and pandas DataFrames instead of numpy arrays), support for
time-varying data inputs and support for multiply-connected networks
using both AC and DC.



What PyPSA uses under the hood
===============================

PyPSA is written and tested with Python 2, but has included Python 3
forward-compatibility (for e.g. printing and integer division) so that
minimal effort should be required to run it in Python 3.

It leans heavily on the following Python packages:

* `pandas <http://ipython.org/>`_ for storing data about components and time series
* numpy and `scipy <http://scipy.org/>`_ for calcuations, such as
  linear algebra and sparse matrix calculations
* `pyomo <http://www.pyomo.org/>`_ for preparing optimisation problems (currently only linear)
* networkx for some network calculations (such as discovering connected networks)

The optimisation uses pyomo so that it is independent of the preferred
solver (you can use e.g. the free software GLPK or the commercial
software Gurobi).

The time-expensive calculations, such as solving sparse linear
equations, are carried out using the scipy.sparse libraries.

Licence
==========

PyPSA is released under the `GPLv3 <http://www.gnu.org/licenses/gpl-3.0.en.html>`_.
