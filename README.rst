

Python for Power Systems Analysis
=================================


PyPSA stands for "Python for Power System Analysis". It is pronounced "pipes-ah".

PyPSA is a `free software
<http://www.gnu.org/philosophy/free-sw.en.html>`_ toolbox for
simulating and optimising modern electric power systems that include
features such as variable wind and solar generation, storage units and
mixed alternating and direct current networks.

Documentation can be found in `sphinx
<http://www.sphinx-doc.org/en/stable/>`_ reStructuredText format in
`doc/ <doc/>`_ and as a `website
<http://pypsa.org/doc/>`_ and as a `pdf
<http://pypsa.org/doc/PyPSA.pdf>`_.


As of 2016 PyPSA is under heavy development and therefore it
is recommended to use caution when using it in a production
environment. Some APIs may change - those liable to be updated are
listed in the `doc/todo.rst <doc/todo.rst>`_.

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
* optimal power flow (optimisation of power plant dispatch within
  network constraints, using the linear network equations)
* total electricity system optimisation (using linear network
  equations, over several snapshots simultaneously for optimisation of
  generation and storage dispatch and the capacities of generation,
  storage and transmission)

It has models for:

* meshed multiply-connected AC and DC networks, with controllable
  converters between AC and DC networks
* conventional dispatchable generators
* generators with time-varying power availability, such as
  wind and solar generators
* storage units with efficiency losses



Functionality that will definitely by added soon (see also `doc/todo.rst <doc/todo.rst>`_):

* Graphical plotting of networks with power flow
* Better modelling of hydroelectricity
* Integration of heating sector (CHPs, heat pumps, etc.)
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



What PyPSA uses under the hood
===============================

PyPSA is written and tested to be compatible with both Python 2.7 and
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

Licence
==========

PyPSA is released as free software under the `GPLv3
<http://www.gnu.org/licenses/gpl-3.0.en.html>`_, see `LICENSE.txt
<LICENSE.txt>`_.
