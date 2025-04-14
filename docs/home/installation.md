################
 Installation
################


Getting Python
==============

If it is your first time with Python, we recommend `conda
<https://docs.conda.io/en/latest/miniconda.html>`_, `mamba
<https://github.com/mamba-org/mamba>`_ or `pip
<https://pip.pypa.io/en/stable/>`_ as easy-to-use package managers. They are
available for Windows, Mac OS X and GNU/Linux.

It is always helpful to use dedicated `conda/mamba environments <https://mamba.readthedocs.io/en/latest/user_guide/mamba.html>`_ or `virtual environments
<https://pypi.python.org/pypi/virtualenv>`_.


Installation with conda
=======================

If you are using ``conda`` you can install PyPSA with::

    conda install -c conda-forge pypsa

Replace ``conda`` with ``mamba`` if you use this alternative.


Installing with pip
===================

If you have the Python package installer ``pip`` then just run::

    pip install pypsa

If you're feeling adventurous, you can also install the latest master branch from github with::

    pip install git+https://github.com/PyPSA/PyPSA.git

Getting a solver
================

PyPSA passes optimisation problems for :doc:`/user-guide/optimal-power-flow` to an
external solver. PyPSA is known to work via ``linopy`` with the free software

- `HiGHS <https://highs.dev/>`_
- `Cbc <https://projects.coin-or.org/Cbc#DownloadandInstall>`_
- `GLPK <https://www.gnu.org/software/glpk/>`_ (`WinGLKP <http://winglpk.sourceforge.net/>`_)
- `SCIP <https://scip.zib.de/>`_

and the non-free software, commercial software (for some of which free academic licenses are available)

- `Gurobi <https://www.gurobi.com/documentation/quickstart.html>`_
- `CPLEX <https://www.ibm.com/products/ilog-cplex-optimization-studio>`_
- `FICO Xpress <https://www.fico.com/en/products/fico-xpress-optimization>`_
- `MOSEK <https://www.mosek.com/>`_
- `COPT <https://www.shanshu.ai/copt>`_
- `MindOpt <https://solver.damo.alibaba.com/doc/en/html/index.html>`_

An installation of PyPSA will automatically install the default solver HiGHS.
For installation instructions of further solvers for your operating system,
follow the links above.

.. note::
    Commercial solvers such as Gurobi, CPLEX, and Xpress currently significantly outperform open-source solvers for large-scale problems.
    It might be the case that you can only retrieve solutions by using a commercial solver.

.. _upgrading-pypsa:

Upgrading PyPSA
===============

We recommend always keeping your PyPSA installation up-to-date, since bugs get
fixed and new features are added. PyPSA is also only tested with the latest
stable versions of all the dependent packages for the respective Python
versions.

To upgrade PyPSA with pip, do at the command line::

    pip install -U pypsa

To upgrade PyPSA with conda, do at the command line::

    conda update pypsa

Don't forget to read the :doc:`/references/release-notes` regarding API changes
that might require you to update your code.

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