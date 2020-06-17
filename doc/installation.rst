################
 Installation
################


Getting Python
==============

If it's your first time with Python, we recommend
`Anaconda <https://www.continuum.io/downloads>`_ as an easy-to-use
environment that includes many basic packages. Anaconda is available
for Windows, Mac OS X and GNU/Linux.

`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ is a minimal installer for conda.

It's always helpful to use a dedicated `conda environment <https://docs.conda.io/en/latest/>`_ or `virtual environment
<https://pypi.python.org/pypi/virtualenv>`_ for your Python
installation (and even easier to use with a `virtualenv-burrito
<https://github.com/brainsik/virtualenv-burrito>`_), in case you
accidentally trash something.



Getting a solver for linear optimisation
========================================

PyPSA passes optimisation problems for :doc:`optimal_power_flow` to an
external solver. PyPSA is known to work with the free software

- `Cbc <https://projects.coin-or.org/Cbc#DownloadandInstall>`_
- `GLPK <https://www.gnu.org/software/glpk/>`_ (`WinGLKP <http://winglpk.sourceforge.net/>`_)

and the non-free software, commercial software (for which free academic licenses are available)

- `Gurobi <https://www.gurobi.com/documentation/quickstart.html>`_
- `CPLEX <https://www.ibm.com/products/ilog-cplex-optimization-studio>`_
- `FICO Xpress <https://www.fico.com/en/products/fico-xpress-optimization>`_

For installation instructions of these solvers for your operating system, follow the links above.

Depending on your operating system, you can also install some of the open-source solvers in a ``conda`` environment.

For GLPK on all operating systems::

    conda install -c conda-forge glpk

For CBC on all operating systems except for Windows::

    conda install -c conda-forge coincbc

.. note::
    Commercial solvers such as Gurobi, CPLEX, and Xpress currently significantly outperform open-source solvers for large-scale problems.
    It might be the case that you can only retrieve solutions by using a commercial solver.


Installing PyPSA with conda
===========================

If you are using ``conda`` you can install PyPSA with::

    conda install -c conda-forge pypsa

or by adding the ``conda-forge`` channel to your ``conda`` installation with::

    conda config --add channels conda-forge

and then installing simply with::

    conda install pypsa


Installing PyPSA with pip
=========================

If you have the Python package installer ``pip`` then just run::

    pip install pypsa

If you're feeling adventurous, you can also install the latest master branch from github with::

    pip install git+https://github.com/PyPSA/PyPSA.git


Manual installation with setuptools
=====================================

PyPSA relies on the following packages which are not contained in a
standard Python installation:

* numpy
* scipy
* pandas
* networkx
* pyomo
* cartopy

It is recommended to use PyPSA with the following additional packages:

* `iPython <http://ipython.org/>`_ for interactive simulations
* `plotly <https://plot.ly/python/>`_ for interactive plotting
* `matplotlib <https://matplotlib.org/>`_ for static plotting
* py.test for unit testing

In a unix-based environment these packages can be obtained with the
`pip <https://pypi.python.org/pypi/pip>`_ Python package manager::

    pip install numpy scipy pandas networkx pyomo ipython


To install PyPSA, you need to download the code from the `PyPSA github
repository <https://github.com/PyPSA/PyPSA/>`_ and then go to the
local repository and run::

    python setup.py install

Or if you want to develop/modify the code in the current directory, run::

    python setup.py develop


Conservative manual installation
================================

If you're very conservative and don't like package managers, you can
just download the code from the `PyPSA github repository
<https://github.com/PyPSA/PyPSA/>`_ and add the directory of PyPSA to
your python path with e.g.::

    import sys

    sys.path.append("path/to/PyPSA")

    import pypsa


.. _upgrading-packages:

Upgrade all packages to the latest versions
===========================================

PyPSA is only tested with the latest stable versions of all the
dependent packages. Therefore it is
very important that you upgrade these packages; otherwise PyPSA may
not work.

To upgrade a package such as ``pandas`` with pip, do at the command
line::

    pip install -U pandas


In Anaconda the `user manual
<http://conda.pydata.org/docs/using/pkgs.html>`_ suggests to upgrade
packages with::

    conda update pandas


.. _upgrading-pypsa:

Upgrading PyPSA
===============

We recommend always keeping your PyPSA installation up-to-date, since
bugs get fixed and new features are added. To upgrade PyPSA with pip,
do at the command line::

    pip install -U pypsa

Don't forget to read the :doc:`release_notes` regarding API changes
that might require you to update your code.
