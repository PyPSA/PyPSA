################
 Installation
################


Getting Python
==============

If it's your first time with Python, people seem to recommend
`Anaconda <https://www.continuum.io/downloads>`_ as an easy-to-use
environment that includes many basic packages. Anaconda is available
for Windows, Mac OS X and GNU/Linux.


For those rolling their own on unix-like systems (GNU/Linux, Mac OS X)
it's always helpful to use a `virtual environment
<https://pypi.python.org/pypi/virtualenv>`_ for your python
installation (and even easier to use with a `virtualenv-burrito
<https://github.com/brainsik/virtualenv-burrito>`_), in case you
accidentally trash something.



Getting a solver for linear optimisation
========================================

PyPSA is known to work with the free software GLPK (versions below
4.58) and the non-free software Gurobi (and whatever else Pyomo works
with).

For Debian-based systems you can get GLPK with::

    sudo apt-get install glpk-utils

and there are similar packages for other GNU/Linux distributions.

For Windows there is `WinGLPK <http://winglpk.sourceforge.net/>`_. For
Mac OS X `brew <http://brew.sh/>`_ is your friend.

WARNING: PyPSA uses the library Pyomo, which is `known not to work
<https://software.sandia.gov/trac/pyomo/ticket/4641>`_ with GLPK
versions >= 4.58. Please install an older version of GLPK.



Installing PyPSA with pip
=========================

If you have the Python package installer ``pip`` then just run::

    pip install pypsa

"Manual" installation with setuptools
=====================================

PyPSA relies on the following packages which are not contained in a
standard Python installation:

* numpy
* scipy
* pandas
* networkx
* pyomo

It is recommended to use PyPSA with the following additional packages:

* `iPython <http://ipython.org/>`_ for interactive simulations
* py.test for unit testing

In a unix-based environment these packages can be obtained with the
`pip <https://pypi.python.org/pypi/pip>`_ Python package manager::

    pip install numpy scipy pandas networkx pyomo ipython


To install PyPSA, you need to download the code from the `PyPSA github
repository <https://github.com/fresna/pypsa/>`_ and then go to the
local repository and run::

    python setup.py install

Or if you want to develop/modify the code in the current directory, run::

    python setup.py develop


Conservative manual installation
================================

If you're very conservative and don't like package managers, you can
just download the code from the `PyPSA github repository
<https://github.com/fresna/pypsa/>`_ and add the directory of PyPSA to
your python path with e.g.::

    import sys

    sys.path.append("path/to/PyPSA")

    import pypsa

Upgrade all packages to the latest versions
===========================================

PyPSA is only tested with the latest stable versions of all the
dependent packages (pandas, networkx, etc.). Therefore it is very
important that you upgrade these packages; otherwise
PyPSA may not work.

To upgrade a package such as ``pandas`` with pip, do at the command line::

    pip install -U pandas



In Anaconda the `user manual <http://conda.pydata.org/docs/using/pkgs.html>`_ suggests to upgrade packages with::

    conda update pandas
