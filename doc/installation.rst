################
 Installation
################

Installation with pip
=====================

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

Conservative manual installation
================================

If you're very conservative and don't like package managers, you can
just download the code from the `PyPSA github repository
<https://github.com/fresna/pypsa/>`_ and add the directory of PyPSA to
your python path with e.g.::

    import sys

    sys.path.append("path/to/PyPSA")

    import pypsa
