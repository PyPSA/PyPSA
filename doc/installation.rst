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


Getting a solver for optimisation
=================================

PyPSA passes optimisation problems for :doc:`optimal_power_flow` to an
external solver. PyPSA is known to work with the free software

- `Cbc <https://projects.coin-or.org/Cbc#DownloadandInstall>`_
- `GLPK <https://www.gnu.org/software/glpk/>`_ (`WinGLKP <http://winglpk.sourceforge.net/>`_)
- `HiGHS <https://highs.dev/>`_

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

In all of the above commands you can replace ``conda`` with ``mamba`` if you use this alternative.


Installing PyPSA with pip
=========================

If you have the Python package installer ``pip`` then just run::

    pip install pypsa

If you're feeling adventurous, you can also install the latest master branch from github with::

    pip install git+https://github.com/PyPSA/PyPSA.git


Conservative installation
=========================

If you're very conservative and don't like package managers, you can
just download the code from the `PyPSA github repository
<https://github.com/PyPSA/PyPSA/>`_ and add the directory of PyPSA to
your python path with e.g.::

    import sys

    sys.path.append("path/to/PyPSA")

    import pypsa


.. _upgrading-packages:

Upgrading dependencies
======================

PyPSA is only tested with the latest stable versions of all the
dependent packages for the respective Python versions. Therefore it is
very important that you upgrade these packages; otherwise PyPSA may
not work.

To upgrade a package such as ``pandas`` with pip, do at the command
line::

    pip install -U pandas


With ``conda/mamba`` upgrade packages with::

    conda update pandas


.. _upgrading-pypsa:

Upgrading PyPSA
===============

We recommend always keeping your PyPSA installation up-to-date, since
bugs get fixed and new features are added.

To upgrade PyPSA with pip, do at the command line::

    pip install -U pypsa

To upgrade PyPSA with conda, do at the command line::

    conda update pypsa

Don't forget to read the :doc:`release_notes` regarding API changes
that might require you to update your code.
