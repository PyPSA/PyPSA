########################
Unit Testing
########################


Unit testing is performed with ``pytest`` that can be installed via

.. code::

    pip install pytest

The tests can be found in ``pypsa/test/`` and can be run from there via

.. code::

    pytest

Or to run individual tests:

.. code::

    pytest test_lpf_against_pypower.py

Power flow is tested against PYPOWER (the Python implementation of MATPOWER)
and pandapower.

.. warning::

    Note that PYPOWER 5.0 has a bug in the linear load flow, which was fixed in the github version in January 2016.

.. note::

    Note also that the test results against which everything is tested
    were generated with the free software LP solver GLPK; other solver may
    give other results (e.g. Gurobi can give a slightly better result).


Unit testing of new GitHub commits is automated with Github Actions.
