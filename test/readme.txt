

These tests are to be run with py.test:

https://pytest.org/

pip install pytest


Usage:

py.test test_lpf_against_pypower.py


Or to test all scripts just:

py.test


Note that PYPOWER 5.0 has a bug in the linear load flow, which was
fixed in the github version in January 2016.

Note also that the test results against which everything is tested
were generated with the free software LP solver GLPK; other solver may
give other results (e.g. Gurobi can give a slightly better result).
