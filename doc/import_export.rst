######################
Data Import and Export
######################

PyPSA is intended to be data format agnostic, but given the reliance
internally on pandas DataFrames, it is natural to use
comma-separated-variable (CSV) files.

The import-export functionality can be found in pypsa/io.py.


Import from CSV
===============

File for each component type, then file for each time-dependent variable.

Adding components one-by-one
============================

Networks can also be built "manually" by calling

.. code:: python

    network.add("Bus","my_bus_0")
    network.add("Bus","my_bus_1")
    network.add("Line","my_line_name",bus0="my_bus_0",bus1="my_bus_1",length=34,r=2,x=4)



Import from Pypower
===================

PyPSA supports import from Pypower's ppc dictionary/numpy.array format
version 2.


.. code:: python

    from pypower.api import case30

    ppc = case30()

    network.import_from_pypower_ppc(ppc)


Export to CSV
=============
