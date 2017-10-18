######################
Data Import and Export
######################

PyPSA is intended to be data format agnostic, but given the reliance
internally on pandas DataFrames, it is natural to use
comma-separated-variable (CSV) files.

The import-export functionality can be found in pypsa/io.py.


Import from folder of CSV files
===============================

Create a folder with CSVs for each component type
(e.g. ``generators.csv, storage_units.csv``), then a CSV for each
time-dependent variable (e.g. ``generators-p_max_pu.csv,
loads-p_set.csv``).

Then run

``network.import_from_csv_folder(csv_folder_name)``

See the :doc:`examples` in pypsa/examples/.

Note that is is NOT necessary to add every single column, only those where values differ from the defaults listed in :doc:`components`. All empty values/columns are filled with the defaults.


.. _export-csv:

Export to folder of CSV files
=============================

The network can be exported as a folder of csv files:

``network.export_to_csv_folder(csv_folder_name)``

If the folder does not exist it will be created.

All non-default static and series attributes of all components will be
exported.

Static attributes are exported in one CSV file per component,
e.g. ``generators.csv``.

Series attributes are exported in one CSV file per component per
attribute, e.g. ``generators-p_set.csv``.


Adding components one-by-one
============================

Networks can also be built "manually" in code by calling

``network.add(class_name, name, **kwargs)``

where ``class_name`` is for example
``"Line","Bus","Generator","StorageUnit`` and ``name`` is the unique
name of the component. Other attributes can also be specified:

.. code:: python

    network.add("Bus","my_bus_0")
    network.add("Bus","my_bus_1",v_nom=380)
    network.add("Line","my_line_name",bus0="my_bus_0",bus1="my_bus_1",length=34,r=2,x=4)

Any attributes which are not specified will be given the default value from :doc:`components`.

This method is slow for many components; instead use pandas DataFrames (see below)

Adding components using pandas DataFrames
=========================================

To add multiple components whose static attributes are given in a
pandas DataFrame, use

``network.import_components_from_dataframe(dataframe, cls_name)``

``dataframe`` is a pandas DataFrame whose index is the names of the
components and whose columns are the non-default
attributes. ``cls_name`` is the component name,
e.g. ``"Line","Bus","Generator","StorageUnit``. If columns are missing
then defaults are used. If extra columns are added, these are left in
the resulting component DataFrame.

.. code:: python

    import pandas as pd

    buses = ['Berlin', 'Frankfurt', 'Munich', 'Hamburg']

    network.import_components_from_dataframe(pd.DataFrame({"v_nom" : 380,
                                                           "control" : 'PV'},
							  index=buses),
					     "Bus")

    network.import_components_from_dataframe(pd.DataFrame({"carrier" : "solar",
                                                           "bus" : buses,
							   "p_nom_extendable" : True,
							   "capital_cost" : 6e4},
							  index=[b+" PV" for b in buses]),
					     "Generator")

To import time-varying information use

``network.import_series_from_dataframe(dataframe, cls_name, attr)``

``cls_name`` is the component name, ``attr`` is the time-varying
attribute and ``dataframe`` is a pandas DataFrame whose index is
``network.snapshots`` and whose columns are a subset of the relevant
components.

Following the previous example:

.. code:: python

    import numpy as np

    network.set_snapshots(range(10))
    network.import_series_from_dataframe(pd.DataFrame(np.random.rand(10,4),
                                                      columns=network.generators.index,
						      index=range(10)),
				         "Generator",
					 "p_max_pu")

Export to HDF5
==============

Export network and components to an HDF store.

Both static and series attributes of components are exported, but only
if they have non-default values.

If path does not already exist, it is created.


``network.export_to_hdf5(filename)``

Import from HDF5
================

Import network data from HDF5 store at `path`:

``network.import_from_hdf5(path)``


Import from Pypower
===================

PyPSA supports import from Pypower's ppc dictionary/numpy.array format
version 2.


.. code:: python

    from pypower.api import case30

    ppc = case30()

    network.import_from_pypower_ppc(ppc)
