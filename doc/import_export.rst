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

This method is slow for many components; instead use ``madd`` or
``import_components_from_dataframe`` (see below).



.. _madd:

Adding multiple components
==========================

Multiple components can be added by calling

``network.madd(class_name, names, **kwargs)``

where ``class_name`` is for example
``"Line","Bus","Generator","StorageUnit`` and ``names`` is a list of
unique names of the components. Other attributes can also be specified
as scalars, lists, arrays, pandas Series or pandas DataFrames.

Make sure when adding static attributes as pandas Series that they are
indexed by names. Make sure when adding time-varying attributes as
pandas DataFrames that their index is a superset of network.snapshots
and their columns are a subset of names.

.. code:: python

    import pandas as pd, numpy as np

    buses = range(13)
    snapshots = range(7)

    n = pypsa.Network()

    n.set_snapshots(snapshots)

    n.madd("Bus",
           buses)

    #add load as numpy array
    n.madd("Load",
           n.buses.index + " load",
           bus=buses,
	   p_set=np.random.rand(len(snapshots),len(buses)))

    #add wind availability as pandas DataFrame
    wind = pd.DataFrame(np.random.rand(len(snapshots),len(buses)),
                        index=n.snapshots,
			columns=buses)
    #use a suffix to avoid boilerplate to rename everything
    n.madd("Generator",
           buses,
	   suffix=' wind',
	   bus=buses,
	   p_nom_extendable=True,
	   capital_cost=1e5,
	   p_max_pu=wind)

Any attributes which are not specified will be given the default value from :doc:`components`.


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

Export to netCDF
================

Export network and components to a netCDF file.

netCDF files take up less space than CSV files and are faster to load.

netCDF is also preferred over HDF5 because netCDF is structured more
cleanly, is easier to use from other programming languages, can limit
float precision to save space and supports lazy loading.

Both static and series attributes of components are exported, but only
if they have non-default values.

``network.export_to_netcdf(file.nc)``

If ``file.nc`` does not already exist, it is created.


Import from netCDF
==================

Import network data from netCDF file ``file.nc``:

``network.import_from_netcdf(file.nc)``


Export to HDF5
==============

Export network and components to an HDF store.

NB: netCDF is preferred over HDF5 because netCDF is structured more
cleanly, is easier to use from other programming languages, can limit
float precision to save space and supports lazy loading.


Both static and series attributes of components are exported, but only
if they have non-default values.

``network.export_to_hdf5(path)``


If ``path`` does not already exist, it is created.



Import from HDF5
================

Import network data from HDF5 store at ``path``:

``network.import_from_hdf5(path)``


Import from Pypower
===================

PyPSA supports import from Pypower's ppc dictionary/numpy.array format
version 2.


.. code:: python

    from pypower.api import case30

    ppc = case30()

    network.import_from_pypower_ppc(ppc)
