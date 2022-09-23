######################
Data Import and Export
######################

PyPSA is intended to be data format agnostic, but given the reliance
internally on pandas DataFrames, it is natural to use
comma-separated-variable (CSV) files.

Import from folder of CSV files
===============================

Create a folder with CSVs for each component type
(e.g. ``generators.csv, storage_units.csv``), then a CSV for each
time-dependent variable (e.g. ``generators-p_max_pu.csv,
loads-p_set.csv``). Then run :py:meth:`pypsa.Network.import_from_csv_folder`

.. note:: It is not necessary to add every single column, only those where values differ from the defaults listed in :doc:`components`. All empty values/columns are filled with the defaults.

.. _export-csv:

Export to folder of CSV files
=============================

The network can be exported as a folder of csv files with :py:meth:`pypsa.Network.export_to_csv_folder`.


Adding and removing components one-by-one
==========================================

Networks can also be built step-by-step for each component by calling :py:meth:`pypsa.Network.add`. Likewise, components can also be removed with :py:meth:`pypsa.Network.remove`.

.. _madd:

Adding and removing multiple components
========================================

Multiple components can be added by calling :py:meth:`pypsa.Network.madd`. Multiple components can be removed by calling :py:meth:`pypsa.Network.mremove`.


Adding components using pandas DataFrames
=========================================

To add multiple components whose static attributes are given in a
pandas DataFrame, use :py:meth:`pypsa.Network.import_components_from_dataframe`

To import time-varying information use :py:meth:`pypsa.Network.import_series_from_dataframe`


Export to netCDF
================

netCDF files take up less space than CSV files and are faster to load.

netCDF is also preferred over HDF5 because netCDF is structured more
cleanly, is easier to use from other programming languages, can limit
float precision to save space and supports lazy loading.

To export network and components to a netCDF file run
:py:meth:`pypsa.Network.export_to_netcdf`.


Import from netCDF
==================

To import network data from netCDF file run :py:meth:`pypsa.Network.import_from_netcdf`.


Export to HDF5
==============

.. note:: netCDF is preferred over HDF5 because netCDF is structured more cleanly, is easier to use from other programming languages, can limit float precision to save space and supports lazy loading.

To export network and components to an HDF store run :py:meth:`pypsa.Network.export_to_hdf5`.


Import from HDF5
================

To import network data from HDF5 store at ``path`` run
:py:meth:`pypsa.Network.import_from_hdf5`.


Import from Pypower
===================

PyPSA supports import from Pypower's ppc dictionary/numpy.array format
version 2, see :py:meth:`pypsa.Network.import_from_pypower_ppc`.

Import from Pandapower
======================

.. warning:: Importing from pandapower is still in beta; not all pandapower data is supported.

PyPSA supports import from `pandapower <http://www.pandapower.org/>`_ using the function :py:meth:`pypsa.Network.import_from_pandapower_net`.
