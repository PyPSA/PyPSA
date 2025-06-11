#################
Import and Export
#################

PyPSA can handle several different data formats, such as CSV, netCDF and HDF5
files. It is also possible to build a ``pypsa.Network`` within a Python script.
There is limited functionality to import from Pypower and Pandapower.

Adding components
=================

Networks can also be built step-by-step for each component by calling 
:py:meth:`pypsa.Network.add` to add a single or multiple components.

To add multiple components whose static attributes are given in a
pandas DataFrame, use :py:meth:`pypsa.Network.import_components_from_dataframe`

To import time-varying information use :py:meth:`pypsa.Network.import_series_from_dataframe`

Removing components
===================

Components can be removed with :py:meth:`pypsa.Network.remove`.

Multiple components can be removed by calling :py:meth:`pypsa.Network.mremove`.

CSV import and export
=====================

Create a folder with CSVs for each component type (e.g. ``generators.csv``,
``storage_units.csv``), then a CSV for each time-dependent variable (e.g.
``generators-p_max_pu.csv, loads-p_set.csv``). Then run
:py:meth:`pypsa.Network.import_from_csv_folder` to **import** the network.

.. note:: It is not necessary to add every single column, only those where values differ from the defaults listed in :doc:`/user-guide/components`. All empty values/columns are filled with the defaults.

A loaded network can be **exported** as a folder of csv files with :py:meth:`pypsa.Network.export_to_csv_folder`.

netCDF import and export
========================

netCDF files take up less space than CSV files and are faster to load.

netCDF is also preferred over HDF5 because netCDF is structured more
cleanly, is easier to use from other programming languages, can limit
float precision to save space and supports lazy loading.

To **export** network and components to a netCDF file run
:py:meth:`pypsa.Network.export_to_netcdf`.

To **import** network data from netCDF file run :py:meth:`pypsa.Network.import_from_netcdf`.


HDF5 import and export
======================

To **export** the network to an HDF store, run
:py:meth:`pypsa.Network.export_to_hdf5`.

To **import** network data from an HDF5 store, run
:py:meth:`pypsa.Network.import_from_hdf5`.


Pypower import
==============

To import a network from Pypower's ppc dictionary/numpy.array format
version 2, run the function :py:meth:`pypsa.Network.import_from_pypower_ppc`.

Pandapower import
=================

.. warning:: Not all pandapower data is supported.

To import a network from `pandapower <http://www.pandapower.org/>`_, run the function :py:meth:`pypsa.Network.import_from_pandapower_net`.


Cloud object storage import and export
======================================
CSV, netCDF and HDF5 files in cloud object storage can be imported and exported by installing the
`cloudpathlib <https://cloudpathlib.drivendata.org/stable/>`_ package. This is available through
the :code:`[cloudpath]` optional dependency, installable via :code:`pip install 'pypsa[cloudpath]'`.

:code:`cloudpathlib` supports AWS S3 (:code:`s3://`), Google Cloud Storage (:code:`gs://`) and
Azure Blob Storage (:code:`az://`) as cloud object storage providers. Users will need to additionally
install the appropriate cloud storage provider client library to interface with the corresponding
cloud storage provider via cloudpathlib (e.g. `boto3`, `google-cloud-storage` or `azure-storage-blob`).

.. code-block:: python

   from pypsa import Network
   n = Network('examples/ac-dc-meshed/ac-dc-data')
   n.export_to_csv_folder('s3://my-s3-bucket/ac-dc-data')
   n = Network('s3://my-s3-bucket/ac-dc-data')
   n.export_to_netcdf('gs://my-gs-bucket/ac-dc-data.nc')
   n = Network('gs://my-gs-bucket/ac-dc-data.nc')
   n.export_to_hdf5('az://my-az-bucket/ac-dc-data.h5')
   n = Network('az://my-az-bucket/ac-dc-data.h5')
