<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Import and Export

PyPSA can handle several different data formats, such as CSV, netCDF and HDF5
files. It is also possible to build a [`pypsa.Network`][] within a Python
script. There is limited functionality to import from Pypower and Pandapower.
Files can be imported from and exported to local files or cloud storage
providers.

## Adding, Removing & Merging

Networks can be built step-by-step for each component by calling 
[`n.add()`][pypsa.network.transform.NetworkTransformMixin.remove] to **add** a single or multiple components.

``` py
n.add("Bus", "my_bus")
n.add("Generator", "gen_1", bus="my_bus", p_nom=100, marginal_cost=100)
```

``` py
n.add("Load", ["load_1", "load_2"], bus="my_bus", p_set=[10, 20])
```

Components can be **removed** with [`n.remove()`][pypsa.network.transform.NetworkTransformMixin.add].

``` py
n.remove("Load", ["load_1", "load_2"])
n.remove("Generator", "my_generator")
```

Two networks with a disjunct set of component indices can be **merged** with [`n1.merge(n2)`][pypsa.network.transform.NetworkTransformMixin.merge]

## CSV Files

Create a folder with CSVs for each component type (e.g. `generators.csv`), then a CSV for each time-dependent variable (e.g.
`generators-p_max_pu.csv`). Then run
[`n.import_from_csv_folder()`][pypsa.network.io.NetworkIOMixin.import_from_csv_folder] to **import** the network or simply provide the path in the [`pypsa.Network`][] constructor.

!!! note

    It is not necessary to add every single column, only those where values differ from the defaults listed in [Components](../user-guide/components.md). All empty values/columns are filled with the defaults.

A network can be **exported** as a folder of csv files with [`n.export_to_csv_folder()`][pypsa.network.io.NetworkIOMixin.export_to_csv_folder].

``` py
n.export_to_csv_folder("foo/bar")
n_import = pypsa.Network("foo/bar")
```

## Excel

To **import** a network from an Excel file, run [`n.import_from_excel()`][pypsa.network.io.NetworkIOMixin.import_from_excel] or simply provide the path in the [`pypsa.Network`][] constructor. To **export** a network to an Excel file, run [`n.export_to_excel()`][pypsa.network.io.NetworkIOMixin.export_to_excel]. Before using this import functionality, ensure that you install `pypsa[excel]` in your environment using `pip install pypsa[excel]`.

``` py
n.export_to_excel("foo/bar.xlsx")
n_import = pypsa.Network("foo/bar.xlsx")
```

To **create** an Excel file compatible with PyPSA, structure the file with separate worksheets for each component type and their associated time-series data.

For static component data, name each worksheet using the exact component name as it appears in PyPSA (e.g., `generators`, `buses`, `lines`, `storage_units`). For time-series data, use the format `<component>-<attribute>` (e.g., `generators-p_max_pu`, `loads-p_set`, `storage_units-inflow`). Component and attribute names are case-sensitive and must match PyPSA conventions.

The I/O functionality will automatically ignore any worksheets that do not follow this naming pattern, allowing you to include documentation, notes, or auxiliary data in separate tabs without interfering with the import/export process.

The snapshots worksheet must contain the time-series index using an appropriate datetime format. Other time-series worksheets should have the same number of rows, with the index column either numbered sequentially (e.g., 1-8760 for hourly annual data) or using the same datetime format as the snapshots. 

!!! note

    1. To maintain data integrity and avoid conflicts, it is recommended to use the data validation feature in Excel: define your buses in the `Bus` worksheet and apply data validation to bus reference columns in component worksheets (e.g., `bus0`, `bus1` fields in the `Line` worksheet) and similarly apply validation for carriers and time-series dataset references where applicable.

    2. It is not necessary to add every single column, only those where values differ from the defaults. All empty values and columns are filled with the defaults.

!!! warning
    
    Excel is resource-intensive and only appropriate for smaller networks. For larger datasets or production workflows, consider using netCDF files.


## netCDF

netCDF files take up less space than CSV files and are faster to load.

To **export** network and components to a netCDF file run [`n.export_to_netcdf()`][pypsa.network.io.NetworkIOMixin.export_to_netcdf].

To **import** network data from netCDF file run [`n.import_from_netcdf()`][pypsa.network.io.NetworkIOMixin.import_from_netcdf]  or simply provide the path in the [`pypsa.Network`][] constructor.

!!! note

    netCDF is preferred over HDF5 because netCDF is structured more
    cleanly, is easier to use from other programming languages, can limit
    float precision to save space and supports lazy loading.

``` py
n.export_to_netcdf("foo/bar.nc")
n_import = pypsa.Network("foo/bar.nc")
```

## HDF5

To **export** the network to an HDF store, run [`n.export_to_hdf5()`][pypsa.network.io.NetworkIOMixin.export_to_hdf5].

To **import** network data from an HDF5 store, run [`n.import_from_hdf5()`][pypsa.network.io.NetworkIOMixin.import_from_hdf5] or simply provide the path in the [`pypsa.Network`][] constructor.

``` py
n.export_to_hdf5("foo/bar.h5")
n_import = pypsa.Network("foo/bar.h5")
```

## PYPOWER

To **import** a network from the [PYPOWER](https://github.com/rwl/PYPOWER)  ppc dictionary/`numpy.array` format
version 2, run the function [`n.import_from_pypower_ppc()`][pypsa.network.io.NetworkIOMixin.import_from_pypower_ppc].

**Exporting** to PYPOWER is not currently supported.

``` py
from pypower.api import case30
ppc = case30()
n.import_from_pypower_ppc(ppc)
```

## Pandapower

To **import** a network from [pandapower](http://www.pandapower.org/), run the function [`n.import_from_pandapower_net()`][pypsa.network.io.NetworkIOMixin.import_from_pandapower_net].

**Exporting** to [pandapower](http://www.pandapower.org/) is not currently supported.

!!! warning 

    Not all pandapower data fields are supported. For instance,
    three-winding transformers, switches, `in_service` status and tap positions
    of transformers.

``` py
import pandapower.networks as pn
net = pn.create_cigre_network_mv(with_der='all')
n = pypsa.Network()
n.import_from_pandapower_net(net, extra_line_data=True)
```

## Cloud Object Storage

CSV, netCDF and HDF5 files in cloud object storage can be imported and exported
by installing the [`cloudpathlib`](https://cloudpathlib.drivendata.org/stable/)
package. This is available through the `[cloudpath]` optional dependency,
installable via `pip install 'pypsa[cloudpath]'`.

`cloudpathlib` supports [AWS S3](https://aws.amazon.com/s3/) (`s3://`), [Google
Cloud Storage](https://cloud.google.com/storage) (`gs://`) and [Azure Blob
Storage](https://azure.microsoft.com/en-us/products/storage/blobs) (`az://`) as
cloud object storage providers. Users will need to additionally install the
corresponding cloud storage provider client library to interface with the cloud
storage provider via `cloudpathlib` (e.g. `boto3`, `google-cloud-storage` or
`azure-storage-blob`).

``` py
from pypsa import Network
n = Network('examples/ac-dc-meshed/ac-dc-data')
n.export_to_csv_folder('s3://my-s3-bucket/ac-dc-data')
n = Network('s3://my-s3-bucket/ac-dc-data')
n.export_to_netcdf('gs://my-gs-bucket/ac-dc-data.nc')
n = Network('gs://my-gs-bucket/ac-dc-data.nc')
n.export_to_excel('az://my-az-bucket/ac-dc-data.xlsx')
n = Network('az://my-az-bucket/ac-dc-data.xlsx')
```