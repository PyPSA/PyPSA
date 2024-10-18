"""
Functions for importing and exporting data.
"""

from __future__ import annotations

import json
import logging
import math
import os
from abc import abstractmethod
from collections.abc import Collection, Iterable, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.request import urlretrieve

from pypsa.utils import deprecated_common_kwargs

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import validators
import xarray as xr
from deprecation import deprecated
from pyproj import CRS

from pypsa.descriptors import update_linkports_component_attrs

if TYPE_CHECKING:
    from typing import TracebackType  # type: ignore

    from pandapower.auxiliary import pandapowerNet

    from pypsa import Network


logger = logging.getLogger(__name__)

# for the writable data directory follow the XDG guidelines
# https://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
_writable_dir = Path(os.path.expanduser("~")) / ".local" / "share"
_data_dir = (
    Path(os.environ.get("XDG_DATA_HOME", os.environ.get("APPDATA", _writable_dir)))
    / "pypsa-networks"
)

_data_dir.mkdir(exist_ok=True, parents=True)


def _retrieve_from_url(path: str) -> str:
    local_path = _data_dir / os.path.basename(path)
    logger.info(f"Retrieving network data from {path}")
    urlretrieve(path, local_path)
    return str(local_path)


# TODO: Restructure abc inheritance


TImpExper = TypeVar("TImpExper", bound="ImpExper")


class ImpExper:
    ds: Any = None

    def __enter__(self: TImpExper) -> TImpExper:
        if self.ds is not None:
            self.ds = self.ds.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type,
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> None:
        if exc_type is None:
            self.finish()

        if self.ds is not None:
            self.ds.__exit__(exc_type, exc_val, exc_tb)

    def finish(self) -> None:
        pass


class Exporter(ImpExper):
    def remove_static(self, list_name: str) -> None:
        pass

    def remove_series(self, list_name: str, attr: str) -> None:
        pass

    @abstractmethod
    def save_attributes(self, attrs: dict) -> None:
        pass

    @abstractmethod
    def save_meta(self, meta: dict) -> None:
        pass

    @abstractmethod
    def save_crs(self, crs: dict) -> None:
        pass

    @abstractmethod
    def save_snapshots(self, snapshots: Sequence) -> None:
        pass

    @abstractmethod
    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        pass

    @abstractmethod
    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        pass


class Importer(ImpExper):
    pass


class ImporterCSV(Importer):
    def __init__(self, csv_folder_name: str | Path, encoding: str | None) -> None:
        self.csv_folder_name = Path(csv_folder_name)
        self.encoding = encoding

        if not self.csv_folder_name.is_dir():
            msg = f"Directory {csv_folder_name} does not exist."
            raise FileNotFoundError(msg)

    def get_attributes(self) -> dict | None:
        fn = self.csv_folder_name.joinpath("network.csv")
        if not fn.is_file():
            return None
        return dict(pd.read_csv(fn, encoding=self.encoding).iloc[0])

    def get_meta(self) -> dict:
        fn = self.csv_folder_name.joinpath("meta.json")
        return {} if not fn.is_file() else json.loads(fn.open().read())

    def get_crs(self) -> dict:
        fn = self.csv_folder_name.joinpath("crs.json")
        return {} if not fn.is_file() else json.loads(fn.open().read())

    def get_snapshots(self) -> pd.Index:
        fn = self.csv_folder_name.joinpath("snapshots.csv")
        if not fn.is_file():
            return None
        df = pd.read_csv(fn, index_col=0, encoding=self.encoding, parse_dates=True)
        # backwards-compatibility: level "snapshot" was rename to "timestep"
        if "snapshot" in df:
            df["snapshot"] = pd.to_datetime(df.snapshot)
        if "timestep" in df:
            df["timestep"] = pd.to_datetime(df.timestep)
        return df

    def get_investment_periods(self) -> pd.Series:
        fn = self.csv_folder_name.joinpath("investment_periods.csv")
        if not fn.is_file():
            return None
        return pd.read_csv(fn, index_col=0, encoding=self.encoding)

    def get_static(self, list_name: str) -> pd.DataFrame:
        fn = self.csv_folder_name.joinpath(list_name + ".csv")
        return (
            pd.read_csv(fn, index_col=0, encoding=self.encoding)
            if fn.is_file()
            else None
        )

    def get_series(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        for fn in self.csv_folder_name.iterdir():
            if fn.name.startswith(list_name + "-") and fn.name.endswith(".csv"):
                attr = fn.name[len(list_name) + 1 : -4]
                df = pd.read_csv(
                    self.csv_folder_name.joinpath(fn.name),
                    index_col=0,
                    encoding=self.encoding,
                    parse_dates=True,
                )
                yield attr, df


class ExporterCSV(Exporter):
    def __init__(self, csv_folder_name: Path | str, encoding: str | None) -> None:
        self.csv_folder_name = Path(csv_folder_name)
        self.encoding = encoding

        # make sure directory exists
        if not self.csv_folder_name.is_dir():
            logger.warning(f"Directory {csv_folder_name} does not exist, creating it")
            self.csv_folder_name.mkdir()

    def save_attributes(self, attrs: dict) -> None:
        name = attrs.pop("name")
        df = pd.DataFrame(attrs, index=pd.Index([name], name="name"))
        fn = self.csv_folder_name.joinpath("network.csv")
        with fn.open("w"):
            df.to_csv(fn, encoding=self.encoding)

    def save_meta(self, meta: dict) -> None:
        fn = self.csv_folder_name.joinpath("meta.json")
        fn.open("w").write(json.dumps(meta))

    def save_crs(self, crs: dict) -> None:
        fn = self.csv_folder_name.joinpath("crs.json")
        fn.open("w").write(json.dumps(crs))

    def save_snapshots(self, snapshots: pd.Index) -> None:
        fn = self.csv_folder_name.joinpath("snapshots.csv")
        with fn.open("w"):
            snapshots.to_csv(fn, encoding=self.encoding)

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        fn = self.csv_folder_name.joinpath("investment_periods.csv")
        with fn.open("w"):
            investment_periods.to_csv(fn, encoding=self.encoding)

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        fn = self.csv_folder_name.joinpath(list_name + ".csv")
        with fn.open("w"):
            df.to_csv(fn, encoding=self.encoding)

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        fn = self.csv_folder_name.joinpath(list_name + "-" + attr + ".csv")
        with fn.open("w"):
            df.to_csv(fn, encoding=self.encoding)

    def remove_static(self, list_name: str) -> None:
        if fns := list(self.csv_folder_name.joinpath(list_name).glob("*.csv")):
            for fn in fns:
                fn.unlink()
            logger.warning(f'Stale csv file(s) {", ".join(fns)} removed')

    def remove_series(self, list_name: str, attr: str) -> None:
        fn = self.csv_folder_name.joinpath(list_name + "-" + attr + ".csv")
        if fn.exists():
            fn.unlink()


class ImporterHDF5(Importer):
    def __init__(self, path: str | pd.HDFStore) -> None:
        self.path = path
        self.ds: pd.HDFStore
        if isinstance(path, (str, Path)):
            if validators.url(str(path)):
                path = _retrieve_from_url(str(path))
            self.ds = pd.HDFStore(path, mode="r")
        self.index: dict = {}

    def get_attributes(self) -> dict:
        return dict(self.ds["/network"].reset_index().iloc[0])

    def get_meta(self) -> dict:
        return json.loads(self.ds["/meta"][0] if "/meta" in self.ds else "{}")

    def get_crs(self) -> dict:
        return json.loads(self.ds["/crs"][0] if "/crs" in self.ds else "{}")

    def get_snapshots(self) -> pd.Series:
        return self.ds["/snapshots"] if "/snapshots" in self.ds else None

    def get_investment_periods(self) -> pd.Series:
        return (
            self.ds["/investment_periods"] if "/investment_periods" in self.ds else None
        )

    def get_static(self, list_name: str) -> pd.DataFrame:
        if "/" + list_name not in self.ds:
            return None

        if self.pypsa_version is None or self.pypsa_version < [0, 13, 1]:  # type: ignore
            df = self.ds["/" + list_name]
        else:
            df = self.ds["/" + list_name].set_index("name")

        self.index[list_name] = df.index
        return df

    def get_series(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        for tab in self.ds:
            if tab.startswith("/" + list_name + "_t/"):
                attr = tab[len("/" + list_name + "_t/") :]
                df = self.ds[tab]
                df.columns = self.index[list_name][df.columns]
                yield attr, df


class ExporterHDF5(Exporter):
    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        path = Path(path)
        self._hdf5_handle = path.open("w")
        self.ds = pd.HDFStore(path, mode="w", **kwargs)
        self.index: dict = {}

    def __exit__(
        self, exc_type: type, exc_val: BaseException, exc_tb: TracebackType
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)

    def save_attributes(self, attrs: dict) -> None:
        name = attrs.pop("name")
        self.ds.put(
            "/network",
            pd.DataFrame(attrs, index=pd.Index([name], name="name")),
            format="table",
            index=False,
        )

    def save_meta(self, meta: dict) -> None:
        self.ds.put("/meta", pd.Series(json.dumps(meta)))

    def save_crs(self, crs: dict) -> None:
        self.ds.put("/crs", pd.Series(json.dumps(crs)))

    def save_snapshots(self, snapshots: Sequence) -> None:
        self.ds.put("/snapshots", snapshots, format="table", index=False)

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        self.ds.put(
            "/investment_periods",
            investment_periods,
            format="table",
            index=False,
        )

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        df = df.rename_axis(index="name")
        self.index[list_name] = df.index
        df = df.reset_index()
        self.ds.put("/" + list_name, df, format="table", index=False)

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        df = df.set_axis(self.index[list_name].get_indexer(df.columns), axis="columns")
        self.ds.put("/" + list_name + "_t/" + attr, df, format="table", index=False)

    def finish(self) -> None:
        self._hdf5_handle.close()


class ImporterNetCDF(Importer):
    ds: xr.Dataset

    def __init__(self, path: str | Path | xr.Dataset) -> None:
        self.path = path
        if isinstance(path, (str, Path)):
            if validators.url(str(path)):
                path = _retrieve_from_url(str(path))
            self.ds = xr.open_dataset(path)
        else:
            self.ds = path

    def __enter__(self) -> ImporterNetCDF:
        if isinstance(self.path, (str, Path)):
            super().__init__()
        return self

    def __exit__(
        self,
        exc_type: type,
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> None:
        if isinstance(self.path, (str, Path)):
            super().__exit__(exc_type, exc_val, exc_tb)

    def get_attributes(self) -> dict:
        return {
            attr[len("network_") :]: val
            for attr, val in self.ds.attrs.items()
            if attr.startswith("network_")
        }

    def get_meta(self) -> dict:
        return json.loads(self.ds.attrs.get("meta", "{}"))

    def get_crs(self) -> dict:
        return json.loads(self.ds.attrs.get("crs", "{}"))

    def get_snapshots(self) -> pd.DataFrame:
        return self.get_static("snapshots", "snapshots")

    def get_investment_periods(self) -> pd.DataFrame:
        return self.get_static("investment_periods", "investment_periods")

    def get_static(self, list_name: str, index_name: str | None = None) -> pd.DataFrame:
        t = list_name + "_"
        i = len(t)
        if index_name is None:
            index_name = list_name + "_i"
        if index_name not in self.ds.coords:
            return None
        index = self.ds.coords[index_name].to_index().rename("name")
        df = pd.DataFrame(index=index)
        for attr in self.ds.data_vars.keys():
            if attr.startswith(t) and attr[i : i + 2] != "t_":
                df[attr[i:]] = self.ds[attr].to_pandas()
        return df

    def get_series(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        t = list_name + "_t_"
        for attr in self.ds.data_vars.keys():
            if attr.startswith(t):
                df = self.ds[attr].to_pandas()
                df.index.name = "name"
                df.columns.name = "name"
                yield attr[len(t) :], df


class ExporterNetCDF(Exporter):
    def __init__(
        self,
        path: str | None,
        compression: dict | None = {"zlib": True, "complevel": 4},
        float32: bool = False,
    ) -> None:
        self.path = path
        self.compression = compression
        self.float32 = float32
        self.ds = xr.Dataset()

    def save_attributes(self, attrs: dict) -> None:
        self.ds.attrs.update(("network_" + attr, val) for attr, val in attrs.items())

    def save_meta(self, meta: dict) -> None:
        self.ds.attrs["meta"] = json.dumps(meta)

    def save_crs(self, crs: dict) -> None:
        self.ds.attrs["crs"] = json.dumps(crs)

    def save_snapshots(self, snapshots: pd.Index) -> None:
        snapshots = snapshots.rename_axis(index="snapshots")
        for attr in snapshots.columns:
            self.ds["snapshots_" + attr] = snapshots[attr]

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        investment_periods = investment_periods.rename_axis(index="investment_periods")
        for attr in investment_periods.columns:
            self.ds["investment_periods_" + attr] = investment_periods[attr]

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        df = df.rename_axis(index=list_name + "_i")
        self.ds[list_name + "_i"] = df.index
        for attr in df.columns:
            self.ds[list_name + "_" + attr] = df[attr]

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        df = df.rename_axis(index="snapshots", columns=list_name + "_t_" + attr + "_i")
        self.ds[list_name + "_t_" + attr] = df

    def set_compression_encoding(self) -> None:
        logger.debug(f"Setting compression encodings: {self.compression}")
        for v in self.ds.data_vars:
            if self.ds[v].dtype.kind not in ["U", "O"]:
                self.ds[v].encoding.update(self.compression)

    def typecast_float32(self) -> None:
        logger.debug("Typecasting float64 to float32.")
        for v in self.ds.data_vars:
            if self.ds[v].dtype == np.float64:
                self.ds[v] = self.ds[v].astype(np.float32)

    def finish(self) -> None:
        if self.float32:
            self.typecast_float32()
        if self.compression:
            self.set_compression_encoding()
        if self.path is not None:
            _path = Path(self.path)
            with _path.open("w"):
                self.ds.to_netcdf(_path)


@deprecated_common_kwargs
def _export_to_exporter(
    n: Network,
    exporter: Exporter,
    basename: str | None = None,
    export_standard_types: bool = False,
) -> None:
    """
    Export to exporter.

    Both static and series attributes of components are exported, but only
    if they have non-default values.

    Parameters
    ----------
    exporter : Exporter
        Initialized exporter instance
    basename : str
        Basename, used for logging
    export_standard_types : boolean, default False
        If True, then standard types are exported too (upon reimporting you
        should then set "ignore_standard_types" when initialising the netowrk).
    """
    if not basename:
        basename = "<unnamed>"
    # exportable component types
    allowed_types = (float, int, bool, str) + tuple(np.sctypeDict.values())

    # first export network properties
    _attrs = {
        attr: getattr(n, attr)
        for attr in dir(n)
        if (not attr.startswith("__") and isinstance(getattr(n, attr), allowed_types))
    }
    exporter.save_attributes(_attrs)

    crs = {}
    if n.crs is not None:
        crs["_crs"] = n.crs.to_wkt()
    exporter.save_crs(crs)

    exporter.save_meta(n.meta)

    # now export snapshots
    if isinstance(n.snapshot_weightings.index, pd.MultiIndex):
        n.snapshot_weightings.index.rename(["period", "timestep"], inplace=True)
    else:
        n.snapshot_weightings.index.rename("snapshot", inplace=True)
    snapshots = n.snapshot_weightings.reset_index()
    exporter.save_snapshots(snapshots)

    # export investment period weightings
    investment_periods = n.investment_period_weightings
    exporter.save_investment_periods(investment_periods)

    exported_components = []
    for component in n.all_components - {"SubNetwork"}:
        list_name = n.components[component]["list_name"]
        attrs = n.components[component]["attrs"]

        static = n.static(component)
        dynamic = n.dynamic(component)

        if component == "Shape":
            static = pd.DataFrame(static).assign(geometry=static["geometry"].to_wkt())

        if not export_standard_types and component in n.standard_type_components:
            static = static.drop(n.components[component]["standard_types"].index)

        # first do static attributes
        static = static.rename_axis(index="name")
        if static.empty:
            exporter.remove_static(list_name)
            continue

        col_export = []
        for col in static.columns:
            # do not export derived attributes
            if col in ["sub_network", "r_pu", "x_pu", "g_pu", "b_pu"]:
                continue
            if (
                col in attrs.index
                and pd.isnull(attrs.at[col, "default"])
                and pd.isnull(static[col]).all()
            ):
                continue
            if (
                col in attrs.index
                and static[col].dtype == attrs.at[col, "dtype"]
                and (static[col] == attrs.at[col, "default"]).all()
            ):
                continue

            col_export.append(col)

        exporter.save_static(list_name, static[col_export])

        # now do varying attributes
        for attr in dynamic:
            if attr not in attrs.index:
                col_export = dynamic[attr].columns
            else:
                default = attrs.at[attr, "default"]

                if pd.isnull(default):
                    col_export = dynamic[attr].columns[
                        (~pd.isnull(dynamic[attr])).any()
                    ]
                else:
                    col_export = dynamic[attr].columns[(dynamic[attr] != default).any()]

            if len(col_export) > 0:
                static = dynamic[attr].reset_index()[col_export]
                exporter.save_series(list_name, attr, static)
            else:
                exporter.remove_series(list_name, attr)

        exported_components.append(list_name)

    logger.info(
        "Exported network '%s' contains: %s", basename, ", ".join(exported_components)
    )


@deprecated_common_kwargs
def import_from_csv_folder(
    n: Network,
    csv_folder_name: str | Path,
    encoding: str | None = None,
    skip_time: bool = False,
) -> None:
    """
    Import network data from CSVs in a folder.

    The CSVs must follow the standard form, see ``pypsa/examples``.

    Parameters
    ----------
    csv_folder_name : string
        Name of folder
    encoding : str, default None
        Encoding to use for UTF when reading (ex. 'utf-8'). `List of Python
        standard encodings
        <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
    skip_time : bool, default False
        Skip reading in time dependent attributes

    Examples
    --------
    >>> n.import_from_csv_folder(csv_folder_name)
    """
    basename = Path(csv_folder_name).name
    with ImporterCSV(csv_folder_name, encoding=encoding) as importer:
        _import_from_importer(n, importer, basename=basename, skip_time=skip_time)


@deprecated_common_kwargs
def export_to_csv_folder(
    n: Network,
    csv_folder_name: str,
    encoding: str | None = None,
    export_standard_types: bool = False,
) -> None:
    """
    Export network and components to a folder of CSVs.

    Both static and series attributes of all components are exported, but only
    if they have non-default values.

    If ``csv_folder_name`` does not already exist, it is created.

    ``csv_folder_name`` may also be a cloud object storage URI if cloudpathlib is installed.

    Static attributes are exported in one CSV file per component,
    e.g. ``generators.csv``.

    Series attributes are exported in one CSV file per component per
    attribute, e.g. ``generators-p_set.csv``.

    Parameters
    ----------
    csv_folder_name : string
        Name of folder to which to export.
    encoding : str, default None
        Encoding to use for UTF when reading (ex. 'utf-8'). `List of Python
        standard encodings
        <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
    export_standard_types : boolean, default False
        If True, then standard types are exported too (upon reimporting you
        should then set "ignore_standard_types" when initialising the network).

    Examples
    --------
    >>> n.export_to_csv_folder(csv_folder_name)
    """

    basename = os.path.basename(csv_folder_name)
    with ExporterCSV(csv_folder_name=csv_folder_name, encoding=encoding) as exporter:
        _export_to_exporter(
            n,
            exporter,
            basename=basename,
            export_standard_types=export_standard_types,
        )


@deprecated_common_kwargs
def import_from_hdf5(n: Network, path: str | Path, skip_time: bool = False) -> None:
    """
    Import network data from HDF5 store at `path`.

    Parameters
    ----------
    path : string, Path
        Name of HDF5 store. The string could be a URL.
    skip_time : bool, default False
        Skip reading in time dependent attributes
    """
    basename = Path(path).name

    with ImporterHDF5(path) as importer:
        _import_from_importer(n, importer, basename=basename, skip_time=skip_time)


@deprecated_common_kwargs
def export_to_hdf5(
    n: Network,
    path: Path | str,
    export_standard_types: bool = False,
    **kwargs: Any,
) -> None:
    """
    Export network and components to an HDF store.

    Both static and series attributes of components are exported, but only
    if they have non-default values.

    If path does not already exist, it is created.

    ``path`` may also be a cloud object storage URI if cloudpathlib is installed.

    Parameters
    ----------
    path : string
        Name of hdf5 file to which to export (if it exists, it is overwritten)
    export_standard_types : boolean, default False
        If True, then standard types are exported too (upon reimporting you
        should then set "ignore_standard_types" when initialising the network).
    **kwargs
        Extra arguments for pd.HDFStore to specify f.i. compression
        (default: complevel=4)

    Examples
    --------
    >>> n.export_to_hdf5(filename)
    """
    kwargs.setdefault("complevel", 4)

    basename = os.path.basename(path)
    with ExporterHDF5(path, **kwargs) as exporter:
        _export_to_exporter(
            n,
            exporter,
            basename=basename,
            export_standard_types=export_standard_types,
        )


@deprecated_common_kwargs
def import_from_netcdf(
    n: Network, path: str | Path | xr.Dataset, skip_time: bool = False
) -> None:
    """
    Import network data from netCDF file or xarray Dataset at `path`.

    ``path`` may also be a cloud object storage URI if cloudpathlib is installed.

    Parameters
    ----------
    path : string|xr.Dataset
        Path to netCDF dataset or instance of xarray Dataset.
        The string could be a URL.
    skip_time : bool, default False
        Skip reading in time dependent attributes
    """
    basename = "" if isinstance(path, xr.Dataset) else Path(path).name
    with ImporterNetCDF(path=path) as importer:
        _import_from_importer(n, importer, basename=basename, skip_time=skip_time)


@deprecated_common_kwargs
def export_to_netcdf(
    n: Network,
    path: str | None = None,
    export_standard_types: bool = False,
    compression: dict | None = None,
    float32: bool = False,
) -> xr.Dataset:
    """
    Export network and components to a netCDF file.

    Both static and series attributes of components are exported, but only
    if they have non-default values.

    If path does not already exist, it is created.

    If no path is passed, no file is exported, but the xarray.Dataset
    is still returned.

    Be aware that this cannot export boolean attributes on the Network
    class, e.g. n.my_bool = False is not supported by netCDF.

    Parameters
    ----------
    path : string|None
        Name of netCDF file to which to export (if it exists, it is overwritten);
        if None is passed, no file is exported.
    export_standard_types : boolean, default False
        If True, then standard types are exported too (upon reimporting you
        should then set "ignore_standard_types" when initialising the network).
    compression : dict|None
        Compression level to use for all features which are being prepared.
        The compression is handled via xarray.Dataset.to_netcdf(...). For details see:
        https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html
        An example compression directive is ``{'zlib': True, 'complevel': 4}``.
        The default is None which disables compression.
    float32 : boolean, default False
        If True, typecasts values to float32.

    Returns
    -------
    ds : xarray.Dataset

    Examples
    --------
    >>> n.export_to_netcdf("my_file.nc")
    """
    basename = os.path.basename(path) if path is not None else None
    with ExporterNetCDF(path, compression, float32) as exporter:
        _export_to_exporter(
            n,
            exporter,
            basename=basename,
            export_standard_types=export_standard_types,
        )
        return exporter.ds


@deprecated_common_kwargs
def _import_from_importer(
    n: Network, importer: Any, basename: str, skip_time: bool = False
) -> None:
    """
    Import network data from importer.

    Parameters
    ----------
    skip_time : bool
        Skip importing time
    """
    attrs = importer.get_attributes()
    n.meta = importer.get_meta()
    crs = importer.get_crs()
    crs = crs.pop("_crs", None)
    if crs is not None:
        crs = CRS.from_wkt(crs)
        n._crs = crs

    current_pypsa_version = [int(s) for s in n.pypsa_version.split(".")]
    pypsa_version = None

    if attrs is not None:
        n.name = attrs.pop("name")

        try:
            pypsa_version = [int(s) for s in attrs.pop("pypsa_version").split(".")]
        except KeyError:
            pypsa_version = None

        for attr, val in attrs.items():
            setattr(n, attr, val)

    ## https://docs.python.org/3/tutorial/datastructures.html#comparing-sequences-and-other-types
    if pypsa_version is None or pypsa_version < current_pypsa_version:
        pypsa_version_str = (
            ".".join(map(str, pypsa_version)) if pypsa_version is not None else "?"
        )
        current_pypsa_version_str = ".".join(map(str, current_pypsa_version))
        msg = (
            f"Importing network from PyPSA version v{pypsa_version_str} while "
            f"current version is v{current_pypsa_version_str}. Read the "
            "release notes at https://pypsa.readthedocs.io/en/latest/release_notes.html "
            "to prepare your network for import."
        )
        logger.warning(msg)

    if pypsa_version is None or pypsa_version < [0, 18, 0]:
        n._multi_invest = 0

    importer.pypsa_version = pypsa_version
    importer.current_pypsa_version = current_pypsa_version

    # if there is snapshots.csv, read in snapshot data
    df = importer.get_snapshots()

    if df is not None:
        if snapshot_levels := {"period", "timestep", "snapshot"}.intersection(
            df.columns
        ):
            df.set_index(sorted(snapshot_levels), inplace=True)
        n.set_snapshots(df.index)

        cols = ["objective", "generators", "stores"]
        if not df.columns.intersection(cols).empty:
            n.snapshot_weightings = df.reindex(index=n.snapshots, columns=cols)
        elif "weightings" in df.columns:
            n.snapshot_weightings = df["weightings"].reindex(n.snapshots)

        n.set_snapshots(df.index)

    # read in investment period weightings
    periods = importer.get_investment_periods()

    if periods is not None:
        n._investment_periods = periods.index

        n._investment_period_weightings = periods.reindex(n.investment_periods)

    imported_components = []

    # now read in other components; make sure buses and carriers come first
    for component in ["Bus", "Carrier"] + sorted(
        n.all_components - {"Bus", "Carrier", "SubNetwork"}
    ):
        list_name = n.components[component]["list_name"]

        df = importer.get_static(list_name)
        if df is None:
            if component == "Bus":
                logger.error("Error, no buses found")
                return
            continue

        if component == "Link":
            update_linkports_component_attrs(n, where=df)

        n.add(component, df.index, **df)

        if not skip_time:
            for attr, df in importer.get_series(list_name):
                df.set_index(n.snapshots, inplace=True)
                _import_series_from_df(n, df, component, attr)

        logger.debug(getattr(n, list_name))

        imported_components.append(list_name)

    logger.info(
        f"Imported network {str(basename or n.name or '<unnamed>')} "
        f"has {', '.join(imported_components)}"
    )


def _sort_attrs(df: pd.DataFrame, attrs_list: list[str], axis: int) -> pd.DataFrame:
    """
    Sort axis of DataFrame according to the order of attrs_list.

    Attributes not in attrs_list are appended at the end. Attributes in the list but
    not in the DataFrame are ignored.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to sort
    attrs_list : list
        List of attributes to sort by
    axis : int
        Axis to sort (0 for index, 1 for columns)

    Returns
    -------
    pandas.DataFrame
    """

    df_cols_set = set(df.columns if axis == 1 else df.index)

    existing_cols = [col for col in attrs_list if col in df_cols_set]
    remaining_cols = list(df_cols_set - set(attrs_list))

    return df.reindex(existing_cols + remaining_cols, axis=axis)


@deprecated(
    deprecated_in="0.29",
    removed_in="1.0",
    details="Use `n.add` instead. E.g. `n.add(class_name, df.index, **df)`.",
)
def import_components_from_dataframe(
    n: Network, dataframe: pd.DataFrame, cls_name: str
) -> None:
    """
    Import components from a pandas DataFrame.

    This function is deprecated. Use :py:meth`pypsa.Network.add` instead. To get the
    same behavior for importing components from a DataFrame, use
    `n.add(cls_name, df.index, **df)`.

    If columns are missing then defaults are used. If extra columns are added, these
    are left in the resulting component dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A DataFrame whose index is the names of the components and
        whose columns are the non-default attributes.
    cls_name : string
        Name of class of component, e.g. ``"Line", "Bus", "Generator", "StorageUnit"``

    Examples
    --------
    >>> import pandas as pd
    >>> buses = ['Berlin', 'Frankfurt', 'Munich', 'Hamburg']
    >>> n.import_components_from_dataframe(
    ...     pd.DataFrame({"v_nom" : 380, "control" : 'PV'},
    ...                 index=buses),
    ...                 "Bus")
    >>> n.import_components_from_dataframe(
    ...     pd.DataFrame({"carrier" : "solar", "bus" : buses, "p_nom_extendable" : True},
    ...                 index=[b+" PV" for b in buses]),
    ...                 "Generator")

    See Also
    --------
    pypsa.Network.madd
    """
    n.add(cls_name, dataframe.index, **dataframe)


@deprecated(
    deprecated_in="0.29",
    removed_in="1.0",
    details="Use `n.add` instead.",
)
def import_series_from_dataframe(
    n: Network, dataframe: pd.DataFrame, cls_name: str, attr: str
) -> None:
    """
    Import time series from a pandas DataFrame.

    This function is deprecated. Use :py:meth:`pypsa.Network.add` instead, but it will
    not work with the same data structure. To get a similar behavior, use
    `n.dynamic(class_name)[attr] = df` but make sure that the index is aligned. Also note
    that this is overwriting the attribute dataframe, not adding to it as before.
    It is better to use :py:meth:`pypsa.Network.add` to import time series data.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A DataFrame whose index is ``n.snapshots`` and
        whose columns are a subset of the relevant components.
    cls_name : string
        Name of class of component
    attr : string
        Name of time-varying series attribute

    Examples
    --------
    >>> import numpy as np
    >>> n.set_snapshots(range(10))
    >>> n.import_series_from_dataframe(
    ...     pd.DataFrame(np.random.rand(10, 4),
    ...         columns=n.generators.index,
    ...                     index=range(10)),
    ...                 "Generator",
    ...                 "p_max_pu")

    --------
    """
    _import_series_from_df(n, dataframe, cls_name, attr)


def _import_components_from_df(
    n: Network, df: pd.DataFrame, cls_name: str, overwrite: bool = False
) -> None:
    """
    Import components from a pandas DataFrame.

    If columns are missing then defaults are used.

    If extra columns are added, these are left in the resulting component dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame whose index is the names of the components and
        whose columns are the non-default attributes.
    cls_name : string
        Name of class of component, e.g. ``"Line", "Bus", "Generator", "StorageUnit"``
    """
    attrs = n.components[cls_name]["attrs"]

    static_attrs = attrs[attrs.static].drop("name")
    non_static_attrs = attrs[~attrs.static]

    if cls_name == "Link":
        update_linkports_component_attrs(n, where=df)

    # Clean dataframe and ensure correct types
    df = pd.DataFrame(df)
    df.index = df.index.astype(str)

    # Fill nan values with default values
    df = df.fillna(attrs["default"].to_dict())

    for k in static_attrs.index:
        if k not in df.columns:
            df[k] = static_attrs.at[k, "default"]
        else:
            if static_attrs.at[k, "type"] == "string":
                df[k] = df[k].replace({np.nan: ""})
            if static_attrs.at[k, "type"] == "int":
                df[k] = df[k].fillna(0)
            if df[k].dtype != static_attrs.at[k, "typ"]:
                if static_attrs.at[k, "type"] == "geometry":
                    geometry = df[k].replace({"": None, np.nan: None})
                    from shapely.geometry.base import BaseGeometry

                    if geometry.apply(lambda x: isinstance(x, BaseGeometry)).all():
                        df[k] = gpd.GeoSeries(geometry)
                    else:
                        df[k] = gpd.GeoSeries.from_wkt(geometry)
                else:
                    df[k] = df[k].astype(static_attrs.at[k, "typ"])

    # check all the buses are well-defined
    # TODO use func from consistency checks
    for attr in [attr for attr in df if attr.startswith("bus")]:
        # allow empty buses for multi-ports
        port = int(attr[-1]) if attr[-1].isdigit() else 0
        mask = ~df[attr].isin(n.buses.index)
        if port > 1:
            mask &= df[attr].ne("")
        missing = df.index[mask]
        if len(missing) > 0:
            logger.warning(
                "The following %s have buses which are not defined:\n%s",
                cls_name,
                missing,
            )

    non_static_attrs_in_df = non_static_attrs.index.intersection(df.columns)
    old_static = n.static(cls_name)
    new_static = df.drop(non_static_attrs_in_df, axis=1)

    # Handle duplicates
    duplicated_components = old_static.index.intersection(new_static.index)
    if len(duplicated_components) > 0:
        if not overwrite:
            logger.warning(
                "The following %s are already defined and will be skipped "
                "(use overwrite=True to overwrite): %s",
                n.components[cls_name]["list_name"],
                ", ".join(duplicated_components),
            )
            new_static = new_static.drop(duplicated_components)
        else:
            old_static = old_static.drop(duplicated_components)

    # Concatenate to new dataframe
    if not old_static.empty:
        new_static = pd.concat((old_static, new_static), sort=False)

    if cls_name == "Shape":
        new_static = gpd.GeoDataFrame(new_static, crs=n.crs)

    # Align index (component names) and columns (attributes)
    new_static = _sort_attrs(new_static, attrs.index, axis=1)

    new_static.index.name = cls_name
    setattr(n, n.components[cls_name]["list_name"], new_static)

    # Now deal with time-dependent properties

    dynamic = n.dynamic(cls_name)

    for k in non_static_attrs_in_df:
        # If reading in outputs, fill the outputs
        dynamic[k] = dynamic[k].reindex(
            columns=new_static.index, fill_value=non_static_attrs.at[k, "default"]
        )
        if overwrite:
            dynamic[k].loc[:, df.index] = df.loc[:, k].values
        else:
            new_components = df.index.difference(duplicated_components)
            dynamic[k].loc[:, new_components] = df.loc[new_components, k].values

    setattr(n, n.components[cls_name]["list_name"] + "_t", dynamic)


def _import_series_from_df(
    n: Network,
    df: pd.DataFrame,
    cls_name: str,
    attr: str,
    overwrite: bool = False,
) -> None:
    """
    Import time series from a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame whose index is ``n.snapshots`` and
        whose columns are a subset of the relevant components.
    cls_name : string
        Name of class of component
    attr : string
        Name of time-varying series attribute
    """
    static = n.static(cls_name)
    dynamic = n.dynamic(cls_name)
    list_name = n.components[cls_name]["list_name"]

    if not overwrite:
        try:
            df = df.drop(df.columns.intersection(dynamic[attr].columns), axis=1)
        except KeyError:
            pass  # Don't drop any columns if the data doesn't exist yet

    df.columns.name = cls_name
    df.index.name = "snapshot"

    # Check if components exist in static df
    diff = df.columns.difference(static.index)
    if len(diff) > 0:
        logger.warning(
            f"Components {diff} for attribute {attr} of {cls_name} "
            f"are not in main components dataframe {list_name}"
        )

    # Get all attributes for the component
    attrs = n.components[cls_name]["attrs"]

    # Add all unknown attributes to the dataframe without any checks
    expected_attrs = attrs[lambda ds: ds.type.str.contains("series")].index
    if attr not in expected_attrs:
        if overwrite or attr not in dynamic:
            dynamic[attr] = df
        return

    # Check if any snapshots are missing
    diff = n.snapshots.difference(df.index)
    if len(diff):
        logger.warning(
            f"Snapshots {diff} are missing from {attr} of {cls_name}."
            f" Filling with default value '{attrs.loc[attr].default}'"
        )
        df = df.reindex(n.snapshots, fill_value=attrs.loc[attr].default)

    if not attrs.loc[attr].static:
        dynamic[attr] = dynamic[attr].reindex(
            columns=df.columns.union(static.index),
            fill_value=attrs.loc[attr].default,
        )
    else:
        dynamic[attr] = dynamic[attr].reindex(
            columns=(df.columns.union(dynamic[attr].columns))
        )

    dynamic[attr].loc[n.snapshots, df.columns] = df.loc[n.snapshots, df.columns]


@deprecated_common_kwargs
def merge(
    n: Network,
    other: Network,
    components_to_skip: Collection[str] | None = None,
    inplace: bool = False,
    with_time: bool = True,
) -> Network | None:
    """
    Merge the components of two networks.

    Requires disjunct sets of component indices and, if time-dependent data is
    merged, identical snapshots and snapshot weightings.

    If a component in ``other`` does not have values for attributes present in
    ``n``, default values are set.

    If a component in ``other`` has attributes which are not present in
    ``n`` these attributes are ignored.

    Parameters
    ----------
    n : pypsa.Network
        Network to add to.
    other : pypsa.Network
        Network to add from.
    components_to_skip : list-like, default None
        List of names of components which are not to be merged e.g. "Bus"
    inplace : bool, default False
        If True, merge into ``n`` in-place, otherwise a copy is made.
    with_time : bool, default True
        If False, only static data is merged.

    Returns
    -------
    receiving_n : pypsa.Network
        Merged network, or None if inplace=True
    """
    to_skip = {"Network", "SubNetwork", "LineType", "TransformerType"}
    if components_to_skip:
        to_skip.update(components_to_skip)
    to_iterate = other.all_components - to_skip
    # ensure buses are merged first
    to_iterate_list = ["Bus"] + sorted(to_iterate - {"Bus"})
    for c in other.iterate_components(to_iterate_list):
        if not c.static.index.intersection(n.static(c.name).index).empty:
            msg = f"Component {c.name} has overlapping indices, cannot merge networks."
            raise ValueError(msg)
    if with_time:
        snapshots_aligned = n.snapshots.equals(other.snapshots)
        weightings_aligned = n.snapshot_weightings.equals(other.snapshot_weightings)
        if not (snapshots_aligned and weightings_aligned):
            msg = (
                "Snapshots or snapshot weightings do not agree, cannot merge networks."
            )
            raise ValueError(msg)
    new = n if inplace else n.copy()
    if other.srid != new.srid:
        logger.warning(
            "Spatial Reference System Indentifier of networks do not agree: "
            f"{new.srid}, {other.srid}. Assuming {new.srid}."
        )
    for c in other.iterate_components(to_iterate_list):
        new.add(c.name, c.static.index, **c.static)
        if with_time:
            for k, v in c.dynamic.items():
                new._import_series_from_df(v, c.name, k)

    return None if inplace else new


@deprecated_common_kwargs
def import_from_pypower_ppc(
    n: Network, ppc: dict, overwrite_zero_s_nom: float | None = None
) -> None:
    """
    Import network from PYPOWER PPC dictionary format version 2.

    Converts all baseMVA to base power of 1 MVA.

    For the meaning of the pypower indices, see also pypower/idx_*.

    Parameters
    ----------
    ppc : PYPOWER PPC dict
    overwrite_zero_s_nom : Float or None, default None

    Examples
    --------
    >>> from pypower.api import case30
    >>> ppc = case30()
    >>> n.import_from_pypower_ppc(ppc)
    """
    version = ppc["version"]
    if int(version) != 2:
        logger.warning(
            "Warning, importing from PYPOWER may not work if PPC version is not 2!"
        )

    logger.warning(
        "Warning: Note that when importing from PYPOWER, some PYPOWER features not supported: areas, gencosts, component status"
    )

    baseMVA = ppc["baseMVA"]

    # add buses

    # integer numbering will be bus names
    index = np.array(ppc["bus"][:, 0], dtype=int)

    columns = [
        "type",
        "Pd",
        "Qd",
        "Gs",
        "Bs",
        "area",
        "v_mag_pu_set",
        "v_ang_set",
        "v_nom",
        "zone",
        "v_mag_pu_max",
        "v_mag_pu_min",
    ]

    pdf = {
        "buses": pd.DataFrame(
            index=index,
            columns=columns,
            data=ppc["bus"][:, 1 : len(columns) + 1],
        )
    }
    if (pdf["buses"]["v_nom"] == 0.0).any():
        logger.warning(
            "Warning, some buses have nominal voltage of 0., setting the nominal voltage of these to 1."
        )
        pdf["buses"].loc[pdf["buses"]["v_nom"] == 0.0, "v_nom"] = 1.0

    # rename controls
    controls = ["", "PQ", "PV", "Slack"]
    pdf["buses"]["control"] = pdf["buses"].pop("type").map(lambda i: controls[int(i)])

    # add loads for any buses with Pd or Qd
    pdf["loads"] = pdf["buses"].loc[
        pdf["buses"][["Pd", "Qd"]].any(axis=1), ["Pd", "Qd"]
    ]
    pdf["loads"]["bus"] = pdf["loads"].index
    pdf["loads"].rename(columns={"Qd": "q_set", "Pd": "p_set"}, inplace=True)
    pdf["loads"].index = [f"L{str(i)}" for i in range(len(pdf["loads"]))]

    # add shunt impedances for any buses with Gs or Bs

    shunt = pdf["buses"].loc[
        pdf["buses"][["Gs", "Bs"]].any(axis=1), ["v_nom", "Gs", "Bs"]
    ]

    # base power for shunt is 1 MVA, so no need to rebase here
    shunt["g"] = shunt["Gs"] / shunt["v_nom"] ** 2
    shunt["b"] = shunt["Bs"] / shunt["v_nom"] ** 2
    pdf["shunt_impedances"] = shunt.reindex(columns=["g", "b"])
    pdf["shunt_impedances"]["bus"] = pdf["shunt_impedances"].index
    pdf["shunt_impedances"].index = [
        f"S{str(i)}" for i in range(len(pdf["shunt_impedances"]))
    ]

    # add gens

    # it is assumed that the pypower p_max is the p_nom

    # could also do gen.p_min_pu = p_min/p_nom

    columns = "bus, p_set, q_set, q_max, q_min, v_set_pu, mva_base, status, p_nom, p_min, Pc1, Pc2, Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf".split(
        ", "
    )

    index_list = [f"G{str(i)}" for i in range(len(ppc["gen"]))]

    pdf["generators"] = pd.DataFrame(
        index=index_list, columns=columns, data=ppc["gen"][:, : len(columns)]
    )

    # make sure bus name is an integer
    pdf["generators"]["bus"] = np.array(ppc["gen"][:, 0], dtype=int)

    # add branchs
    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax

    columns = "bus0, bus1, r, x, b, s_nom, rateB, rateC, tap_ratio, phase_shift, status, v_ang_min, v_ang_max".split(
        ", "
    )

    pdf["branches"] = pd.DataFrame(
        columns=columns, data=ppc["branch"][:, : len(columns)]
    )

    pdf["branches"]["original_index"] = pdf["branches"].index

    pdf["branches"]["bus0"] = pdf["branches"]["bus0"].astype(int)
    pdf["branches"]["bus1"] = pdf["branches"]["bus1"].astype(int)

    # s_nom = 0 indicates an unconstrained line
    zero_s_nom = pdf["branches"]["s_nom"] == 0.0
    if zero_s_nom.any():
        if overwrite_zero_s_nom is not None:
            pdf["branches"].loc[zero_s_nom, "s_nom"] = overwrite_zero_s_nom
        else:
            logger.warning(
                f"Warning: there are {zero_s_nom.sum()} branches with s_nom equal to zero, they will probably lead to infeasibilities and should be replaced with a high value using the `overwrite_zero_s_nom` argument."
            )

    # determine bus voltages of branches to detect transformers
    v_nom = pdf["branches"].bus0.map(pdf["buses"].v_nom)
    v_nom_1 = pdf["branches"].bus1.map(pdf["buses"].v_nom)

    # split branches into transformers and lines
    transformers = (
        (v_nom != v_nom_1)
        | (
            (pdf["branches"].tap_ratio != 0.0) & (pdf["branches"].tap_ratio != 1.0)
        )  # NB: PYPOWER has strange default of 0. for tap ratio
        | (pdf["branches"].phase_shift != 0)
    )
    pdf["transformers"] = pd.DataFrame(pdf["branches"][transformers])
    pdf["lines"] = pdf["branches"][~transformers].drop(
        ["tap_ratio", "phase_shift"], axis=1
    )

    # convert transformers from base baseMVA to base s_nom
    pdf["transformers"]["r"] = (
        pdf["transformers"]["r"] * pdf["transformers"]["s_nom"] / baseMVA
    )
    pdf["transformers"]["x"] = (
        pdf["transformers"]["x"] * pdf["transformers"]["s_nom"] / baseMVA
    )
    pdf["transformers"]["b"] = (
        pdf["transformers"]["b"] * baseMVA / pdf["transformers"]["s_nom"]
    )

    # correct per unit impedances
    pdf["lines"]["r"] = v_nom**2 * pdf["lines"]["r"] / baseMVA
    pdf["lines"]["x"] = v_nom**2 * pdf["lines"]["x"] / baseMVA
    pdf["lines"]["b"] = pdf["lines"]["b"] * baseMVA / v_nom**2

    if (pdf["transformers"]["tap_ratio"] == 0.0).any():
        logger.warning(
            "Warning, some transformers have a tap ratio of 0., setting the tap ratio of these to 1."
        )
        pdf["transformers"].loc[
            pdf["transformers"]["tap_ratio"] == 0.0, "tap_ratio"
        ] = 1.0

    # name them nicely
    pdf["transformers"].index = [f"T{str(i)}" for i in range(len(pdf["transformers"]))]
    pdf["lines"].index = [f"L{str(i)}" for i in range(len(pdf["lines"]))]

    # TODO

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0

    for component in [
        "Bus",
        "Load",
        "Generator",
        "Line",
        "Transformer",
        "ShuntImpedance",
    ]:
        n.add(
            component,
            pdf[n.components[component]["list_name"]].index,
            **pdf[n.components[component]["list_name"]],
        )

    n.generators["control"] = n.generators.bus.map(n.buses["control"])

    # for consistency with pypower, take the v_mag set point from the generators
    n.buses.loc[n.generators.bus, "v_mag_pu_set"] = np.asarray(n.generators["v_set_pu"])


@deprecated_common_kwargs
def import_from_pandapower_net(
    n: Network,
    net: pandapowerNet,
    extra_line_data: bool = False,
    use_pandapower_index: bool = False,
) -> None:
    """
    Import PyPSA network from pandapower net.

    Importing from pandapower is still in beta;
    not all pandapower components are supported.

    Unsupported features include:
    - three-winding transformers
    - switches
    - in_service status and
    - tap positions of transformers

    Parameters
    ----------
    net : pandapower network
    extra_line_data : boolean, default: False
        if True, the line data for all parameters is imported instead of only the type
    use_pandapower_index : boolean, default: False
        if True, use integer numbers which is the pandapower index standard
        if False, use any net.name as index (e.g. 'Bus 1' (str) or 1 (int))

    Examples
    --------
    >>> n.import_from_pandapower_net(net)
    OR
    >>> import pypsa
    >>> import pandapower as pp
    >>> import pandapower.networks as pn
    >>> net = pn.create_cigre_network_mv(with_der='all')
    >>> n = pypsa.Network()
    >>> n.import_from_pandapower_net(net, extra_line_data=True)
    """
    logger.warning(
        "Warning: Importing from pandapower is still in beta; not all pandapower data is supported.\nUnsupported features include: three-winding transformers, switches, in_service status, shunt impedances and tap positions of transformers."
    )

    d = {
        "Bus": pd.DataFrame(
            {"v_nom": net.bus.vn_kv.values, "v_mag_pu_set": 1.0},
            index=net.bus.name,
        )
    }

    d["Bus"].loc[net.bus.name.loc[net.gen.bus].values, "v_mag_pu_set"] = (
        net.gen.vm_pu.values  # fmt: skip
    )

    d["Bus"].loc[net.bus.name.loc[net.ext_grid.bus].values, "v_mag_pu_set"] = (
        net.ext_grid.vm_pu.values  # fmt: skip
    )

    d["Load"] = pd.DataFrame(
        {
            "p_set": (net.load.scaling * net.load.p_mw).values,
            "q_set": (net.load.scaling * net.load.q_mvar).values,
            "bus": net.bus.name.loc[net.load.bus].values,
        },
        index=net.load.name,
    )

    # deal with PV generators
    _tmp_gen = pd.DataFrame(
        {
            "p_set": (net.gen.scaling * net.gen.p_mw).values,
            "q_set": 0.0,
            "bus": net.bus.name.loc[net.gen.bus].values,
            "control": "PV",
        },
        index=net.gen.name,
    )

    # deal with PQ "static" generators
    _tmp_sgen = pd.DataFrame(
        {
            "p_set": (net.sgen.scaling * net.sgen.p_mw).values,
            "q_set": (net.sgen.scaling * net.sgen.q_mvar).values,
            "bus": net.bus.name.loc[net.sgen.bus].values,
            "control": "PQ",
        },
        index=net.sgen.name,
    )

    _tmp_ext_grid = pd.DataFrame(
        {
            "control": "Slack",
            "p_set": 0.0,
            "q_set": 0.0,
            "bus": net.bus.name.loc[net.ext_grid.bus].values,
        },
        index=net.ext_grid.name.fillna("External Grid"),
    )

    # concat all generators and index according to option
    d["Generator"] = pd.concat(
        [_tmp_gen, _tmp_sgen, _tmp_ext_grid], ignore_index=use_pandapower_index
    )

    if extra_line_data is False:
        d["Line"] = pd.DataFrame(
            {
                "type": net.line.std_type.values,
                "bus0": net.bus.name.loc[net.line.from_bus].values,
                "bus1": net.bus.name.loc[net.line.to_bus].values,
                "length": net.line.length_km.values,
                "num_parallel": net.line.parallel.values,
            },
            index=net.line.name,
        )
    else:
        r = net.line.r_ohm_per_km.values * net.line.length_km.values
        x = net.line.x_ohm_per_km.values * net.line.length_km.values
        # capacitance values from pandapower in nF; transformed here:
        f = net.f_hz
        b = net.line.c_nf_per_km.values * net.line.length_km.values * 1e-9
        b = b * 2 * math.pi * f

        u = net.bus.vn_kv.loc[net.line.from_bus].values
        s_nom = u * net.line.max_i_ka.values

        d["Line"] = pd.DataFrame(
            {
                "r": r,
                "x": x,
                "b": b,
                "s_nom": s_nom,
                "bus0": net.bus.name.loc[net.line.from_bus].values,
                "bus1": net.bus.name.loc[net.line.to_bus].values,
                "length": net.line.length_km.values,
                "num_parallel": net.line.parallel.values,
            },
            index=net.line.name,
        )

    # check, if the trafo is based on a standard-type:
    if net.trafo.std_type.any():
        d["Transformer"] = pd.DataFrame(
            {
                "type": net.trafo.std_type.values,
                "bus0": net.bus.name.loc[net.trafo.hv_bus].values,
                "bus1": net.bus.name.loc[net.trafo.lv_bus].values,
                "tap_position": net.trafo.tap_pos.values,
            },
            index=net.trafo.name,
        )
    else:
        s_nom = net.trafo.sn_mva.values

        # documented at https://pandapower.readthedocs.io/en/develop/elements/trafo.html?highlight=transformer#impedance-values
        z = net.trafo.vk_percent.values / 100.0 / net.trafo.sn_mva.values
        r = net.trafo.vkr_percent.values / 100.0 / net.trafo.sn_mva.values
        x = np.sqrt(z**2 - r**2)

        y = net.trafo.i0_percent.values / 100.0
        g = (
            net.trafo.pfe_kw.values
            / net.trafo.sn_mva.values
            / 1000
            / net.trafo.sn_mva.values
        )
        b = np.sqrt(y**2 - g**2)

        d["Transformer"] = pd.DataFrame(
            {
                "phase_shift": net.trafo.shift_degree.values,
                "s_nom": s_nom,
                "bus0": net.bus.name.loc[net.trafo.hv_bus].values,
                "bus1": net.bus.name.loc[net.trafo.lv_bus].values,
                "r": r,
                "x": x,
                "g": g,
                "b": b,
                "tap_position": net.trafo.tap_pos.values,
            },
            index=net.trafo.name,
        )
    d["Transformer"] = d["Transformer"].fillna(0)

    # documented at https://pypsa.readthedocs.io/en/latest/components.html#shunt-impedance
    g_shunt = net.shunt.p_mw.values / net.shunt.vn_kv.values**2
    b_shunt = net.shunt.q_mvar.values / net.shunt.vn_kv.values**2

    d["ShuntImpedance"] = pd.DataFrame(
        {
            "bus": net.bus.name.loc[net.shunt.bus].values,
            "g": g_shunt,
            "b": b_shunt,
        },
        index=net.shunt.name,
    )
    d["ShuntImpedance"] = d["ShuntImpedance"].fillna(0)

    for component_name in [
        "Bus",
        "Load",
        "Generator",
        "Line",
        "Transformer",
        "ShuntImpedance",
    ]:
        n.add(component_name, d[component_name].index, **d[component_name])

    # amalgamate buses connected by closed switches

    bus_switches = net.switch[(net.switch.et == "b") & net.switch.closed]

    bus_switches["stays"] = bus_switches.bus.map(net.bus.name)
    bus_switches["goes"] = bus_switches.element.map(net.bus.name)

    to_replace = pd.Series(bus_switches.stays.values, bus_switches.goes.values)

    for i in to_replace.index:
        n.remove("Bus", i)

    for component in n.iterate_components({"Load", "Generator", "ShuntImpedance"}):
        component.static.replace({"bus": to_replace}, inplace=True)

    for component in n.iterate_components({"Line", "Transformer"}):
        component.static.replace({"bus0": to_replace}, inplace=True)
        component.static.replace({"bus1": to_replace}, inplace=True)
