"""Functions for importing and exporting data."""

from __future__ import annotations

import functools
import json
import logging
import math
import tempfile
from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, overload
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import pandas as pd
import validators
import xarray as xr
from deprecation import deprecated
from pandas.errors import ParserError
from pyproj import CRS
from typing_extensions import Self

from pypsa._options import options
from pypsa.common import _check_for_update, check_optional_dependency, deprecated_kwargs
from pypsa.descriptors import _update_linkports_component_attrs
from pypsa.network.abstract import _NetworkABC
from pypsa.version import __version_semver__, __version_semver_tuple__

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path
if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from pandapower.auxiliary import pandapowerNet

    from pypsa import Network
logger = logging.getLogger(__name__)


@overload
def _retrieve_from_url(
    url: str, io_function: Callable[[Path], pd.read_excel]
) -> pd.DataFrame: ...


@overload
def _retrieve_from_url(
    url: str, io_function: Callable[[Path], pd.HDFStore | xr.Dataset]
) -> Network: ...


@functools.lru_cache(maxsize=128)
def _retrieve_from_url(url: str, io_function: Callable) -> pd.DataFrame | Network:
    # Check if network requests are allowed
    if not options.get_option("general.allow_network_requests"):
        msg = "Network requests are disabled. Set `pypsa.options.general.allow_network_requests = True` to enable URL loading."
        raise ValueError(msg)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = Path(temp_file.name)
        logger.info("Retrieving network data from %s.", url)
        if not url.startswith("http"):
            msg = f"Invalid URL: {url}"
            raise ValueError(msg)
        try:
            urlretrieve(url, file_path)  # noqa: S310
        except Exception as e:
            msg = f"Failed to retrieve network data from {url}: {e}"
            raise ValueError(msg) from e
        return io_function(file_path)


# TODO: Restructure abc inheritance


class _ImpExper:
    """Base class for importers and exporters."""

    ds: Any = None

    def __enter__(self) -> Self:
        """Enter the context manager."""
        if self.ds is not None:
            self.ds = self.ds.__enter__()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        """Exit the context manager."""
        if exc_type is None:
            self.finish()

        if self.ds is not None:
            self.ds.__exit__(exc_type, exc_val, exc_tb)

    @abstractmethod
    def finish(self) -> None:
        """Post-processing when process is finished."""


class _Exporter(_ImpExper):
    """_Exporter class."""

    path: Path

    def remove_static(self, list_name: str) -> None:
        """Remove static components data."""

    def remove_series(self, list_name: str, attr: str) -> None:
        """Remove dynamic components data."""

    @abstractmethod
    def save_attributes(self, attrs: dict) -> None:
        """Save generic network attributes."""

    @abstractmethod
    def save_meta(self, meta: dict) -> None:
        """Save meta data (`n.meta`)."""

    @abstractmethod
    def save_crs(self, crs: dict) -> None:
        """Save CRS of shapes of network."""

    @abstractmethod
    def save_snapshots(self, snapshots: Sequence) -> None:
        """Save snapshots data."""

    @abstractmethod
    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        """Save investment periods data."""

    @abstractmethod
    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        """Save static components data."""

    @abstractmethod
    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save dynamic components data."""


class _Importer(_ImpExper):
    """Importer class."""


class _ImporterCSV(_Importer):
    """Importer class for CSV files."""

    def __init__(self, path: str | Path, encoding: str | None, quotechar: str) -> None:
        """Initialize the importer for CSV files.

        Parameters
        ----------
        path : str | Path
            Path to the CSV folder.
        encoding : str | None
            Encoding to use for the CSV files.
        quotechar : str
            Quote character to use for the CSV files.

        """
        self.path = Path(path)
        self.encoding = encoding
        self.quotechar = quotechar

        if not self.path.is_dir():
            msg = f"Directory {path} does not exist."
            raise FileNotFoundError(msg)

    def get_attributes(self) -> dict | None:
        """Get generic network attributes."""
        fn = self.path.joinpath("network.csv")
        if not fn.is_file():
            return None

        dtypes = {"pypsa_version": str, "name": str}
        return dict(
            pd.read_csv(
                fn, encoding=self.encoding, dtype=dtypes, quotechar=self.quotechar
            ).iloc[0]
        )

    def get_meta(self) -> dict:
        """Get meta data (`n.meta`)."""
        fn = self.path.joinpath("meta.json")
        return {} if not fn.is_file() else json.loads(fn.open().read())

    def get_crs(self) -> dict:
        """Get CRS of shapes of network."""
        fn = self.path.joinpath("crs.json")
        return {} if not fn.is_file() else json.loads(fn.open().read())

    def get_snapshots(self) -> pd.Index:
        """Get snapshots data."""
        fn = self.path.joinpath("snapshots.csv")
        if not fn.is_file():
            return None
        df = pd.read_csv(
            fn,
            index_col=0,
            encoding=self.encoding,
            quotechar=self.quotechar,
        )

        # Convert snapshot and timestep to datetime (if possible)
        if "snapshot" in df and df.snapshot.iloc[0] != "now":
            try:
                df["snapshot"] = pd.to_datetime(df.snapshot)
            except (ValueError, ParserError):
                pass
        if "timestep" in df and df.timestep.iloc[0] != "now":
            try:
                df["timestep"] = pd.to_datetime(df.timestep)
            except (ValueError, ParserError):
                pass
        return df

    def get_investment_periods(self) -> pd.Series:
        """Get investment periods data."""
        fn = self.path.joinpath("investment_periods.csv")
        if not fn.is_file():
            return None
        return pd.read_csv(
            fn, index_col=0, encoding=self.encoding, quotechar=self.quotechar
        )

    def get_static(self, list_name: str) -> pd.DataFrame:
        """Get static components data."""
        fn = self.path.joinpath(list_name + ".csv")
        if not fn.is_file():
            return None

        df = pd.read_csv(
            fn, index_col=0, encoding=self.encoding, quotechar=self.quotechar
        )

        # Convert NaN to empty strings for object dtype columns to handle custom attributes
        object_cols = [col for col in df.columns if df[col].dtype == "object"]
        if object_cols:
            df[object_cols] = df[object_cols].fillna("")

        return df

    def get_series(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get dynamic components data."""
        for fn in self.path.iterdir():
            if fn.name.startswith(list_name + "-") and fn.name.endswith(".csv"):
                attr = fn.name[len(list_name) + 1 : -4]
                df = pd.read_csv(
                    self.path.joinpath(fn.name),
                    index_col=0,
                    encoding=self.encoding,
                    quotechar=self.quotechar,
                    parse_dates=True,
                )
                yield attr, df

    def finish(self) -> None:
        """Finish the import process."""


class _ExporterCSV(_Exporter):
    """Exporter class for CSV files."""

    def __init__(self, path: Path | str, encoding: str | None, quotechar: str) -> None:
        """Initialize the exporter for CSV files.

        Parameters
        ----------
        path : Path | str
            Path to the CSV folder.
        encoding : str | None
            Encoding to use for the CSV files.
        quotechar : str
            Quote character to use for the CSV files.

        """
        self.path = Path(path)
        self.encoding = encoding
        self.quotechar = quotechar

        # make sure directory exists
        if not self.path.is_dir():
            logger.warning("Directory %s does not exist, creating it", path)
            self.path.mkdir()

    def save_attributes(self, attrs: dict) -> None:
        """Save generic network attributes."""
        name = attrs.pop("name")
        df = pd.DataFrame(attrs, index=pd.Index([name], name="name"))
        fn = self.path.joinpath("network.csv")
        with fn.open("w"):
            df.to_csv(fn, encoding=self.encoding, quotechar=self.quotechar)

    def save_meta(self, meta: dict) -> None:
        """Save meta data (`n.meta`)."""
        fn = self.path.joinpath("meta.json")
        fn.open("w").write(json.dumps(meta))

    def save_crs(self, crs: dict) -> None:
        """Save CRS of shapes of network."""
        fn = self.path.joinpath("crs.json")
        fn.open("w").write(json.dumps(crs))

    def save_snapshots(self, snapshots: pd.Index) -> None:
        """Save snapshots data."""
        fn = self.path.joinpath("snapshots.csv")
        with fn.open("w"):
            snapshots.to_csv(fn, encoding=self.encoding, quotechar=self.quotechar)

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        """Save investment periods data."""
        fn = self.path.joinpath("investment_periods.csv")
        with fn.open("w"):
            investment_periods.to_csv(
                fn, encoding=self.encoding, quotechar=self.quotechar
            )

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        """Save static components data."""
        fn = self.path.joinpath(list_name + ".csv")
        with fn.open("w"):
            df.to_csv(fn, encoding=self.encoding, quotechar=self.quotechar)

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save dynamic components data."""
        fn = self.path.joinpath(list_name + "-" + attr + ".csv")
        with fn.open("w"):
            df.to_csv(fn, encoding=self.encoding, quotechar=self.quotechar)

    def remove_static(self, list_name: str) -> None:
        """Remove static components data.

        Needed to not have stale sheets for empty components.
        """
        if fns := list(self.path.joinpath(list_name).glob("*.csv")):
            for fn in fns:
                fn.unlink()
            logger.warning("Stale csv file(s) %s removed", ", ".join(fns))

    def remove_series(self, list_name: str, attr: str) -> None:
        """Remove dynamic components data.

        Needed to not have stale sheets for empty components.
        """
        fn = self.path.joinpath(list_name + "-" + attr + ".csv")
        if fn.exists():
            fn.unlink()

    def finish(self) -> None:
        """Finish the export process."""


class _ImporterExcel(_Importer):
    """Importer class for Excel files."""

    def __init__(self, path: str | Path, engine: str = "calamine") -> None:
        """Initialize the importer for Excel files.

        Parameters
        ----------
        path : str | Path
            Path to the Excel file.
        engine : str
            Engine to use for the Excel file.

        """
        if engine == "calamine":
            check_optional_dependency(
                "python_calamine",
                "Missing optional dependencies to use Excel files. Install them via "
                "`pip install pypsa[excel]`. If you passed any other engine, "
                "make sure it is installed.",
            )
        if not isinstance(path, (str | Path)):
            msg = f"Invalid path type. Expected str or Path, got {type(path)}."
            raise TypeError(msg)

        path = Path(path)
        if not path.is_file():
            msg = f"Excel file {path} does not exist."
            raise FileNotFoundError(msg)
        self.engine = engine

        reader = partial(pd.read_excel, sheet_name=None, engine=self.engine)
        if validators.url(str(path)):
            self.sheets = _retrieve_from_url(str(path), reader)
        else:
            self.sheets = reader(path)
        self.index: dict = {}

    def get_attributes(self) -> dict | None:
        """Get generic network attributes."""
        try:
            # Ensure name and pypsa_version are read as strings to prevent
            # automatic type conversion (e.g., numeric names like "123")
            df = self.sheets["network"]
            if "name" in df.columns:
                df["name"] = df["name"].astype(str)

                df["name"] = df["name"].replace("nan", "")
            if "pypsa_version" in df.columns:
                df["pypsa_version"] = df["pypsa_version"].astype(str)
            return dict(df.iloc[0])
        except (ValueError, KeyError):
            return None

    def get_meta(self) -> dict:
        """Get meta data (`n.meta`)."""
        try:
            df = self.sheets["meta"]
            if not df.empty:
                meta = {}
                for _, row in df.iterrows():
                    key = row["Key"]
                    value = row["Value"]

                    # Try to parse JSON strings back into dictionaries
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass

                    meta[key] = value
                return meta
        except (ValueError, KeyError):
            return {}
        else:
            return {}

    def get_crs(self) -> dict:
        """Get CRS of shapes of network."""
        try:
            df = self.sheets["crs"]
            if not df.empty:
                # Assuming first column is keys and second column is values
                return dict(zip(df.iloc[:, 0], df.iloc[:, 1], strict=False))
        except (ValueError, KeyError):
            return {}
        else:
            return {}

    def get_snapshots(self) -> pd.Index:
        """Get snapshots data."""
        df = self.sheets["snapshots"]
        df = df.set_index(df.columns[0])
        # Convert snapshot and timestep to datetime (if possible)
        if "snapshot" in df and df.snapshot.iloc[0] != "now":
            try:
                df["snapshot"] = pd.to_datetime(df.snapshot)
            except (ValueError, ParserError):
                pass
        if "timestep" in df and df.timestep.iloc[0] != "now":
            try:
                df["timestep"] = pd.to_datetime(df.timestep)
            except (ValueError, ParserError):
                pass
        return df

    def get_investment_periods(self) -> pd.Series:
        """Get investment periods data."""
        try:
            df = self.sheets["investment_periods"]
            df = df.set_index(df.columns[0])
        except (ValueError, KeyError):
            return None
        else:
            return df

    def get_static(self, list_name: str) -> pd.DataFrame:
        """Get static components data."""
        try:
            df = self.sheets[list_name]
            df = df.set_index(df.columns[0])

            # Handle DataFrames with only index values that were exported from PyPSA
            # Otherwise the column row is read in as a component
            if len(df.columns) == 0 and len(df.index) > 0 and df.index[0] == "name":
                df = df.iloc[1:]  # Remove the first row which contains the index name

            # Convert NaN to empty strings for object dtype columns to handle custom attributes
            object_cols = [col for col in df.columns if df[col].dtype == "object"]
            if object_cols:
                df[object_cols] = df[object_cols].fillna("")

        except (ValueError, KeyError):
            return None
        else:
            return df

    def get_series(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get dynamic components data."""
        for sheet_name, df in self.sheets.items():
            if sheet_name.startswith(list_name + "-"):
                attr = sheet_name[len(list_name) + 1 :]
                df = df.set_index(df.columns[0])
                yield attr, df

    def finish(self) -> None:
        """Finish the import process."""


class _ExporterExcel(_Exporter):
    """Exporter class for Excel files."""

    def __init__(self, path: Path | str, engine: str = "openpyxl") -> None:
        """Initialize the exporter for Excel files.

        Parameters
        ----------
        path : Path | str
            Path to save the Excel file.
        engine : str
            Engine to use for the Excel file.

        """
        if engine == "openpyxl":
            check_optional_dependency(
                "openpyxl",
                "Missing optional dependencies to use Excel files. Install them via "
                "`pip install pypsa[excel]`. If you passed any other engine, "
                "make sure it is installed.",
            )
        self.engine = engine
        self.path = Path(path)
        # Create an empty Excel file if it doesn't exist
        if not self.path.exists():
            logger.warning("Excel file %s does not exist, creating it", path)
            with pd.ExcelWriter(self.path, engine=self.engine) as writer:
                pd.DataFrame().to_excel(writer, sheet_name="_temp")

        # Keep track of sheets to avoid overwriting
        self._writer = None

    @property
    def writer(self) -> pd.ExcelWriter:
        """Get the Excel writer object.

        If the writer object is not already created, create it.
        """
        if self._writer is None:
            self._writer = pd.ExcelWriter(
                self.path,
                engine=self.engine,
                mode="a" if self.path.exists() else "w",
                if_sheet_exists="replace",
            )
        return self._writer

    def save_attributes(self, attrs: dict) -> None:
        """Save generic network attributes."""
        name = attrs.pop("name")
        df = pd.DataFrame(attrs, index=pd.Index([name], name="name"))
        df.to_excel(self.writer, sheet_name="network")

    def save_meta(self, meta: dict) -> None:
        """Save meta data (`n.meta`)."""
        # Convert meta dictionary to DataFrame with proper handling of nested dicts
        meta_items = []
        for key, value in meta.items():
            # If value is a dict, serialize it as JSON
            if isinstance(value, dict):
                value = json.dumps(value)
            meta_items.append([key, value])

        df = pd.DataFrame(meta_items, columns=["Key", "Value"])
        df.to_excel(self.writer, sheet_name="meta", index=False)

    def save_crs(self, crs: dict) -> None:
        """Save CRS of shapes of network."""
        df = pd.DataFrame(list(crs.items()), columns=["Key", "Value"])
        df.to_excel(self.writer, sheet_name="crs", index=False)

    def save_snapshots(self, snapshots: pd.Index) -> None:
        """Save snapshots data."""
        snapshots.to_excel(self.writer, sheet_name="snapshots")

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        """Save investment periods data."""
        investment_periods.to_excel(self.writer, sheet_name="investment_periods")

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        """Save static components data."""
        df.to_excel(self.writer, sheet_name=list_name)

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save dynamic components data."""
        sheet_name = f"{list_name}-{attr}"
        df.to_excel(self.writer, sheet_name=sheet_name)

    def remove_static(self, list_name: str) -> None:
        """Remove static components data.

        Needed to not have stale sheets for empty components.

        """
        if list_name in self.writer.book.sheetnames:
            del self.writer.book[list_name]
            logger.warning("Stale sheet %s removed", list_name)

    def remove_series(self, list_name: str, attr: str) -> None:
        """Remove dynamic components data.

        Needed to not have stale sheets for empty components.
        """
        sheet_name = f"{list_name}-{attr}"
        if sheet_name in self.writer.book.sheetnames:
            del self.writer.book[sheet_name]
            logger.warning("Stale sheet %s removed", sheet_name)

    def finish(self) -> None:
        """Postprocessing of exporting process."""
        # Remove temp sheet if it exists
        if "_temp" in self.writer.book.sheetnames:
            del self.writer.book["_temp"]
        # Close writer
        if self.writer is not None:
            self.writer.close()


class _ImporterHDF5(_Importer):
    """Importer class for HDF5 files."""

    def __init__(self, path: str | pd.HDFStore) -> None:
        """Initialize the importer for HDF5 files.

        Parameters
        ----------
        path : str | pd.HDFStore
            Path to the HDF5 file or an hdfstore object.

        """
        check_optional_dependency(
            "tables",
            "Missing optional dependencies to use HDF5 files. Install them via "
            "`pip install pypsa[hdf5]` or `conda install -c conda-forge pypsa[hdf5]`.",
        )
        self.path = path
        self.ds: pd.HDFStore
        if isinstance(path, (str | Path)):
            reader = partial(pd.HDFStore, mode="r")
            if validators.url(str(path)):
                self.ds = _retrieve_from_url(str(path), reader)
            else:
                self.ds = reader(Path(path))

        self.index: dict = {}

    def get_attributes(self) -> dict:
        """Get generic network attributes."""
        return dict(self.ds["/network"].reset_index().iloc[0])

    def get_meta(self) -> dict:
        """Get meta data (`n.meta`)."""
        return json.loads(self.ds["/meta"][0] if "/meta" in self.ds else "{}")

    def get_crs(self) -> dict:
        """Get CRS of shapes of network."""
        return json.loads(self.ds["/crs"][0] if "/crs" in self.ds else "{}")

    def get_snapshots(self) -> pd.Series:
        """Get snapshots data."""
        return self.ds["/snapshots"] if "/snapshots" in self.ds else None  # noqa: SIM401

    def get_investment_periods(self) -> pd.Series:
        """Get investment periods data."""
        return (
            self.ds["/investment_periods"] if "/investment_periods" in self.ds else None  # noqa: SIM401
        )

    def get_static(self, list_name: str) -> pd.DataFrame:
        """Get static components data."""
        if "/" + list_name not in self.ds:
            return None

        df = self.ds["/" + list_name].set_index("name")

        self.index[list_name] = df.index
        return df

    def get_series(self, list_name: str) -> Iterable[tuple[str, pd.DataFrame]]:
        """Get dynamic components data."""
        for tab in self.ds:
            if tab.startswith("/" + list_name + "_t/"):
                attr = tab[len("/" + list_name + "_t/") :]
                df = self.ds[tab]
                df.columns = self.index[list_name][df.columns]
                yield attr, df

    def finish(self) -> None:
        """Finish the import process."""


class _ExporterHDF5(_Exporter):
    """Exporter class for HDF5 files."""

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        """Initialize exporter for HDF5 files.

        Parameters
        ----------
        path : str | Path
            Path to save the HDF5 file.
        **kwargs : Any
            Additional keyword arguments for the HDFStore.

        """
        check_optional_dependency(
            "tables",
            "Missing optional dependencies to use HDF5 files. Install them via "
            "`pip install pypsa[hdf5]` or `conda install -c conda-forge pypsa[hdf5]`.",
        )
        self.path = Path(path)
        self._hdf5_handle = self.path.open("w")
        self.ds = pd.HDFStore(self.path, mode="w", **kwargs)
        self.index: dict = {}

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit the context manager."""
        super().__exit__(exc_type, exc_val, exc_tb)

    def save_attributes(self, attrs: dict) -> None:
        """Save generic network attributes."""
        name = attrs.pop("name")
        self.ds.put(
            "/network",
            pd.DataFrame(attrs, index=pd.Index([name], name="name")),
            format="table",
            index=False,
        )

    def save_meta(self, meta: dict) -> None:
        """Save meta data (`n.meta`)."""
        self.ds.put("/meta", pd.Series(json.dumps(meta)))

    def save_crs(self, crs: dict) -> None:
        """Save CRS of shapes of network."""
        self.ds.put("/crs", pd.Series(json.dumps(crs)))

    def save_snapshots(self, snapshots: Sequence) -> None:
        """Save snapshots data."""
        self.ds.put("/snapshots", snapshots, format="table", index=False)

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        """Save investment periods data."""
        self.ds.put(
            "/investment_periods",
            investment_periods,
            format="table",
            index=False,
        )

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        """Save a static components data."""
        df = df.rename_axis(index="name")
        self.index[list_name] = df.index
        df = df.reset_index()
        self.ds.put("/" + list_name, df, format="table", index=False)

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save dynamic components data."""
        df = df.set_axis(self.index[list_name].get_indexer(df.columns), axis="columns")
        self.ds.put("/" + list_name + "_t/" + attr, df, format="table", index=False)

    def finish(self) -> None:
        """Postprocessing of exporting process."""
        self._hdf5_handle.close()


class _ImporterNetCDF(_Importer):
    """Importer class for netCDF files."""

    ds: xr.Dataset

    def __init__(self, path: str | Path | xr.Dataset) -> None:
        """Initialize the importer for netCDF files.

        Parameters
        ----------
        path : str | Path | xr.Dataset
            Path to the netCDF file or an xarray.Dataset.

        """
        self.path = path
        if isinstance(path, (str | Path)):
            if validators.url(str(path)):
                self.ds = _retrieve_from_url(str(path), xr.open_dataset)
            else:
                self.ds = xr.open_dataset(Path(path))
        else:
            self.ds = path

    def __enter__(self) -> Self:
        """Enter the context manager."""
        if isinstance(self.path, (str | Path)):
            super().__init__()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        """Exit the context manager."""
        if isinstance(self.path, (str | Path)):
            super().__exit__(exc_type, exc_val, exc_tb)

    def get_attributes(self) -> dict:
        """Get generic network attributes."""
        return {
            attr[len("network_") :]: val
            for attr, val in self.ds.attrs.items()
            if attr.startswith("network_")
        }

    def get_meta(self) -> dict:
        """Get meta data (`n.meta`)."""
        return json.loads(self.ds.attrs.get("meta", "{}"))

    def get_crs(self) -> dict:
        """Get CRS of shapes of network."""
        return json.loads(self.ds.attrs.get("crs", "{}"))

    def get_snapshots(self) -> pd.DataFrame:
        """Get snapshots data."""
        return self.get_static("snapshots", "snapshots")

    def get_investment_periods(self) -> pd.DataFrame:
        """Get investment periods data."""
        return self.get_static("investment_periods", "investment_periods")

    def get_static(self, list_name: str, index_name: str | None = None) -> pd.DataFrame:
        """Get static components data."""
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
        """Get dynamic components data."""
        t = list_name + "_t_"
        for attr in self.ds.data_vars.keys():
            if attr.startswith(t):
                df = self.ds[attr].to_pandas()
                df.index.name = "name"
                df.columns.name = "name"
                yield attr[len(t) :], df

    def finish(self) -> None:
        """Finish the import process."""


class _ExporterNetCDF(_Exporter):
    """Exporter class for netCDF files."""

    def __init__(
        self,
        path: str | None,
        compression: dict | None = None,
        float32: bool = False,
    ) -> None:
        """Initialize exporter for netCDF files.

        Parameters
        ----------
        path : str | None
            Path to save the netCDF file.
        compression : dict | None, default None
            Compression settings for the netCDF file.
        float32 : bool, default False
            If True, typecast float64 to float32.

        """
        self.path = path
        if compression is None:
            compression = {"zlib": True, "complevel": 4}
        self.compression = compression
        self.float32 = float32
        self.ds = xr.Dataset()

    def save_attributes(self, attrs: dict) -> None:
        """Save generic network attributes."""
        self.ds.attrs.update(("network_" + attr, val) for attr, val in attrs.items())

    def save_meta(self, meta: dict) -> None:
        """Save meta data (`n.meta`)."""
        self.ds.attrs["meta"] = json.dumps(meta)

    def save_crs(self, crs: dict) -> None:
        """Save CRS of shapes of network."""
        self.ds.attrs["crs"] = json.dumps(crs)

    def save_snapshots(self, snapshots: pd.Index) -> None:
        """Save snapshots data."""
        snapshots = snapshots.rename_axis(index="snapshots")
        for attr in snapshots.columns:
            self.ds["snapshots_" + attr] = snapshots[attr]

    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        """Save investment periods data."""
        investment_periods = investment_periods.rename_axis(index="investment_periods")
        for attr in investment_periods.columns:
            self.ds["investment_periods_" + attr] = investment_periods[attr]

    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        """Save a static components data."""
        df = df.rename_axis(index=list_name + "_i")
        self.ds[list_name + "_i"] = df.index
        for attr in df.columns:
            self.ds[list_name + "_" + attr] = df[attr]

    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save a dynamic components data."""
        df = df.rename_axis(index="snapshots", columns=list_name + "_t_" + attr + "_i")
        self.ds[list_name + "_t_" + attr] = df

    def set_compression_encoding(self) -> None:
        """Set compression encoding for all variables."""
        logger.debug("Setting compression encodings: %s", self.compression)
        for v in self.ds.data_vars:
            if self.ds[v].dtype.kind not in ["U", "O"]:
                self.ds[v].encoding.update(self.compression)

    def typecast_float32(self) -> None:
        """Typecast float64 to float32 for all variables."""
        logger.debug("Typecasting float64 to float32.")
        for v in self.ds.data_vars:
            if self.ds[v].dtype == np.float64:
                self.ds[v] = self.ds[v].astype(np.float32)

    def finish(self) -> None:
        """Finish the export process.

        Runs post-processing, compression and saving to disk.
        """
        if self.float32:
            self.typecast_float32()
        if self.compression:
            self.set_compression_encoding()
        if self.path is not None:
            _path = Path(self.path)
            with _path.open("w"):
                self.ds.to_netcdf(_path)


def _sort_attrs(df: pd.DataFrame, attrs_list: list[str], axis: int) -> pd.DataFrame:
    """Sort axis of DataFrame according to the order of attrs_list.

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
    pd.DataFrame
        Sorted DataFrame

    """
    df_cols_set = set(df.columns if axis == 1 else df.index)

    existing_cols = [col for col in attrs_list if col in df_cols_set]
    remaining_cols = [
        col for col in (df.columns if axis == 1 else df.index) if col not in attrs_list
    ]
    return df.reindex(existing_cols + remaining_cols, axis=axis)


class NetworkIOMixin(_NetworkABC):
    """Mixin class for network I/O methods.

    Class only inherits to [pypsa.Network][] and should not be used directly.
    All attributes and methods can be used within any Network instance.
    """

    def _export_to_exporter(
        self,
        exporter: _Exporter,
        quotechar: str = '"',
        export_standard_types: bool = False,
    ) -> None:
        """Export to exporter.

        Both static and series attributes of components are exported, but only
        if they have non-default values.

        Parameters
        ----------
        exporter : _Exporter
            Initialized exporter instance
        quotechar : str, default '"'
            String of length 1. Character used to denote the start and end of a
            quoted item. Quoted items can include "," and it will be ignored
        export_standard_types : boolean, default False
            If True, then standard types are exported too (upon reimporting you
            should then set "ignore_standard_types" when initialising the netowrk).

        """
        # exportable component types
        allowed_types = (float, int, bool, str) + tuple(np.sctypeDict.values())

        # first export network properties
        _attrs = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if (
                not attr.startswith("__")
                and isinstance(getattr(self, attr), allowed_types)
            )
        }
        _attrs = {}
        for attr in dir(self):
            if not attr.startswith("__"):
                value = getattr(self, attr)
                if isinstance(value, allowed_types):
                    # TODO: This needs to be refactored with NetworkData class
                    # Skip properties without setter, but not 'pypsa_version'
                    prop = getattr(self.__class__, attr, None)
                    if (
                        isinstance(prop, property)
                        and prop.fset is None
                        and attr not in ["pypsa_version"]
                    ):
                        continue
                    # Skip `_name` since it is writable
                    if attr in ["_name", "_pypsa_version"]:
                        continue
                    _attrs[attr] = value
        exporter.save_attributes(_attrs)

        crs = {}
        if self.crs is not None:
            crs["_crs"] = self.crs.to_wkt()
        exporter.save_crs(crs)

        exporter.save_meta(self.meta)

        # now export snapshots
        if isinstance(self.snapshot_weightings.index, pd.MultiIndex):
            self.snapshot_weightings.index.rename(["period", "timestep"], inplace=True)
        else:
            self.snapshot_weightings.index.rename("snapshot", inplace=True)
        snapshots = self.snapshot_weightings.reset_index()
        exporter.save_snapshots(snapshots)

        # export investment period weightings
        investment_periods = self.investment_period_weightings
        exporter.save_investment_periods(investment_periods)

        exported_components = []
        for component in self.all_components - {"SubNetwork"}:
            list_name = self.components[component]["list_name"]
            attrs = self.components[component]["attrs"]

            static = self.static(component)
            dynamic = self.dynamic(component)

            if component == "Shape":
                static = pd.DataFrame(static).assign(
                    geometry=static["geometry"].to_wkt()
                )

            if not export_standard_types and component in self.standard_type_components:
                static = static.drop(self.components[component]["standard_types"].index)

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
                        col_export = dynamic[attr].columns[
                            (dynamic[attr] != default).any()
                        ]

                if len(col_export) > 0:
                    static = dynamic[attr].reset_index()[col_export]
                    exporter.save_series(list_name, attr, static)
                else:
                    exporter.remove_series(list_name, attr)

            exported_components.append(list_name)

        logger.info(
            "Exported network '%s'%s contains: %s",
            self.name,
            f"saved to '{exporter.path}" if exporter.path else "no file",
            ", ".join(exported_components),
        )

    def _import_from_importer(
        self, importer: Any, basename: str, skip_time: bool = False
    ) -> None:
        """Import network data from importer.

        Parameters
        ----------
        importer : Any
            Importer to import from.
        basename : str
            Name of the network.
        skip_time : bool
            Skip importing time

        """
        attrs = importer.get_attributes()
        self.meta = importer.get_meta()
        crs = importer.get_crs()
        crs = crs.pop("_crs", None)
        if crs is not None:
            crs = CRS.from_wkt(crs)
            self._crs = crs

        pypsa_version_tuple = (0, 0, 0)

        if attrs is not None:
            name = attrs.pop("name")
            self.name = name if pd.notna(name) else ""

            major = int(attrs.pop("pypsa_version", [0, 0, 0])[0])
            minor = int(attrs.pop("pypsa_version", [0, 0, 0])[1])
            patch = int(attrs.pop("pypsa_version", [0, 0, 0])[2])

            pypsa_version_tuple = (major, minor, patch)

            for attr, val in attrs.items():
                if attr in ["model", "objective", "objective_constant"]:
                    setattr(self, f"_{attr}", val)
                else:
                    setattr(self, attr, val)

        ## https://docs.python.org/3/tutorial/datastructures.html#comparing-sequences-and-other-types
        if pypsa_version_tuple < __version_semver_tuple__:
            pypsa_version_str = ".".join(map(str, pypsa_version_tuple))
            logger.warning(
                "Importing network from PyPSA version v%s while current version is v%s. Read the "
                "release notes at https://pypsa.readthedocs.io/en/latest/release_notes.html "
                "to prepare your network for import.",
                pypsa_version_str,
                __version_semver__,
            )

        # Check for newer PyPSA version available
        update_msg = _check_for_update(__version_semver_tuple__, "PyPSA", "pypsa")
        if update_msg:
            logger.info(update_msg)

        if pypsa_version_tuple < (0, 18, 0):
            self._multi_invest = 0

        # if there is snapshots.csv, read in snapshot data
        df = importer.get_snapshots()

        if df is not None:
            if snapshot_levels := {"period", "timestep", "snapshot"}.intersection(
                df.columns
            ):
                df.set_index(sorted(snapshot_levels), inplace=True)
            self.set_snapshots(df.index)

            cols = ["objective", "stores", "generators"]
            if not df.columns.intersection(cols).empty:
                # Preserve the default column order from Network.__init__
                existing_cols = [col for col in cols if col in df.columns]
                self.snapshot_weightings = df.reindex(
                    index=self.snapshots, columns=existing_cols
                )
            elif "weightings" in df.columns:
                self.snapshot_weightings = df["weightings"].reindex(self.snapshots)

            self.set_snapshots(df.index)

        # read in investment period weightings
        periods = importer.get_investment_periods()

        if periods is not None:
            self.periods = periods.index

            self._investment_period_weightings = periods.reindex(
                self.investment_periods
            )

        imported_components = []

        # now read in other components; make sure buses and carriers come first
        for component in ["Bus", "Carrier"] + sorted(
            self.all_components - {"Bus", "Carrier", "SubNetwork"}
        ):
            list_name = self.components[component]["list_name"]

            df = importer.get_static(list_name)
            if df is None:
                if component == "Bus":
                    logger.error("Error, no buses found")
                    return
                continue

            if component == "Link":
                _update_linkports_component_attrs(self, where=df)

            self.add(component, df.index, **df)

            if not skip_time:
                for attr, df in importer.get_series(list_name):
                    df.set_index(self.snapshots, inplace=True)
                    self._import_series_from_df(df, component, attr)

            logger.debug(getattr(self, list_name))

            imported_components.append(list_name)

        logger.info(
            "Imported network '%s' has %s",
            self.name,
            ", ".join(imported_components),
        )

    @deprecated_kwargs(deprecated_in="0.35", removed_in="1.0", csv_folder_name="path")
    def import_from_csv_folder(
        self,
        path: str | Path,
        encoding: str | None = None,
        quotechar: str = '"',
        skip_time: bool = False,
    ) -> None:
        """Import network data from CSVs in a folder.

        The CSVs must follow the standard form, see ``pypsa/examples``.

        Parameters
        ----------
        path : string
            Name of folder
        encoding : str, default None
            Encoding to use for UTF when reading (ex. 'utf-8'). `List of Python
            standard encodings
            <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
        quotechar : str, default '"'
            String of length 1. Character used to denote the start and end of a
            quoted item. Quoted items can include "," and it will be ignored
        skip_time : bool, default False
            Skip reading in time dependent attributes

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.import_from_csv_folder"./my_network") # doctest: +SKIP

        """
        basename = Path(path).name
        with _ImporterCSV(path, encoding=encoding, quotechar=quotechar) as importer:
            self._import_from_importer(importer, basename=basename, skip_time=skip_time)

    @deprecated_kwargs(deprecated_in="0.35", removed_in="1.0", csv_folder_name="path")
    def export_to_csv_folder(
        self,
        path: str,
        encoding: str | None = None,
        quotechar: str = '"',
        export_standard_types: bool = False,
    ) -> None:
        """Export network and components to a folder of CSVs.

        Both static and series attributes of all components are exported, but only
        if they have non-default values.

        If ``path`` does not already exist, it is created.

        ``path`` may also be a cloud object storage URI if cloudpathlib is installed.

        Static attributes are exported in one CSV file per component,
        e.g. ``generators.csv``.

        Series attributes are exported in one CSV file per component per
        attribute, e.g. ``generators-p_set.csv``.

        Parameters
        ----------
        path : string
            Name of folder to which to export.
        encoding : str, default None
            Encoding to use for UTF when reading (ex. 'utf-8'). `List of Python
            standard encodings
            <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
        quotechar : str, default '"'
            String of length 1. Character used to quote fields.
        export_standard_types : boolean, default False
            If True, then standard types are exported too (upon reimporting you
            should then set "ignore_standard_types" when initialising the network).

        Examples
        --------
        >>> n.export_to_csv_folder("my_network") # doctest: +SKIP

        See Also
        --------
        export_to_netcdf : Export to a netCDF file
        export_to_hdf5 : Export to an HDF5 file
        export_to_excel : Export to an Excel file

        """
        with _ExporterCSV(
            path=path, encoding=encoding, quotechar=quotechar
        ) as exporter:
            self._export_to_exporter(
                exporter, export_standard_types=export_standard_types
            )

    @deprecated_kwargs(deprecated_in="0.35", removed_in="1.0", excel_file_path="path")
    def import_from_excel(
        self,
        path: str | Path,
        skip_time: bool = False,
        engine: str = "calamine",
    ) -> None:
        """Import network data from an Excel file.

        The Excel file must follow the standard form with appropriate sheets.

        Parameters
        ----------
        path : string or Path
            Path to the Excel file
        skip_time : bool, default False
            Skip reading in time dependent attributes
        engine : string, default "calamine"
            The engine to use for reading the Excel file. See `pandas.read_excel
            <https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html>`_

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.import_from_excel("my_network.xlsx") # doctest: +SKIP

        """
        basename = Path(path).stem
        with _ImporterExcel(path, engine=engine) as importer:
            self._import_from_importer(importer, basename=basename, skip_time=skip_time)

    @deprecated_kwargs(deprecated_in="0.35", removed_in="1.0", excel_file_path="path")
    def export_to_excel(
        self,
        path: str | Path,
        export_standard_types: bool = False,
        engine: str = "openpyxl",
    ) -> None:
        """Export network and components to an Excel file.

        It is recommended to only use the Excel format if needed and for small networks.
        Excel files are not as efficient as other formats and can be slow to read/write.

        Both static and series attributes of all components are exported, but only
        if they have non-default values.

        If ``path`` does not already exist, it is created.

        Static attributes are exported in one sheet per component,
        e.g. a sheet named ``generators``.

        Series attributes are exported in one sheet per component per
        attribute, e.g. a sheet named ``generators-p_set``.

        Parameters
        ----------
        path : string or Path
            Path to the Excel file to which to export.
        export_standard_types : boolean, default False
            If True, then standard types are exported too (upon reimporting you
            should then set "ignore_standard_types" when initialising the network).
        engine : string, default "openpyxl"
            The engine to use for writing the Excel file. See `pandas.ExcelWriter
            <https://pandas.pydata.org/docs/reference/api/pandas.ExcelWriter.html>`_

        Examples
        --------
        >>> n.export_to_excel("my_network.xlsx") # doctest: +SKIP

        See Also
        --------
        export_to_netcdf : Export to a netCDF file
        export_to_hdf5 : Export to an HDF5 file
        export_to_csv_folder : Export to a folder of CSVs

        """
        with _ExporterExcel(path, engine=engine) as exporter:
            self._export_to_exporter(
                exporter, export_standard_types=export_standard_types
            )

    def import_from_hdf5(self, path: str | Path, skip_time: bool = False) -> None:
        """Import network data from HDF5 store at `path`.

        Parameters
        ----------
        path : string, Path
            Name of HDF5 store. The string could be a URL.
        skip_time : bool, default False
            Skip reading in time dependent attributes

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.import_from_hdf5("my_network.h5") # doctest: +SKIP

        """
        basename = Path(path).name

        with _ImporterHDF5(path) as importer:
            self._import_from_importer(importer, basename=basename, skip_time=skip_time)

    def export_to_hdf5(
        self,
        path: Path | str,
        export_standard_types: bool = False,
        **kwargs: Any,
    ) -> None:
        """Export network and components to an HDF store.

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
        >>> n.export_to_hdf5("my_network.h5") # doctest: +SKIP

        See Also
        --------
        export_to_netcdf : Export to a netCDF file
        export_to_csv_folder : Export to a folder of CSVs
        export_to_excel : Export to an Excel file

        """
        kwargs.setdefault("complevel", 4)

        with _ExporterHDF5(path, **kwargs) as exporter:
            self._export_to_exporter(
                exporter,
                export_standard_types=export_standard_types,
            )

    def import_from_netcdf(
        self, path: str | Path | xr.Dataset, skip_time: bool = False
    ) -> None:
        """Import network data from netCDF file or xarray Dataset at `path`.

        ``path`` may also be a cloud object storage URI if cloudpathlib is installed.

        Parameters
        ----------
        path : string|xr.Dataset
            Path to netCDF dataset or instance of xarray Dataset.
            The string could be a URL.
        skip_time : bool, default False
            Skip reading in time dependent attributes

        Examples
        --------
        >>> n = pypsa.Network()
        >>> n.import_from_netcdf("my_network.nc") # doctest: +SKIP

        """
        basename = "" if isinstance(path, xr.Dataset) else Path(path).name
        with _ImporterNetCDF(path=path) as importer:
            self._import_from_importer(importer, basename=basename, skip_time=skip_time)

    def export_to_netcdf(
        self,
        path: str | None = None,
        export_standard_types: bool = False,
        compression: dict | None = None,
        float32: bool = False,
    ) -> xr.Dataset:
        """Export network and components to a netCDF file.

        Both static and series attributes of components are exported, but only
        if they have non-default values.

        If path does not already exist, it is created.

        If no path is passed, no file is exported, but the xarray.Dataset
        is still returned.

        Be aware that this cannot export boolean attributes on the Network
        class, e.g. n.my_bool = False is not supported by netCDF.

        Parameters
        ----------
        path : string | None
            Name of netCDF file to which to export (if it exists, it is overwritten);
            if None is passed, no file is exported and only the xarray.Dataset is returned.
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
        >>> n = pypsa.Network()
        >>> n.export_to_netcdf("my_file.nc") # doctest: +SKIP

        See Also
        --------
        export_to_hdf5 : Export to an HDF5 file
        export_to_csv_folder : Export to a folder of CSVs
        export_to_excel : Export to an Excel file

        """
        with _ExporterNetCDF(path, compression, float32) as exporter:
            self._export_to_exporter(
                exporter, export_standard_types=export_standard_types
            )
            return exporter.ds

    @deprecated(
        deprecated_in="0.29",
        removed_in="1.0",
        details="Use `n.add` instead. E.g. `n.add(class_name, df.index, **df)`.",
    )
    def import_components_from_dataframe(
        self, dataframe: pd.DataFrame, cls_name: str
    ) -> None:
        """Import components from a pandas DataFrame.

        This function is deprecated. Use :py:meth`pypsa.Network.add` instead. To get the
        same behavior for importing components from a DataFrame, use
        `n.add(cls_name, df.index, **df)`.

        If columns are missing then defaults are used. If extra columns are added, these
        are left in the resulting component dataframe.

        !!! warning "Deprecated in v0.34"
            Use :py:meth:`pypsa.Network.add` instead.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            A DataFrame whose index is the names of the components and
            whose columns are the non-default attributes.
        cls_name : string
            Name of class of component, e.g. ``"Line", "Bus", "Generator", "StorageUnit"``

        See Also
        --------
        pypsa.Network.madd

        """
        self.add(cls_name, dataframe.index, **dataframe)

    @deprecated(
        deprecated_in="0.29",
        removed_in="1.0",
        details="Use `n.add` instead.",
    )
    def import_series_from_dataframe(
        self, dataframe: pd.DataFrame, cls_name: str, attr: str
    ) -> None:
        """Import time series from a pandas DataFrame.

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

        --------

        """
        self._import_series_from_df(dataframe, cls_name, attr)

    def _import_components_from_df(
        self, df: pd.DataFrame, cls_name: str, overwrite: bool = False
    ) -> None:
        """Import components from a pandas DataFrame.

        If columns are missing then defaults are used.

        If extra columns are added, these are left in the resulting component dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame whose index is the names of the components and
            whose columns are the non-default attributes.
        cls_name : string
            Name of class of component, e.g. ``"Line", "Bus", "Generator", "StorageUnit"``
        overwrite : bool, default False
            If True, overwrite existing components.

        """
        attrs = self.components[cls_name]["attrs"]

        static_attrs = attrs[attrs.static].drop("name")
        non_static_attrs = attrs[~attrs.static]

        if cls_name == "Link":
            _update_linkports_component_attrs(self, where=df)

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
            mask = ~df[attr].isin(self.components.buses.static.index)
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
        old_static = self.static(cls_name)
        new_static = df.drop(non_static_attrs_in_df, axis=1)

        # Handle duplicates
        duplicated_components = old_static.index.intersection(new_static.index)
        if len(duplicated_components) > 0:
            if not overwrite:
                logger.warning(
                    "The following %s are already defined and will be skipped "
                    "(use overwrite=True to overwrite): %s",
                    self.components[cls_name]["list_name"],
                    ", ".join(duplicated_components),
                )
                new_static = new_static.drop(duplicated_components)
            else:
                old_static = old_static.drop(duplicated_components)

        # Concatenate to new dataframe
        if not old_static.empty:
            new_static = pd.concat((old_static, new_static), sort=False)

        if cls_name == "Shape":
            new_static = gpd.GeoDataFrame(new_static, crs=self.crs)

        # Align index (component names) and columns (attributes)
        new_static = _sort_attrs(new_static, attrs.index, axis=1)

        new_static.index.name = cls_name
        self.components[cls_name].static = new_static

        # Now deal with time-dependent properties

        dynamic = self.dynamic(cls_name)

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

        self.components[cls_name].dynamic = dynamic

    def _import_series_from_df(
        self,
        df: pd.DataFrame,
        cls_name: str,
        attr: str,
        overwrite: bool = False,
    ) -> None:
        """Import time series from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame whose index is ``n.snapshots`` and
            whose columns are a subset of the relevant components.
        cls_name : string
            Name of class of component
        attr : string
            Name of time-varying series attribute
        overwrite : bool, default False
            If True, overwrite existing time series.

        """
        static = self.static(cls_name)
        dynamic = self.dynamic(cls_name)
        list_name = self.components[cls_name]["list_name"]

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
                "Components %s for attribute %s of %s are not in main components dataframe %s",
                diff,
                attr,
                cls_name,
                list_name,
            )

        # Get all attributes for the component
        attrs = self.components[cls_name]["attrs"]

        # Add all unknown attributes to the dataframe without any checks
        expected_attrs = attrs[lambda ds: ds.type.str.contains("series")].index
        if attr not in expected_attrs:
            if overwrite or attr not in dynamic:
                dynamic[attr] = df
            return

        # Check if any snapshots are missing
        diff = self.snapshots.difference(df.index)
        if len(diff):
            logger.warning(
                "Snapshots %s are missing from %s of %s. Filling with default value '%s'",
                diff,
                attr,
                cls_name,
                attrs.loc[attr].default,
            )
            df = df.reindex(self.snapshots, fill_value=attrs.loc[attr].default)

        if not attrs.loc[attr].static:
            dynamic[attr] = dynamic[attr].reindex(
                columns=df.columns.union(static.index),
                fill_value=attrs.loc[attr].default,
            )
        else:
            dynamic[attr] = dynamic[attr].reindex(
                columns=(df.columns.union(dynamic[attr].columns))
            )

        dynamic[attr].loc[self.snapshots, df.columns] = df.loc[
            self.snapshots, df.columns
        ]

    def import_from_pypower_ppc(
        self, ppc: dict, overwrite_zero_s_nom: float | None = None
    ) -> None:
        """Import network from PYPOWER PPC dictionary format version 2.

        Converts all baseMVA to base power of 1 MVA.

        For the meaning of the pypower indices, see also pypower/idx_*.

        Parameters
        ----------
        ppc : PYPOWER PPC dict
            PYPOWER PPC dictionary to import from.
        overwrite_zero_s_nom : Float or None, default None
            If a float, all branches with s_nom of 0 will be set to this value.

        Examples
        --------
        >>> from pypower.api import case30 # doctest: +SKIP
        >>> ppc = case30() # doctest: +SKIP
        >>> n.import_from_pypower_ppc(ppc) # doctest: +SKIP

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
        pdf["buses"]["control"] = (
            pdf["buses"].pop("type").map(lambda i: controls[int(i)])
        )

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

        columns = [
            "bus",
            "p_set",
            "q_set",
            "q_max",
            "q_min",
            "v_set_pu",
            "mva_base",
            "status",
            "p_nom",
            "p_min",
            "Pc1",
            "Pc2",
            "Qc1min",
            "Qc1max",
            "Qc2min",
            "Qc2max",
            "ramp_agc",
            "ramp_10",
            "ramp_30",
            "ramp_q",
            "apf",
        ]

        index_list = [f"G{str(i)}" for i in range(len(ppc["gen"]))]

        pdf["generators"] = pd.DataFrame(
            index=index_list, columns=columns, data=ppc["gen"][:, : len(columns)]
        )

        # make sure bus name is an integer
        pdf["generators"]["bus"] = np.array(ppc["gen"][:, 0], dtype=int)

        # add branchs
        ## branch data
        # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax

        columns = [
            "bus0",
            "bus1",
            "r",
            "x",
            "b",
            "s_nom",
            "rateB",
            "rateC",
            "tap_ratio",
            "phase_shift",
            "status",
            "v_ang_min",
            "v_ang_max",
        ]

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
                    "Warning: there are %d branches with s_nom equal to zero, they will probably lead to infeasibilities and should be replaced with a high value using the `overwrite_zero_s_nom` argument.",
                    zero_s_nom.sum(),
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
        pdf["transformers"].index = [
            f"T{str(i)}" for i in range(len(pdf["transformers"]))
        ]
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
            self.add(
                component,
                pdf[self.components[component]["list_name"]].index,
                **pdf[self.components[component]["list_name"]],
            )

        self.generators["control"] = self.generators.bus.map(self.buses["control"])

        # for consistency with pypower, take the v_mag set point from the generators
        self.buses.loc[self.generators.bus, "v_mag_pu_set"] = np.asarray(
            self.generators["v_set_pu"]
        )

    def import_from_pandapower_net(
        self,
        net: pandapowerNet,
        extra_line_data: bool = False,
        use_pandapower_index: bool = False,
    ) -> None:
        """Import PyPSA network from pandapower net.

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
            pandapower network to import from.
        extra_line_data : boolean, default: False
            if True, the line data for all parameters is imported instead of only the type
        use_pandapower_index : boolean, default: False
            if True, use integer numbers which is the pandapower index standard
            if False, use any net.name as index (e.g. 'Bus 1' (str) or 1 (int))

        Examples
        --------
        >>> n.import_from_pandapower_net(net) # doctest: +SKIP
        OR

        >>> import pandapower as pp # doctest: +SKIP
        >>> import pandapower.networks as pn # doctest: +SKIP
        >>> net = pn.create_cigre_network_mv(with_der='all') # doctest: +SKIP
        >>> n = pypsa.Network()
        >>> n.import_from_pandapower_net(net, extra_line_data=True)  # doctest: +SKIP

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
            self.add(component_name, d[component_name].index, **d[component_name])

        # amalgamate buses connected by closed switches

        bus_switches = net.switch[(net.switch.et == "b") & net.switch.closed]

        bus_switches["stays"] = bus_switches.bus.map(net.bus.name)
        bus_switches["goes"] = bus_switches.element.map(net.bus.name)

        to_replace = pd.Series(bus_switches.stays.values, bus_switches.goes.values)

        for i in to_replace.index:
            self.remove("Bus", i)

        for component in self.iterate_components(
            {"Load", "Generator", "ShuntImpedance"}
        ):
            component.static.replace({"bus": to_replace}, inplace=True)

        for component in self.iterate_components({"Line", "Transformer"}):
            component.static.replace({"bus0": to_replace}, inplace=True)
            component.static.replace({"bus1": to_replace}, inplace=True)
