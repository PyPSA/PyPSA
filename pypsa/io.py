"""Functions for importing and exporting data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deprecation import deprecated

from pypsa.common import deprecated_common_kwargs

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    import pandas as pd
    import xarray as xr
    from pandapower.auxiliary import pandapowerNet

    from pypsa import Network

logger = logging.getLogger(__name__)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.import_from_csv_folder` instead.",
)
@deprecated_common_kwargs
def import_from_csv_folder(
    n: Network,
    csv_folder_name: str | Path,
    encoding: str | None = None,
    quotechar: str = '"',
    skip_time: bool = False,
) -> None:
    """Import network data from CSVs in a folder."""
    n.import_from_csv_folder(csv_folder_name, encoding, quotechar, skip_time)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.export_to_csv_folder` instead.",
)
@deprecated_common_kwargs
def export_to_csv_folder(
    n: Network,
    csv_folder_name: str,
    encoding: str | None = None,
    quotechar: str = '"',
    export_standard_types: bool = False,
) -> None:
    """Export network and components to a folder of CSVs."""
    n.export_to_csv_folder(csv_folder_name, encoding, quotechar, export_standard_types)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.import_from_excel` instead.",
)
@deprecated_common_kwargs
def import_from_excel(
    n: Network,
    excel_file_path: str | Path,
    skip_time: bool = False,
    engine: str = "calamine",
) -> None:
    """Import network data from an Excel file."""
    n.import_from_excel(excel_file_path, skip_time, engine)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.export_to_excel` instead.",
)
@deprecated_common_kwargs
def export_to_excel(
    n: Network,
    excel_file_path: str | Path,
    export_standard_types: bool = False,
    engine: str = "openpyxl",
) -> None:
    """Export network and components to an Excel file."""
    n.export_to_excel(excel_file_path, export_standard_types, engine)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.import_from_hdf5` instead.",
)
@deprecated_common_kwargs
def import_from_hdf5(n: Network, path: str | Path, skip_time: bool = False) -> None:
    """Import network data from HDF5 store at `path`."""
    n.import_from_hdf5(path, skip_time)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.export_to_hdf5` instead.",
)
@deprecated_common_kwargs
def export_to_hdf5(
    n: Network,
    path: Path | str,
    export_standard_types: bool = False,
    **kwargs: Any,
) -> None:
    """Export network and components to an HDF store."""
    n.export_to_hdf5(path, export_standard_types, **kwargs)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.import_from_netcdf` instead.",
)
@deprecated_common_kwargs
def import_from_netcdf(
    n: Network, path: str | Path | xr.Dataset, skip_time: bool = False
) -> None:
    """Import network data from netCDF file or xarray Dataset at `path`."""
    n.import_from_netcdf(path, skip_time)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.export_to_netcdf` instead.",
)
@deprecated_common_kwargs
def export_to_netcdf(
    n: Network,
    path: str | None = None,
    export_standard_types: bool = False,
    compression: dict | None = None,
    float32: bool = False,
) -> None:
    """Export network and components to a netCDF file."""
    n.export_to_netcdf(path, export_standard_types, compression, float32)


@deprecated(
    deprecated_in="0.29",
    removed_in="1.0",
    details="Use `n.add` instead. E.g. `n.add(class_name, df.index, **df)`.",
)
def import_components_from_dataframe(
    n: Network, dataframe: pd.DataFrame, cls_name: str
) -> None:
    """Import components from a pandas DataFrame."""
    n.add(cls_name, dataframe.index, **dataframe)


@deprecated(
    deprecated_in="0.29",
    removed_in="1.0",
    details="Use `n.add` instead.",
)
def import_series_from_dataframe(
    n: Network, dataframe: pd.DataFrame, cls_name: str, attr: str
) -> None:
    """Import time series from a pandas DataFrame."""
    n._import_series_from_df(dataframe, cls_name, attr)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.merge` instead.",
)
@deprecated_common_kwargs
def merge(
    n: Network,
    other: Network,
    components_to_skip: Collection[str] | None = None,
    inplace: bool = False,
    with_time: bool = True,
) -> Network | None:
    """Merge the components of two networks."""
    return n.merge(other, components_to_skip, inplace, with_time)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.import_from_pypower_ppc` instead.",
)
@deprecated_common_kwargs
def import_from_pypower_ppc(
    n: Network, ppc: dict, overwrite_zero_s_nom: float | None = None
) -> None:
    """Import network from PYPOWER PPC dictionary format version 2."""
    n.import_from_pypower_ppc(ppc, overwrite_zero_s_nom)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.import_from_pandapower_net` instead.",
)
@deprecated_common_kwargs
def import_from_pandapower_net(
    n: Network,
    net: pandapowerNet,
    extra_line_data: bool = False,
    use_pandapower_index: bool = False,
) -> None:
    """Import PyPSA network from pandapower net."""
    n.import_from_pandapower_net(net, extra_line_data, use_pandapower_index)
