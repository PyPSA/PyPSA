import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd

from pypsa.consistency import check_for_unknown_buses
from pypsa.typing import is_1d_list_like

if TYPE_CHECKING:
    from pypsa import Components

logger = logging.getLogger(__name__)


def _sort_attrs(df: pd.DataFrame, attrs_list: Sequence, axis: int) -> pd.DataFrame:
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


def _add_static_data(
    c: "Components", df: pd.DataFrame, overwrite: bool = False
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
    # if c.name == "Link":  # TODO
    # update_linkports_component_attrs(c.n_save, where=df)

    old_static = c.static

    # Handle duplicates
    duplicated_components = old_static.index.intersection(df.index)
    if len(duplicated_components) > 0:
        if not overwrite:
            logger.warning(
                "The following %s are already defined and will be skipped "
                "(use overwrite=True to overwrite): %s",
                c.list_name,
                ", ".join(duplicated_components),
            )
            df = df.drop(duplicated_components)
        else:
            old_static = old_static.drop(duplicated_components)

    # Concatenate to new dataframe
    if not old_static.empty:
        df = pd.concat((old_static, df), sort=False)

    # Align index (component names) and columns (attributes)
    df = _sort_attrs(df, c.defaults.index, axis=1)
    df.index.name = c.name

    # Apply schema
    df = c.ctype.schema_static(df)

    # Convert to Shape Components to GeoDataFrame
    if c.name == "Shape":
        df = gpd.GeoDataFrame(df)
        if df.crs:
            df.to_crs(c.n_save.crs)
        else:
            df.crs = c.n_save.crs

    # Check that all buses are well-defined
    check_for_unknown_buses(c.n_save, c, df)

    setattr(c.n_save, c.n_save.components[c.name]["list_name"], df)


def _add_dynamic_data(
    c: "Components",
    df: pd.DataFrame,
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
    overwrite = True
    if not overwrite:
        raise NotImplementedError()
        existing = set(c.static.index) | set(
            c.dynamic.get(attr, pd.DataFrame()).columns
        )
        df = df.drop(df.columns.intersection(existing), axis=1)

    # Ignore if no data to be imported
    if df.empty:
        return

    df.columns.name = c.name
    df.index.name = "snapshot"

    # Check if components exist in static df
    diff = df.columns.difference(c.static.index)
    if len(diff) > 0:
        logger.warning(
            f"Components {diff} for attribute {attr} of {c.name} "
            f"are not in main components dataframe {c.list_name}"
        )
        # TODO

    # Add all unknown attributes to the dataframe without any checks
    expected_attrs = c.attrs[lambda ds: ds.type.str.contains("series")].index
    if attr not in expected_attrs:
        if overwrite or attr not in c.dynamic:
            c.dynamic[attr] = df
        return

    # # Check if any snapshots are missing
    # diff = c.snapshots.difference(df.index)
    # if len(diff):
    #     logger.warning(
    #         f"Snapshots {diff} are missing from {attr} of {c.name}. "
    #         f"Filling with default value '{c.attrs.loc[attr].default}'"
    #     )
    #     df = df.reindex(c.snapshots, fill_value=c.attrs.loc[attr].default)

    # Check if any components are missing

    # Apply schema # TODO Currently just checks type, default, nullable, move full
    # col/ index check to it
    df = c.ctype.schemas_input_dynamic[attr](df)

    diff = c.static.index.difference(df.columns)
    if len(diff):
        logger.warning(
            f"Components {diff} are missing from {attr} of {c.name}. "
            f"Filling with default value '{c.attrs.loc[attr].default}'"
        )
        df = df.reindex(columns=c.static.index, fill_value=c.attrs.loc[attr].default)

    if not c.attrs.loc[attr].static:
        c.dynamic[attr] = c.dynamic[attr].reindex(
            columns=df.columns.union(c.static.index),
            fill_value=c.attrs.loc[attr].default,
        )
    else:
        c.dynamic[attr] = c.dynamic[attr].reindex(
            columns=(df.columns.union(c.dynamic[attr].columns))
        )

    c.dynamic[attr].loc[c.snapshots, df.columns] = df.loc[c.snapshots, df.columns]


def add(
    c: "Components",
    name: str | int | Sequence[int | str],
    suffix: str = "",
    overwrite: bool = False,
    ignore_checks: bool = False,
    **kwargs: Any,
) -> pd.Index:
    # Process name/names to pandas.Index of strings and add suffix
    single_component = np.isscalar(name)
    names = pd.Index([name]) if single_component else pd.Index(name)

    if len(names) == 0:
        return pd.Index([])

    if not ignore_checks and names.isnull().any():
        msg = f"Names for {c.name} must not contain NaN values."
        raise ValueError(msg)

    names = names.astype(str) + suffix

    if not ignore_checks and (names.str.strip() == "").any():
        msg = f"Names for {c.name} must not be empty strings."
        raise ValueError(msg)

    names_str = "name" if single_component else "names"
    # Read kwargs into static and time-varying attributes
    series = {}
    static = {}

    # Check if names are unique
    if not ignore_checks and not names.is_unique:
        msg = f"Names for {c.name} must be unique."
        raise ValueError(msg)

    for k, v in kwargs.items():
        if not ignore_checks and k in c.defaults[c.defaults.status == "output"].index:
            msg = (
                f"Attribute '{k}' is an output attribute for {c.list_name} and "
                " cannot be set."
            )
            raise ValueError(msg)
        # If index/ columnes are passed (pd.DataFrame or pd.Series)
        # - cast names index to string and add suffix
        # - check if passed index/ columns align
        msg = "{} has an index which does not align with the passed {}."
        if isinstance(v, pd.Series) and single_component:
            if not v.index.equals(c.snapshots):
                raise ValueError(msg.format(f"Series {k}", "network snapshots"))
        elif isinstance(v, pd.Series):
            # Cast names index to string + suffix
            v = v.rename(
                index=lambda s: str(s) if str(s).endswith(suffix) else str(s) + suffix
            )
            if not v.index.equals(names):
                raise ValueError(msg.format(f"Series {k}", names_str))
        if isinstance(v, pd.DataFrame):
            # Cast names columns to string + suffix
            v = v.rename(
                columns=lambda s: str(s) if str(s).endswith(suffix) else str(s) + suffix
            )
            if not v.index.equals(c.snapshots):
                raise ValueError(msg.format(f"DataFrame {k}", "network snapshots"))
            if not v.columns.equals(names):
                if not all(col in names for col in v.columns):
                    raise ValueError(
                        msg.format(
                            f"DataFrame {k}. It must at least be a subset.", names_str
                        )
                    )
                v = v.reindex(columns=names).fillna(
                    float(c.attrs.default[k]),
                )  # TODO generalise

        # Convert list-like and 1-dim array to pandas.Series
        if is_1d_list_like(v):
            try:
                if single_component:
                    v = pd.Series(v, index=c.snapshots)
                else:
                    v = pd.Series(v)
                    if len(v) == 1:
                        v = v.iloc[0]
                        logger.debug(
                            f"Single value sequence for {k} is treated as a scalar "
                            f"and broadcasted to all components. It is recommended "
                            f"to explicitly pass a scalar instead."
                        )
                    else:
                        v.index = names
            except ValueError:
                expec_str = (
                    f"{len(c.snapshots)} for each snapshot."
                    if single_component
                    else f"{len(names)} for each component name."
                )
                msg = f"Data for {k} has length {len(v)} but expected {expec_str}"
                raise ValueError(msg)
        # Convert 2-dim array to pandas.DataFrame
        if isinstance(v, np.ndarray):
            if v.shape == (len(c.snapshots), len(names)):
                v = pd.DataFrame(v, index=c.snapshots, columns=names)
            else:
                msg = (
                    f"Array {k} has shape {v.shape} but expected "
                    f"({len(c.snapshots)}, {len(names)})."
                )
                raise ValueError(msg)

        if isinstance(v, dict):
            msg = (
                "Dictionaries are not supported as attribute values. Please use "
                "pandas.Series or pandas.DataFrame instead."
            )
            raise NotImplementedError(msg)

        # Handle addition of single component
        if single_component:
            # Read 1-dim data as time-varying attribute
            if isinstance(v, pd.Series):
                series[k] = pd.DataFrame(v.values, index=c.snapshots, columns=names)
            # Read 0-dim data as static attribute
            else:
                static[k] = v

        # Handle addition of multiple components
        elif not single_component:
            # Read 2-dim data as time-varying attribute
            if isinstance(v, pd.DataFrame):
                series[k] = v
            # Read 1-dim data as static attribute
            elif isinstance(v, pd.Series):
                static[k] = v.values
            # Read scalar data as static attribute
            else:
                static[k] = v

    # Load static attributes as components
    if static:
        static_df = pd.DataFrame(static, index=names)
    else:
        static_df = pd.DataFrame(index=names)

    # Cast existing static attribiutes to dynamic
    needs_upscaling = list(set(series.keys()) & set(c.static.columns))
    if needs_upscaling:
        for attr in needs_upscaling:
            logger.info(
                "Transforming static attribute %s from static to dynamic since "
                " a component with dynamic data is being added.",
                attr,
            )
            series[attr][c.static[attr].index] = c.static[attr].values
            c.static = c.static.drop(columns=attr)
        c.ctype.schema_static = c.ctype.schema_static.remove_columns(
            list(needs_upscaling)
        )

    if set(static_df.columns) & set(c.dynamic.keys()):
        msg = "Transform mechansim c.dynamic to c.static missing"
        raise NotImplementedError(msg)

    _add_static_data(c, static_df, overwrite=overwrite)

    # Load time-varying attributes as components
    for attr, v in series.items():
        _add_dynamic_data(c, v, attr, overwrite=overwrite)

    assert not set(c.dynamic.keys()) & set(
        c.static.columns
    ), f"Attributes are not exclusive: {c.dynamic.keys() & c.static.columns}"

    return names
