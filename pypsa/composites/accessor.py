# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Network accessor for composite components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from linopy import options as linopy_options

from pypsa._options import options
from pypsa.composites.definition import (
    PARAM_DTYPES,
    CompositeDefinition,
    member_name,
)
from pypsa.deprecations import COMPONENT_ALIAS_DICT
from pypsa.network.transform import _build_suffixed_names
from pypsa.statistics.grouping import groupers

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    import xarray as xr
    from linopy import Model

    from pypsa.networks import Network

INSTANCE_COL = "composite"
TYPE_COL = "composite_type"
PARAM_COL_PREFIX = "composite_param_"
PANDAS_DTYPES = {"bool": bool, "float": float, "str": object}


class Composite:
    """Handle on one registered composite definition within a network."""

    def __init__(self, n: Network, definition: CompositeDefinition) -> None:
        """Bind a definition to a network."""
        self._n = n
        self.definition = definition

    @property
    def name(self) -> str:
        """Definition name."""
        return self.definition.name

    @property
    def layer(self) -> str:
        """Name of the math-spec layer on ``n.model``."""
        return f"composite-{self.name}"

    def __repr__(self) -> str:
        """Summarise members and instance count."""
        return (
            f"Composite '{self.name}' with members {list(self.definition.members)} "
            f"and {len(self.instances)} instance(s)"
        )

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str | Sequence[str] = "",
        overwrite: bool = False,
        return_names: bool | None = None,
        **params: Any,
    ) -> pd.Index | None:
        """Add instances, creating every member component.

        Follows the calling convention of ``Network.add``: a single name treats
        non-scalar parameters as time series, a list of names treats them as
        per-instance values (2D for time series per instance).

        Parameters
        ----------
        name : str or list of str
            Instance name(s). Member components are named ``<name>-<member>``.
        suffix : str or list of str, default ""
            Suffix added to each name.
        overwrite : bool, default False
            Overwrite existing instances instead of raising.
        return_names : bool | None, default None
            Return the instance names. Defaults to the module wide option.
        **params
            Exposed parameters of the definition.

        """
        if return_names is None:
            return_names = options.params.add.return_names
        names = _build_suffixed_names(name, suffix)
        single = np.isscalar(name) and isinstance(suffix, str)
        existing = names.intersection(self.instances)
        if not existing.empty:
            if not overwrite:
                msg = (
                    f"Composite '{self.name}' already holds instances "
                    f"{existing.tolist()}. Pass overwrite=True to replace them."
                )
                raise ValueError(msg)
            self.remove(existing)
        definition = self.definition
        values = definition.values(params)
        instance = names[0] if single else names
        stored = {
            f"{PARAM_COL_PREFIX}{k}": v
            for k, v in values.items()
            if type(definition.parameters[k]) in PARAM_DTYPES
        }
        for member, attrs in definition.resolve(instance, values).items():
            if member == definition.primary_member:
                attrs.update(stored)
            attrs.update({INSTANCE_COL: instance, TYPE_COL: self.name})
            components = member_name(instance, member)
            if not single:
                attrs = {k: _relabel(v, names, components) for k, v in attrs.items()}
            self._n.add(
                definition.members[member], components, overwrite=overwrite, **attrs
            )
        return names if return_names else None

    def remove(
        self, name: str | int | Sequence[int | str], suffix: str | Sequence[str] = ""
    ) -> None:
        """Remove instances and all their member components."""
        names = _build_suffixed_names(name, suffix)
        missing = names.difference(self.instances)
        if not missing.empty:
            msg = f"Composite '{self.name}' has no instances {missing.tolist()}."
            raise ValueError(msg)
        rows = self.members[self.members["instance"].isin(names)]
        for cls, group in rows.groupby("class"):
            self._n.remove(cls, group["component"].tolist())

    @property
    def members(self) -> pd.DataFrame:
        """All member components of all instances."""
        frames = []
        for cls in self.definition.components:
            static = self._n.c[cls].static
            if TYPE_COL not in static.columns:
                continue
            rows = static[static[TYPE_COL] == self.name]
            instances = rows[INSTANCE_COL].to_numpy()
            frames.append(
                pd.DataFrame(
                    {
                        "instance": instances,
                        "member": [
                            c[len(i) + 1 :]
                            for c, i in zip(rows.index, instances, strict=True)
                        ],
                        "class": cls,
                        "component": rows.index.to_numpy(),
                    }
                )
            )
        columns = ["instance", "member", "class", "component"]
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)[columns]

    @property
    def instances(self) -> pd.Index:
        """Names of the added instances."""
        return pd.Index(self.members["instance"].unique(), name=self.name)

    def _lookups(self) -> dict[str, pd.Series]:
        members = self.members
        bound = members[members["class"].isin(self.definition.bound_classes)]
        return {
            member: group.set_index("component")["instance"]
            .rename_axis("name")
            .rename(member)
            for member, group in bound.groupby("member")
        }

    def _parameters(self) -> dict[str, pd.Series | pd.DataFrame]:
        """Math parameters per instance, as a DataFrame over snapshots where dynamic."""
        definition = self.definition
        members = self.members
        primary = members[members["member"] == definition.primary_member]
        components = primary.set_index("instance")["component"].reindex(self.instances)
        c = self._n.c[definition.members[definition.primary_member]]
        out: dict[str, pd.Series | pd.DataFrame] = {}
        for key, decl in definition.parameter_decls().items():
            col = f"{PARAM_COL_PREFIX}{key}"
            dtype = PANDAS_DTYPES[decl["dtype"]]
            static = c.static.reindex(index=components, columns=[col])[col]
            series = static.set_axis(self.instances).fillna(definition.parameters[key])
            series = series.astype(dtype).rename(key)
            dynamic = c.dynamic.get(col, pd.DataFrame())
            varying = components[components.isin(dynamic.columns)]
            if varying.empty:
                out[key] = series
                continue
            if self._n.has_investment_periods:
                msg = (
                    f"Composite '{self.name}': dynamic parameter '{key}' is "
                    "not supported on networks with investment periods."
                )
                raise NotImplementedError(msg)
            frame = pd.DataFrame(
                [series] * len(self._n.snapshots), index=self._n.snapshots
            )
            frame.update(dynamic[varying.to_numpy()].set_axis(varying.index, axis=1))
            out[key] = frame.astype(dtype)
        return out

    def sources(self, model: Model, names: pd.Index) -> dict[str, Any]:
        """Build the ``sources`` mapping for ``Model.add_spec``.

        ``names`` is the component axis shared by every composite layer on
        ``model``: the union of the assets of all bound classes.
        """
        parameters = self._parameters()
        sources: dict[str, Any] = {
            self.name: self.instances,
            **self._lookups(),
            **parameters,
        }
        if any(isinstance(v, pd.DataFrame) for v in parameters.values()):
            sources["snapshot"] = self._n.snapshots
        if self.definition.bound_classes:
            sources["name"] = names
            for var in self.definition.math["variables"]:
                sources[var] = model.variables[var.replace("_", "-", 1)]
        return sources

    def add_spec(self, model: Model, names: pd.Index) -> None:
        """Layer the math fragment onto ``model``, binding variables over ``names``."""
        sources = self.sources(model, names)
        dynamic = [k for k, v in sources.items() if isinstance(v, pd.DataFrame)]
        model.add_spec(self.definition.spec_text(dynamic), sources, name=self.layer)

    @property
    def expressions(self) -> dict[str, pd.Series | pd.DataFrame]:
        """Solved output expressions of the math fragment, as pandas objects."""
        layer = self._n.model.spec[self.layer]
        return {
            name: _to_pandas(expr.solution) for name, expr in layer.expressions.items()
        }


class CompositesAccessor:
    """Registry of composite definitions on a network, exposed as ``n.composites``."""

    def __init__(self, n: Network) -> None:
        """Create an empty registry for ``n``."""
        self._n = n
        self._definitions: dict[str, Composite] = {}

    def register(
        self, source: str | Path | dict[str, Any] | CompositeDefinition
    ) -> Composite:
        """Register a definition from a YAML path, YAML text, dict or definition."""
        if isinstance(source, CompositeDefinition):
            definition = source
        elif isinstance(source, dict):
            definition = CompositeDefinition.from_dict(source)
        else:
            definition = CompositeDefinition.from_yaml(source)
        if definition.name in self._definitions:
            msg = f"Composite '{definition.name}' is already registered."
            raise ValueError(msg)
        if definition.name in COMPONENT_ALIAS_DICT.keys() | set(
            COMPONENT_ALIAS_DICT.values()
        ):
            msg = f"Composite '{definition.name}' clashes with a component class name."
            raise ValueError(msg)
        composite = Composite(self._n, definition)
        self._definitions[definition.name] = composite
        return composite

    def __getitem__(self, name: str) -> Composite:
        """Return the composite registered under ``name``."""
        return self._definitions[name]

    def __getattr__(self, name: str) -> Composite:
        """Return the composite registered under ``name`` as an attribute."""
        if name.startswith("_") or name not in self._definitions:
            raise AttributeError(name)
        return self._definitions[name]

    def __iter__(self) -> Iterator[Composite]:
        """Iterate over registered composites."""
        return iter(self._definitions.values())

    def __len__(self) -> int:
        """Return the number of registered composites."""
        return len(self._definitions)

    def __contains__(self, name: str) -> bool:
        """Return whether a composite ``name`` is registered."""
        return name in self._definitions

    def __repr__(self) -> str:
        """List registered composites."""
        if not self._definitions:
            return "Composites: none registered"
        return "Composites:\n" + "\n".join(f" - {c!r}" for c in self)

    def _add_spec_layers(self, model: Model) -> None:
        active = [c for c in self if c.definition.math and not c.instances.empty]
        if not active:
            return
        if linopy_options["semantics"] != "v1":
            msg = (
                "Composite math requires linopy's v1 semantics. Set "
                "linopy.options['semantics'] = 'v1' before optimizing."
            )
            raise ValueError(msg)
        names = self._component_axis(active)
        for composite in active:
            composite.add_spec(model, names)

    def _component_axis(self, active: list[Composite]) -> pd.Index:
        """One ``name`` axis for all layers: the assets of every bound class, concatenated."""
        classes = dict.fromkeys(c for a in active for c in a.definition.bound_classes)
        assets = [self._n.c[cls].active_assets for cls in classes]
        names = pd.Index([], dtype=object, name="name").append(assets)
        if names.has_duplicates:
            duplicated = names[names.duplicated()].unique().tolist()
            msg = (
                f"Component names {duplicated} occur in several of the classes "
                f"{list(classes)} bound by composite math and must be unique across classes."
            )
            raise ValueError(msg)
        return names


def composite_grouper(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> pd.Series:
    """Group components by composite instance; components outside any composite are dropped."""
    static = n.c[c].static
    if INSTANCE_COL not in static.columns:
        return pd.Series(pd.NA, index=static.index, name=INSTANCE_COL, dtype=object)
    return static[INSTANCE_COL].rename(INSTANCE_COL)


groupers.add_grouper(INSTANCE_COL, composite_grouper)


def _relabel(value: Any, instances: pd.Index, components: pd.Index) -> Any:
    if isinstance(value, pd.DataFrame) and value.columns.equals(instances):
        return value.set_axis(components, axis=1)
    if isinstance(value, pd.Series) and value.index.equals(instances):
        return value.set_axis(components)
    return value


def _to_pandas(da: xr.DataArray) -> pd.Series | pd.DataFrame:
    if "snapshot" in da.dims and da.ndim == 2:
        return da.transpose("snapshot", ...).to_pandas()
    return da.to_pandas()
