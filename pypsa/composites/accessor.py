# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Network accessor for composite components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from linopy import options as linopy_options

from pypsa.composites.definition import (
    PARAM_DTYPES,
    CompositeDefinition,
    member_name,
)
from pypsa.statistics.grouping import groupers

if TYPE_CHECKING:
    from collections.abc import Iterator
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

    def add(self, name: str, **params: Any) -> str:
        """Add one instance, creating every member component.

        Parameters
        ----------
        name : str
            Instance name. Member components are named ``<name>-<member>``.
        **params
            Exposed parameters of the definition. Scalars or time series.

        """
        if name in self.instances:
            msg = f"Composite '{self.name}' already holds an instance '{name}'."
            raise ValueError(msg)
        definition = self.definition
        values = definition.values(params)
        stored = {
            f"{PARAM_COL_PREFIX}{k}": v
            for k, v in values.items()
            if type(v) in PARAM_DTYPES
        }
        for member, attrs in definition.resolve(name, values).items():
            if member == definition.primary_member:
                attrs.update(stored)
            attrs.update({INSTANCE_COL: name, TYPE_COL: self.name})
            self._n.add(definition.members[member], member_name(name, member), **attrs)
        return name

    def remove(self, name: str) -> None:
        """Remove one instance and all its member components."""
        rows = self.members.query("instance == @name")
        if rows.empty:
            msg = f"Composite '{self.name}' has no instance '{name}'."
            raise ValueError(msg)
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
        cls = self.definition.bound_class
        return {
            member: group.set_index("component")["instance"]
            .rename_axis("name")
            .rename(member)
            for member, group in members[members["class"] == cls].groupby("member")
        }

    def _parameters(self) -> dict[str, pd.Series]:
        definition = self.definition
        members = self.members
        primary = members[members["member"] == definition.primary_member]
        static = self._n.c[definition.members[definition.primary_member]].static
        rows = static.loc[primary["component"]].set_index(INSTANCE_COL)
        out = {}
        for key, decl in definition.parameter_decls().items():
            col = f"{PARAM_COL_PREFIX}{key}"
            series = rows[col] if col in rows.columns else pd.Series(index=rows.index)
            series = series.fillna(definition.parameters[key]).rename(key)
            series = series.astype(PANDAS_DTYPES[decl["dtype"]])
            out[key] = series.rename_axis(self.name).reindex(self.instances)
        return out

    def sources(self, model: Model) -> dict[str, Any]:
        """Build the ``sources`` mapping for ``Model.add_spec``."""
        cls = self.definition.bound_class
        sources: dict[str, Any] = {
            self.name: self.instances,
            **self._lookups(),
            **self._parameters(),
        }
        if cls is not None:
            sources["name"] = self._n.c[cls].active_assets
            for var in self.definition.math["variables"]:
                sources[var] = model.variables[var.replace("_", "-", 1)]
        return sources

    def add_spec(self, model: Model) -> None:
        """Layer the math fragment onto ``model``."""
        if not self.definition.math or self.instances.empty:
            return
        model.add_spec(
            self.definition.spec_text(), self.sources(model), name=self.layer
        )

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
        for composite in active:
            composite.add_spec(model)


def composite_grouper(
    n: Network, c: str, port: str = "", nice_names: bool = True
) -> pd.Series:
    """Group components by composite instance; components outside any composite are dropped."""
    static = n.c[c].static
    if INSTANCE_COL not in static.columns:
        return pd.Series(pd.NA, index=static.index, name=INSTANCE_COL, dtype=object)
    return static[INSTANCE_COL].rename(INSTANCE_COL)


groupers.add_grouper(INSTANCE_COL, composite_grouper)


def _to_pandas(da: xr.DataArray) -> pd.Series | pd.DataFrame:
    if "snapshot" in da.dims and da.ndim == 2:
        return da.transpose("snapshot", ...).to_pandas()
    return da.to_pandas()
