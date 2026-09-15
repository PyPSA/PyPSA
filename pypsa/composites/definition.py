# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Composite definitions: a recipe of fundamental components plus a math-spec fragment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pypsa.components import types as component_types

if TYPE_CHECKING:
    from collections.abc import Collection

PARAM_PREFIX = "$"
MEMBER_PREFIX = "@"
MATH_KEYS = frozenset({"variables", "expressions", "constraints"})
PARAM_DTYPES = {bool: "bool", int: "float", float: "float", str: "str"}


class CompositeDefinition(BaseModel):
    """A reusable recipe of fundamental components.

    A pydantic model. Subclass it to define a composite in Python, or build it
    from a YAML file or dict with ``from_yaml`` and ``from_dict``.

    Parameters
    ----------
    name : str
        Definition name, also the instance dimension in the math fragment.
    parameters : dict
        Exposed parameters with defaults. A ``None`` default marks a required
        parameter of ``add``.
    components : dict
        ``{class_name: {member: {attr: value}}}``. Values starting with ``$``
        reference an exposed parameter, values starting with ``@`` reference a
        member component of the same instance.
    math : dict
        math-spec fragment with keys ``variables``, ``expressions`` and
        ``constraints``. Variables bind PyPSA model variables, named
        ``<Component>_<attr>``.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    components: dict[str, dict[str, dict[str, Any]]]
    parameters: dict[str, Any] = Field(default_factory=dict)
    math: dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    @model_validator(mode="after")
    def _validate(self) -> CompositeDefinition:
        members = self.members
        if len(members) != sum(len(m) for m in self.components.values()):
            msg = (
                f"Composite '{self.name}': member names must be unique across classes."
            )
            raise ValueError(msg)
        for cls in self.components:
            component_types.get(cls)
        for member, attrs in self._member_attrs():
            for attr, value in attrs.items():
                ref = _reference(value)
                if ref is None:
                    continue
                prefix, target = ref
                pool = self.parameters if prefix == PARAM_PREFIX else members
                if target not in pool:
                    msg = (
                        f"Composite '{self.name}', member '{member}', attribute "
                        f"'{attr}': unknown reference '{value}'."
                    )
                    raise ValueError(msg)
        unknown = set(self.math) - MATH_KEYS
        if unknown:
            msg = f"Composite '{self.name}': unsupported math keys {sorted(unknown)}."
            raise ValueError(msg)
        for name, decl in self.math.get("variables", {}).items():
            if not isinstance(decl, dict):
                msg = (
                    f"Composite '{self.name}': bound variable '{name}' must be a "
                    "mapping declaring 'foreach'."
                )
                raise ValueError(msg)  # noqa: TRY004
            extra = set(decl) - {"foreach"}
            if extra:
                msg = (
                    f"Composite '{self.name}': bound variable '{name}' may only "
                    f"declare 'foreach', found {sorted(extra)}."
                )
                raise ValueError(msg)
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompositeDefinition:
        """Build a definition from a mapping with the YAML layout."""
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, source: str | Path) -> CompositeDefinition:
        """Load a definition from a YAML file path or YAML text."""
        import yaml  # noqa: PLC0415

        text = (
            source
            if isinstance(source, str) and "\n" in source
            else Path(source).read_text()
        )
        return cls.from_dict(yaml.safe_load(text))

    @property
    def members(self) -> dict[str, str]:
        """Member name mapped to its component class."""
        return {m: cls for cls, members in self.components.items() for m in members}

    @property
    def required(self) -> list[str]:
        """Exposed parameters without a default."""
        return [k for k, v in self.parameters.items() if v is None]

    @property
    def bound_classes(self) -> list[str]:
        """Component classes whose model variables the math binds, in declaration order."""
        return list(
            dict.fromkeys(v.split("_", 1)[0] for v in self.math.get("variables", {}))
        )

    def _member_attrs(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (m, a) for members in self.components.values() for m, a in members.items()
        ]

    def values(self, params: dict[str, Any]) -> dict[str, Any]:
        """Exposed parameter values for one instance, defaults filled in."""
        missing = [k for k in self.required if k not in params]
        if missing:
            msg = f"Composite '{self.name}': missing required parameters {missing}."
            raise ValueError(msg)
        unknown = set(params) - set(self.parameters)
        if unknown:
            msg = f"Composite '{self.name}': unknown parameters {sorted(unknown)}."
            raise ValueError(msg)
        return {**self.parameters, **params}

    @property
    def primary_member(self) -> str:
        """The member that carries the instance parameters, first of a bound class."""
        bound = self.bound_classes
        return next(m for m, c in self.members.items() if not bound or c in bound)

    def resolve(
        self, instance: str | pd.Index, values: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Component attributes per member for one instance, references resolved."""
        resolved = {}
        for member, attrs in self._member_attrs():
            resolved[member] = {
                attr: _resolve_value(value, values, instance)
                for attr, value in attrs.items()
            }
        return resolved

    def spec(self, dynamic: Collection[str] = ()) -> dict[str, Any]:
        """Expand the fragment into a complete math-spec dict for ``add_spec``.

        Parameters in ``dynamic`` are declared over ``[snapshot, <name>]``
        instead of ``[<name>]``.
        """
        bound = self.bound_classes
        dims: dict[str, Any] = {
            "snapshot": {"dtype": "datetime"},
            self.name: {"dtype": "str"},
        }
        if bound:
            dims["name"] = {"dtype": "str"}
        lookups = {
            member: {"over": "name", "into": self.name}
            for member, mcls in self.members.items()
            if mcls in bound
        }
        return {
            "description": self.description or f"composite '{self.name}'",
            "dimensions": dims,
            "lookups": lookups,
            "parameters": self.parameter_decls(dynamic),
            **{
                key: self.math.get(key, {})
                for key in ("variables", "expressions", "constraints")
            },
        }

    def parameter_decls(
        self, dynamic: Collection[str] = ()
    ) -> dict[str, dict[str, Any]]:
        """Spec declarations for exposed parameters used by the math."""
        used = set(re.findall(r"[A-Za-z_]\w*", " ".join(self._math_bodies())))
        return {
            key: {
                "dims": ["snapshot", self.name] if key in dynamic else [self.name],
                "dtype": PARAM_DTYPES[type(value)],
            }
            for key, value in self.parameters.items()
            if type(value) in PARAM_DTYPES and key in used
        }

    def _math_bodies(self) -> list[str]:
        bodies = []
        for key in ("expressions", "constraints"):
            for decl in self.math.get(key, {}).values():
                if isinstance(decl, str):
                    bodies.append(decl)
                else:
                    bodies.extend(str(decl.get(k, "")) for k in ("expression", "where"))
        return bodies

    def spec_text(self, dynamic: Collection[str] = ()) -> str:
        """Dump the expanded spec as YAML text."""
        import yaml  # noqa: PLC0415

        return yaml.safe_dump(self.spec(dynamic), sort_keys=False)


def member_name(instance: str | pd.Index, member: str) -> str | pd.Index:
    """Component name of a member inside one instance or an index of instances."""
    if isinstance(instance, pd.Index):
        return instance.astype(str) + f"-{member}"
    return f"{instance}-{member}"


def _reference(value: Any) -> tuple[str, str] | None:
    if isinstance(value, str) and value[:1] in (PARAM_PREFIX, MEMBER_PREFIX):
        return value[0], value[1:]
    return None


def _resolve_value(value: Any, params: dict[str, Any], instance: str | pd.Index) -> Any:
    ref = _reference(value)
    if ref is None:
        return value
    prefix, target = ref
    if prefix == PARAM_PREFIX:
        return params[target]
    return member_name(instance, target)
