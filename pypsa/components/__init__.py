"""Package for component specific functionality in PyPSA."""

from typing import Any

from pypsa.components.abstract import Components
from pypsa.components.components_ import Component, Generators, GenericComponents
from pypsa.components.store import ComponentsStore
from pypsa.components.types import ComponentType, ComponentTypeEnum


def __getattr__(name: str) -> Any:
    if name in ["Network", "SubNetwork"]:
        msg = (
            f"Cannot import '{name}' from 'pypsa.components'. "
            "Import it from 'pypsa' instead."
        )
        raise ImportError(msg)
    return getattr(Components, name)


__all__ = [
    "ComponentType",
    "ComponentTypeEnum",
    "ComponentsStore",
    "Components",
    "GenericComponents",
    "Generators",
    "Component",
]
