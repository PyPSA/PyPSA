"""Package for component specific functionality in PyPSA."""

from typing import Any

from pypsa.components._types import (
    Buses,
    Carriers,
    Generators,
    GlobalConstraints,
    Lines,
    LineTypes,
    Links,
    Loads,
    Shapes,
    ShuntImpedances,
    StorageUnits,
    Stores,
    SubNetworks,
    Transformers,
    TransformerTypes,
)
from pypsa.components.components import Components
from pypsa.components.legacy import Component


def __getattr__(name: str) -> Any:
    if name in ["Network", "SubNetwork"]:
        msg = (
            f"Cannot import '{name}' from 'pypsa.components'. "
            "Import it from 'pypsa' instead."
        )
        raise ImportError(msg)
    return getattr(Components, name)


__all__ = [
    "Component",
    "Components",
    "Buses",
    "Carriers",
    "Generators",
    "GlobalConstraints",
    "LineTypes",
    "Lines",
    "Links",
    "Loads",
    "Shapes",
    "SubNetworks",
    "ShuntImpedances",
    "StorageUnits",
    "Stores",
    "TransformerTypes",
    "Transformers",
]
