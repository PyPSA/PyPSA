from dataclasses import dataclass
from typing import Any

import pandas as pd
from deprecation import deprecated


@dataclass(frozen=True)
class ComponentType:
    name: str
    list_name: str
    description: str
    category: str
    defaults: pd.DataFrame

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ComponentType):
            return NotImplemented

        return (
            self.name == other.name
            and self.list_name == other.list_name
            and self.description == other.description
            and str(self.category) == str(other.category)
            and self.defaults.equals(other.defaults)
        )

    @property
    @deprecated(
        deprecated_in="0.32.0",
        details="Use the 'category' attribute instead.",
    )
    def type(self) -> str:
        return self.category

    @property
    @deprecated(
        deprecated_in="0.32.0",
        details="Use the 'defaults' attribute instead.",
    )
    def attrs(self) -> pd.DataFrame:
        return self.defaults
