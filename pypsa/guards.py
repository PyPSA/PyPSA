"""Assertion guards for runtime verification of PyPSA.

Methods of this module should only be called when
pypsa.options.debug.runtime_verification is True. By default and in production,
this is False to avoid overhead. In development and testing, it can be enabled
to catch errors early.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypsa.common import UnexpectedError

if TYPE_CHECKING:
    import xarray

    from pypsa.components import Components


def _verify_xarray_data_consistency(
    component: Components, res: xarray.DataArray
) -> None:
    if component.has_scenarios and list(res.scenario.values) != list(
        component.scenarios
    ):
        msg = f"Scenario order mismatch: {list(res.scenario.values)} != {list(component.scenarios)}"
        raise UnexpectedError(msg)

    if list(res.coords["name"].values) != list(component.component_names):
        msg = f"Component order mismatch: {list(res.name.values)} != {list(component.component_names)}"
        raise UnexpectedError(msg)
