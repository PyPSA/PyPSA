"""Power flow functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deprecation import deprecated

from pypsa.common import deprecated_common_kwargs
from pypsa.network.power_flow import logger  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from components import Network, SubNetwork

    from pypsa.definitions.structures import Dict


def zsum(s: pd.Series, *args: Any, **kwargs: Any) -> Any:
    """Define a custom zsum function.

    Pandas 0.21.0 changes sum() behavior so that the result of applying sum
    over an empty DataFrame is NaN.

    Meant to be set as pd.Series.zsum = zsum.
    """
    msg = "`zsum` was deprecated in pypsa 0.35.0."
    raise DeprecationWarning(msg)


def normed(s: pd.Series) -> pd.Series:
    msg = "`normed` was deprecated in pypsa 0.35.0."
    raise DeprecationWarning(msg)


def real(X: pd.Series) -> pd.Series:
    msg = "`real` was deprecated in pypsa 0.35.0."
    raise DeprecationWarning(msg)


def imag(X: pd.Series) -> pd.Series:
    msg = "`imag` was deprecated in pypsa 0.35.0."
    raise DeprecationWarning(msg)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `pypsa.network.power_flow.allocate_series_dataframes` instead.",
)
@deprecated_common_kwargs
def allocate_series_dataframes(*args: Any, **kwargs: Any) -> Any:
    """Use `pypsa.network.power_flow.allocate_series_dataframes` instead."""
    from pypsa.network.power_flow import allocate_series_dataframes

    return allocate_series_dataframes(*args, **kwargs)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.pf` instead.",
)
@deprecated_common_kwargs
def network_pf(
    n: Network,
    snapshots: Sequence | None = None,
    skip_pre: bool = False,
    x_tol: float = 1e-6,
    use_seed: bool = False,
    distribute_slack: bool = False,
    slack_weights: str = "p_set",
) -> Dict:
    """Use `n.pf` instead."""
    return n.pf(
        snapshots=snapshots,
        skip_pre=skip_pre,
        x_tol=x_tol,
        use_seed=use_seed,
        distribute_slack=distribute_slack,
        slack_weights=slack_weights,
    )


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `pypsa.network.power_flow.newton_raphson_sparse` instead.",
)
@deprecated_common_kwargs
def newton_raphson_sparse(*args: Any, **kwargs: Any) -> Any:
    """Use `pypsa.network.power_flow.newton_raphson_sparse` instead."""
    from pypsa.network.power_flow import newton_raphson_sparse

    return newton_raphson_sparse(*args, **kwargs)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `pypsa.network.power_flow.sub_network_pf_singlebus` instead.",
)
@deprecated_common_kwargs
def sub_network_pf_singlebus(*args: Any, **kwargs: Any) -> Any:
    """Use `pypsa.network.power_flow.sub_network_pf_singlebus` instead."""
    from pypsa.network.power_flow import sub_network_pf_singlebus

    return sub_network_pf_singlebus(*args, **kwargs)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.pf` instead.",
)
@deprecated_common_kwargs
def sub_network_pf(
    sub_network: SubNetwork, *args: Any, **kwargs: Any
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Use `sub_network.pf` instead."""
    return sub_network.pf(*args, **kwargs)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.lpf` instead.",
)
@deprecated_common_kwargs
def network_lpf(n: Network, *args: Any, **kwargs: Any) -> Any:
    """Use `n.lpf` instead."""
    return n.lpf(*args, **kwargs)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `pypsa.network.power_flow.apply_line_types` instead.",
)
@deprecated_common_kwargs
def apply_line_types(n: Network) -> None:
    """Use `pypsa.network.power_flow.apply_line_types` instead."""
    from pypsa.network.power_flow import apply_line_types

    return apply_line_types(n)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `pypsa.network.power_flow.apply_transformer_types` instead.",
)
@deprecated_common_kwargs
def apply_transformer_types(n: Network) -> None:
    """Use `pypsa.network.power_flow.apply_transformer_types` instead."""
    from pypsa.network.power_flow import apply_transformer_types

    return apply_transformer_types(n)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `pypsa.network.power_flow.wye_to_delta` instead.",
)
@deprecated_common_kwargs
def wye_to_delta(*args: Any, **kwargs: Any) -> tuple[float, float, float]:
    """Use `pypsa.network.power_flow.wye_to_delta` instead."""
    from pypsa.network.power_flow import wye_to_delta

    return wye_to_delta(*args, **kwargs)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `pypsa.network.power_flow.apply_transformer_t_model` instead.",
)
@deprecated_common_kwargs
def apply_transformer_t_model(n: Network) -> None:
    """Use `pypsa.network.power_flow.apply_transformer_t_model` instead."""
    from pypsa.network.power_flow import apply_transformer_t_model

    return apply_transformer_t_model(n)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `n.calculate_dependent_values` instead.",
)
@deprecated_common_kwargs
def calculate_dependent_values(n: Network) -> None:
    """Use `n.calculate_dependent_values` instead."""
    return n.calculate_dependent_values()


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.find_slack_bus` instead.",
)
@deprecated_common_kwargs
def find_slack_bus(sub_network: SubNetwork) -> None:
    """Use `sub_network.find_slack_bus` instead."""
    return sub_network.find_slack_bus()


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.find_bus_controls` instead.",
)
@deprecated_common_kwargs
def find_bus_controls(sub_network: SubNetwork) -> None:
    """Use `sub_network.find_bus_controls` instead."""
    return sub_network.find_bus_controls()


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.calculate_B_H` instead.",
)
@deprecated_common_kwargs
def calculate_B_H(sub_network: SubNetwork, skip_pre: bool = False) -> None:
    """Use `sub_network.calculate_B_H` instead."""
    return sub_network.calculate_B_H(skip_pre)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.calculate_PTDF` instead.",
)
@deprecated_common_kwargs
def calculate_PTDF(sub_network: SubNetwork, skip_pre: bool = False) -> None:
    """Use `sub_network.calculate_PTDF` instead."""
    return sub_network.calculate_PTDF(skip_pre)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.calculate_Y` instead.",
)
@deprecated_common_kwargs
def calculate_Y(
    sub_network: SubNetwork, skip_pre: bool = False, active_branches_only: bool = True
) -> None:
    """Use `sub_network.calculate_Y` instead."""
    return sub_network.calculate_Y(skip_pre, active_branches_only)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.calculate_Y_bus` instead.",
)
@deprecated_common_kwargs
def calculate_Y_bus(
    sub_network: SubNetwork, skip_pre: bool = False, active_branches_only: bool = True
) -> None:
    """Use `sub_network.calculate_Y_bus` instead."""
    return sub_network.calculate_Y_bus(skip_pre, active_branches_only)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.aggregate_multi_graph` instead.",
)
def aggregate_multi_graph(sub_network: SubNetwork) -> None:
    """Use `sub_network.aggregate_multi_graph` instead."""
    return sub_network.aggregate_multi_graph()


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.find_tree` instead.",
)
def find_tree(sub_network: SubNetwork, weight: str = "x_pu") -> None:
    """Use `sub_network.find_tree` instead."""
    return sub_network.find_tree(weight=weight)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.find_cycles` instead.",
)
@deprecated_common_kwargs
def find_cycles(sub_network: SubNetwork, weight: str = "x_pu") -> None:
    """Use `sub_network.find_cycles` instead."""
    return sub_network.find_cycles(weight=weight)


@deprecated(
    deprecated_in="0.35.0",
    removed_in="1.0.0",
    details="Use `sub_network.lpf` instead.",
)
@deprecated_common_kwargs
def sub_network_lpf(
    sub_network: SubNetwork,
    snapshots: Sequence | None = None,
    skip_pre: bool = False,
) -> None:
    """Use `sub_network.lpf` instead."""
    return sub_network.lpf(snapshots=snapshots, skip_pre=skip_pre)
