from collections.abc import Callable
from functools import partial, wraps
from inspect import signature
from typing import TYPE_CHECKING, Any

from pypsa.statistics.expressions import StatisticsAccessor

if TYPE_CHECKING:
    from pypsa import Network


# Dynamically generate plot methods configuration by inspecting StatisticsAccessor
def _generate_plot_methods() -> dict[str, dict]:
    """
    Dynamically generate plot methods configuration by inspecting StatisticsAccessor.

    Returns
    -------
    dict
        Dictionary of method names mapped to their configuration
    """
    methods = {}

    for name in dir(StatisticsAccessor):
        if name.startswith("_"):
            continue

        method = getattr(StatisticsAccessor, name)

        if not callable(method):
            continue

        sig = signature(method)
        supports_snapshot = "aggregate_time" in sig.parameters

        special_params = []
        if "storage" in sig.parameters:
            special_params.append("storage")

        methods[name] = {
            "supports_snapshot": supports_snapshot,
            "special_params": special_params,
        }

    return methods


# Generate the methods dictionary
STATISTIC_PLOT_METHODS = _generate_plot_methods()


class StatisticsPlotAccessor:
    """
    Main statistics plot accessor providing access to different plot types
    """

    _INTERNAL_METHOD_CONFIG: dict[str, Any] = {}

    def __init__(self, n: "Network") -> None:
        """
        Initialize the statistics plot accessor.

        Parameters
        ----------
        n : Network
            PyPSA network object
        """
        self._network = n
        self._n = n  # Add alias for consistency with MapPlotAccessor
        self._statistics = n.statistics
        self._create_plot_methods()

    def _create_plot_methods(self) -> None:
        """Create plotting methods that mirror statistics methods"""
        for method_name, config in STATISTIC_PLOT_METHODS.items():
            if not hasattr(self._statistics, method_name):
                continue  # Skip if the statistics method doesn't exist

            # Create the plot method with proper configuration
            plot_method = self._create_plot_method(self, method_name=method_name)

            # Use partial to properly bind the instance to the method
            bound_method = partial(plot_method, self)
            bound_method.__name__ = method_name
            bound_method.__doc__ = plot_method.__doc__

            # Attach the method to this instance
            setattr(self, method_name, bound_method)

    @staticmethod
    def _create_plot_method(accessor: Any, method_name: str) -> Callable:
        """
        Create a plotting method for the given statistics method.

        Parameters
        ----------
        method_name : str
            Name of the statistics method to create a plot method for
        config : dict
            Configuration parameters for the plotting method

        Returns
        -------
        Callable
            The generated plotting method
        """

        @wraps(accessor._plot_statistics)
        def plot_method(self_instance: Any, *args: Any, **kwargs: Any) -> Any:
            # Combine default config with user-provided kwargs
            merged_kwargs = {
                **accessor._INTERNAL_METHOD_CONFIG.get(method_name, {}),
                **kwargs,
            }
            # Call _plot_statistics with proper self reference
            return self_instance._plot_statistics(method_name, *args, **merged_kwargs)

        return plot_method

    def _generate_common_docstring(self, method_name: str) -> str:
        """
        Generate a common docstring for plot methods.

        Parameters
        ----------
        method_name : str
            Name of the method to generate docstring for

        Returns
        -------
        str
            Generated docstring
        """
        doc = f"Plot {method_name.replace('_', ' ')}.\n\n"
        doc += "Parameters\n----------\n"
        return doc
