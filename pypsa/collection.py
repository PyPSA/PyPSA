import logging
import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from pypsa.networks import Network

logger = logging.getLogger(__name__)


class NetworkCollection:
    """
    A collection of networks that can be accessed like a single network.
    """

    def __init__(self, *networks: Network) -> None:
        """
        Initialize the NetworkCollection with one or more networks.

        Parameters
        ----------
        *networks : Network
            One or more Network objects to include in the collection.
        """
        if not networks:
            raise ValueError("At least one network must be provided")

        if not all(isinstance(n, Network) for n in networks):
            raise TypeError("All arguments must be Network instances")

        self._networks = pd.Series(list(networks), index=range(len(networks)))
        self._statistics = None

    def __getattr__(self, name: str) -> Any:
        """
        Get attribute from all networks in the collection.

        Returns an MemberWrapper that will either call the method on each network
        when invoked or retrieve the property values when accessed.
        """
        if not self._networks.any():
            msg = "Please provide at least one network."
            raise AttributeError(msg)

        try:
            return MemberWrapper(self, lambda n: getattr(n, name), name)
        except AttributeError as e:
            msg = "Only attributes as they are defined in the Network class can be accessed."
            raise AttributeError(msg) from e

    def __getitem__(self, key: Any) -> Any:
        """Get a subset of networks using pandas Series indexing."""
        if isinstance(key, slice | pd.Series):
            selected = self._networks[key]
            return NetworkCollection(selected) if len(selected) > 0 else None
        return self._networks[key]


class MemberWrapper:
    """
    Wrapper for network accessors that combines results from multiple networks.

    This class handles arbitrary nesting of accessor methods and properties,
    dynamically proxying calls to the underlying network objects.
    """

    # Method patterns for special handling
    _method_patterns = [
        # Add your regex patterns here, e.g.:
        (r"^name$", "handle_dev")
    ]

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Create new instance of MemberWrapper.

        If no Wrapper is needed, since the return value is not a callable, immediately
        return the result of the default processor function.
        """
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)

        # Immediately end recursion for non callable returns
        first_accessor = instance._accessor_func(instance._collection._networks[0])
        if not callable(first_accessor):
            if instance._accessor_path:
                # Check for pattern-based method processor
                for pattern, processor_name in instance._method_patterns:
                    if re.match(pattern, instance._accessor_path):
                        processor = getattr(instance, processor_name)
                        return processor(is_call=False)

            return instance.default_processor(is_call=False)
        return instance

    def __init__(
        self,
        collection: NetworkCollection,
        accessor_func: Callable,
        accessor_path: str = "",
    ) -> None:
        """
        Initialize the wrapper.

        Parameters
        ----------
        collection : NetworkCollection
            The collection of networks to operate on
        accessor_func : callable
            Function that returns the appropriate accessor for a given network
        accessor_path : str, optional
            The dot-separated path of accessor names from NetworkCollection to this wrapper
        """
        self._collection = collection
        self._accessor_func = accessor_func
        self._accessor_path = accessor_path

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Handle direct calls to the accessor.

        For methods with custom implementations defined by regex patterns,
        uses the matching handler method. Otherwise uses the default behavior
        of collecting results from each network.
        """
        # Get the method name from the accessor function
        if len(self._collection._networks) > 0:
            first_accessor = self._accessor_func(self._collection._networks[0])
            method_name = getattr(first_accessor, "__name__", None)

            if method_name:
                # Check for pattern-based method processor
                for pattern, processor_name in self._method_patterns:
                    if re.match(pattern, method_name):
                        processor = getattr(self, processor_name)
                        return processor(is_call=True, *args, **kwargs)

        # Default behavior for methods
        return self.default_processor(is_call=True, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Handle attribute access on the accessor.

        This method handles three cases:
        1. The attribute is another accessor object (returns a new MemberWrapper)
        2. The attribute is a method (returns a function that aggregates results)
        3. The attribute is a property (returns a ResultWrapper of property values)
        """
        # Get the attribute from the first accessor to determine its type
        if len(self._collection._networks) == 0:
            raise AttributeError(
                f"Cannot access attribute '{name}' on empty collection"
            )

        # Create the new accessor path by appending the attribute name
        new_path = f"{self._accessor_path}.{name}" if self._accessor_path else name

        # For any attribute, create a new accessor function that chains the attribute access
        return MemberWrapper(
            self._collection, lambda n: getattr(self._accessor_func(n), name), new_path
        )

    def default_processor(self, is_call: bool, *args: Any, **kwargs: Any) -> Any:
        """
        Default processing behavior for properties and methods.

        Collects property values/ method results from all networks and returns them as
        a list.

        Parameters
        ----------
        is_call : bool
            Whether this is a method call (needs to invoke the accessor)
        *args, **kwargs
            Arguments to pass to the method if is_call=True
        """
        if not is_call and (args or kwargs):
            msg = "Arguments are not allowed for property accessors"
            raise ValueError(msg)

        results = []
        for network in self._collection._networks:
            accessor = self._accessor_func(network)
            if is_call:
                result = accessor(*args, **kwargs)
            else:
                result = accessor
            results.append(result)
        return results

    # -----------------
    # Custom processors
    # Any custom processor are defined below. They need to be added with the same
    # signature and added to the _method_patterns list above.
    # -----------------

    def handle_dev(self, is_call: bool, *args: Any, **kwargs: Any) -> Any:
        """
        Custom handler for methods matching the 'get_*' pattern.

        Parameters
        ----------
        accessor_func : callable
            Function that returns the appropriate accessor for a given network
        path : str
            The full accessor path from NetworkCollection to current method (e.g. 'plot.energy_balance')
        *args, **kwargs
            Arguments passed to the method
        """
        # Implement custom behavior here
        if not is_call and (args or kwargs):
            msg = "Arguments are not allowed for property accessors"
            raise ValueError(msg)

        results = []
        for network in self._collection._networks:
            accessor = self._accessor_func(network)
            if is_call:
                result = accessor(*args, **kwargs)
            else:
                result = accessor
            results.append(result)
        return ",".join(results)
