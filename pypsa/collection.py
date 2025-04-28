import logging
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pandas as pd

from pypsa.networks import Network

logger = logging.getLogger(__name__)


class NetworkCollection:
    """
    A collection of networks that can be accessed like a single network.
    """

    def __init__(
        self, networks: Sequence[Network] | pd.Series, index: pd.Index | None = None
    ) -> None:
        """
        Initialize the NetworkCollection with one or more networks.

        Parameters
        ----------
        *networks : Network
            One or more Network objects to include in the collection.
        """
        # if not networks:
        #     raise ValueError("At least one network must be provided")

        # if not all(isinstance(n, Network) for n in networks):
        #     raise TypeError("All arguments must be Network instances")

        networks = pd.Series(networks, index=index)
        if networks.index.name is None:
            networks.index.name = "network"
        if not all(isinstance(n, Network) for n in networks):
            raise TypeError("All values in the Series must be PyPSA Network objects.")
        self.networks = networks

    def __getattr__(self, name: str) -> Any:
        """
        Get attribute from all networks in the collection.

        Returns an MemberWrapper that will either call the method on each network
        when invoked or retrieve the property values when accessed.
        """
        if not self.networks.any():
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
            selected = self.networks[key]
            return NetworkCollection(selected) if len(selected) > 0 else None
        return self.networks[key]

    def __len__(self) -> int:
        return len(self.networks)

    def __iter__(self) -> Iterator[Network]:
        """
        Iterate over the Network objects in the container.
        """
        return iter(self.networks)

    @property
    def index(self) -> pd.Index:
        """
        Get the index of the NetworkCollection.

        Returns
        -------
        pd.Index
            The index of the NetworkCollection.
        """
        return self.networks.index

    @property
    def _index_names(self) -> list[str]:
        """
        Get the names of the index of the NetworkCollection.

        Returns
        -------
        list[str]
            The names of the index of the NetworkCollection.
        """
        return self.index.names or [self.index.name]

    @property
    def carriers(self) -> pd.DataFrame:
        """
        Get a unique DataFrame of carriers across all contained networks.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the unique carriers found in all networks,
            indexed by carrier name.
        """
        all_carriers = [n.carriers for n in self.networks]
        combined_carriers = pd.concat(all_carriers)
        # Keep the first occurrence of each carrier based on the index
        unique_carriers = combined_carriers[
            ~combined_carriers.index.duplicated(keep="first")
        ]
        return unique_carriers.sort_index()


class MemberWrapper:
    """
    Wrapper for network accessors that combines results from multiple networks.

    This class handles arbitrary nesting of accessor methods and properties,
    dynamically proxying calls to the underlying network objects.
    """

    # Method patterns for special handling
    _method_patterns = [
        # Add your regex patterns here, e.g.:
        (
            r"^StatisticsAccessor|sub_networks|buses|carriers|global_constraints|lines|"
            r"line_types|transformers|transformer_types|links|loads|generators|"
            r"storage_units|stores|shunt_impedances|shapes$",
            "basic_concat",
        ),
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
        first_accessor = instance.accessor_func(instance.collection.networks[0])
        if not callable(first_accessor):
            if instance.accessor_path:
                # Check for pattern-based method processor
                for pattern, processor_name in instance._method_patterns:
                    if re.match(pattern, instance.accessor_path):
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
        self.collection = collection
        self.accessor_func = accessor_func
        self.accessor_path = accessor_path

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Handle direct calls to the accessor.

        For methods with custom implementations defined by regex patterns,
        uses the matching handler method. Otherwise uses the default behavior
        of collecting results from each network.
        """
        # Get the method name from the accessor function
        if len(self.collection.networks) > 0:
            first_accessor = self.accessor_func(self.collection.networks[0])
            method_name = getattr(first_accessor, "__class__", None)
            if not method_name:
                method_name = getattr(first_accessor, "__name__", None)
            else:
                method_name = getattr(method_name, "__name__", None)

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
        if len(self.collection.networks) == 0:
            raise AttributeError(
                f"Cannot access attribute '{name}' on empty collection"
            )

        # Create the new accessor path by appending the attribute name
        new_path = f"{self.accessor_path}.{name}" if self.accessor_path else name

        # For any attribute, create a new accessor function that chains the attribute access
        return MemberWrapper(
            self.collection, lambda n: getattr(self.accessor_func(n), name), new_path
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
        for network in self.collection.networks:
            accessor = self.accessor_func(network)
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

    def basic_concat(self, is_call: bool, *args: Any, **kwargs: Any) -> Any:
        results = {}

        for idx, network in self.collection.networks.items():
            accessor = self.accessor_func(network)
            if is_call:
                result = accessor(*args, **kwargs)
            else:
                result = accessor
            results[idx] = result

        combined = pd.concat(
            results, names=[self.collection.networks.index.name or "network"]
        )

        return combined
