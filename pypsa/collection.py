import logging
import re
from collections.abc import Callable, Iterator, Sequence
from itertools import product
from typing import Any

import pandas as pd

from pypsa.definitions.structures import Dict
from pypsa.networks import Network
from pypsa.statistics.expressions import StatisticsAccessor

logger = logging.getLogger(__name__)


class NetworkCollection:
    """
    A collection of networks that can be accessed like a single network.

    Note:
    ----
    A single network is mirrored in two ways:
        1. For each nested method or property of a network, the collection will
        dynamically create a new MemberProxy object that wraps around it and allows for
        custom processing. The '_method_patterns' dictionary in the MemberProxy class
        defines which processor is used for which method or property. If no processor
        is defined, a NotImplementedError is raised.
        2. Some accessors of the Network class already support Networks and
            NetworkCollections, since via the step above the NetworkCollection can
            already duck-type to a Network. If this is the case, the accessor is
            directly initialised with a NetworkCollection instead.
    """

    def __init__(
        self,
        networks: pd.Series | Sequence[Network],
        index: pd.Index | pd.MultiIndex | Sequence | None = None,
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

        if isinstance(networks, pd.Series) and index is not None:
            msg = (
                "When passing a pandas Series, the index must be None, "
                "as the Series index is used as the collection index."
            )
            raise ValueError(msg)

        if not all(isinstance(n, Network) for n in networks):
            raise TypeError("All values in the Series must be PyPSA Network objects.")

        if not isinstance(networks, pd.Series):
            # Format and validate index
            if index is None and not isinstance(networks, pd.Series):
                names = ["network" if not n.name else n.name for n in networks]

                # Add a unique suffix to each network name to avoid duplicates, if necessary
                counts = {}
                for i, item in enumerate(names):
                    if item in counts:
                        counts[item] += 1
                        names[i] = f"{item}_{counts[item]}"
                        logger.info(
                            'Network "%s" is duplicated, renamed one to "%s"',
                            item,
                            names[i],
                        )
                    else:
                        counts[item] = 0
                index = pd.Index(names, name="network")
            elif isinstance(index, Sequence):
                if len(index) != len(networks):
                    msg = "The length of the index must match the number of networks provided."
                    raise ValueError(msg)
                index = pd.Index(index)
            elif not isinstance(index, pd.Index | pd.MultiIndex):
                msg = (
                    "The index must be a pandas Index or a sequence of names matching the "
                    "number of networks provided."
                )
                raise TypeError(msg)

            networks = pd.Series(networks, index=index)

        # Only set default index name for non-MultiIndex
        if (
            not isinstance(networks.index, pd.MultiIndex)
            and networks.index.name is None
        ):
            networks.index.name = "network"

        self.networks = networks

        # Validate index names
        if isinstance(self.networks.index, pd.MultiIndex):
            if any(name is None for name in self.networks.index.names):
                raise ValueError("All levels of MultiIndex must have names")

        # Initialize accessors which support NetworkCollections and don't need a proxy
        # member
        self.statistics = StatisticsAccessor(self)

    def __getattr__(self, name: str) -> Any:
        """
        Get attribute from all networks in the collection.

        Returns an MemberProxy that will either call the method on each network
        when invoked or retrieve the property values when accessed.
        """
        if not self.networks.any():
            msg = "Please provide at least one network."
            raise AttributeError(msg)

        try:
            return MemberProxy(self, lambda n: getattr(n, name), name)
        except AttributeError as e:
            msg = (
                "Only members as they are defined in any Network class can be accessed."
            )
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


class MemberProxy:
    """
    Wrapper for network accessors that combines results from multiple networks.

    This class handles arbitrary nesting of accessor methods and properties,
    dynamically proxying calls to the underlying network objects.
    """

    # Method patterns for special handling
    _method_patterns = {
        "vertical_concat": r"^"  # Start of string
        # Static component dataframes
        r"(sub_networks|buses|carriers|global_constraints|lines|line_types|"
        r"transformers|transformer_types|links|loads|generators|storage_units|"
        r"stores|shunt_impedances|shapes|"
        r"static|"
        r"get_active_assets|"
        r"get_committable_i|"
        # statistics and all statistics expressions
        r"statistics|"
        r"statistics\.[^\.\s]+)"
        r"$",  # End of string
        "horizontal_concat": r"^"  # Start of string
        r"(sub_networks|buses|carriers|global_constraints|lines|line_types|"
        r"transformers|transformer_types|links|loads|generators|storage_units|"
        r"stores|shunt_impedances|shapes)_t|"
        r"dynamic"
        r"$",  # End of string
        "return_from_first": r"^"
        r"\S+_components|"
        r"snapshots|"
        r"snapshot_weightings"
        r"$",  # End of string
    }

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Create new instance of MemberProxy.

        If no Wrapper is needed, since the return value is not a callable, immediately
        return the result of the default processor function.
        """
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)

        # Immediately end recursion for non callable returns
        first_accessor = instance.accessor_func(instance.collection.networks.iloc[0])
        if not callable(first_accessor):
            processor = instance.get_processor()
            return processor(is_call=False)

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
        processor = self.get_processor()
        return processor(True, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Handle attribute access on the accessor.

        This method handles three cases:
        1. The attribute is another accessor object (returns a new MemberProxy)
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
        return MemberProxy(
            self.collection, lambda n: getattr(self.accessor_func(n), name), new_path
        )

    def get_processor(self) -> Any:
        # Check for pattern-based method processor
        for processor_name, pattern in self._method_patterns.items():
            if re.match(pattern, self.accessor_path):
                processor = getattr(self, processor_name)
                return processor

        msg = (
            f"'{self.accessor_path}' is currently not a supported method/ property for "
            f"network collections. This might change in the future."
        )
        raise NotImplementedError(msg)

    # Helper functions for processing
    # -------------------------------

    def _concat_indexes(self, results: dict) -> Any:
        # Extract all values from indices
        values_list = [list(idx) for idx in results.values()]

        # Get all combinations of values
        combinations = list(product(*values_list))

        # Get names from keys (handle both string and tuple cases)
        if all(isinstance(k, str) for k in results.keys()):
            # All keys are strings
            names = list(results.keys())
        else:
            # All keys are tuples
            names = [item for tup in results.keys() for item in tup]

        return pd.MultiIndex.from_tuples(combinations, names=names)

    def _do_concat(self, results: Any, axis: int) -> Any:
        # Check if values are dictionaries
        if all(isinstance(v, dict) for v in results.values()):
            # Get all unique keys across all dictionaries
            all_keys = set().union(*[d.keys() for d in results.values()])

            merged_results = Dict()
            for key in all_keys:
                key_results = {
                    idx: results[idx].get(key) for idx in results if key in results[idx]
                }

                # Recursively call on subsets
                merged_results[key] = self._do_concat(key_results, axis=axis)
            return merged_results
        else:
            # Default case - simple concatenation
            first_result = next(iter(results.values()))

            if isinstance(first_result, pd.Index):
                result = self._concat_indexes(results)
            else:
                result = pd.concat(results, axis=axis)
            if axis == 0:
                result.index.names = (
                    self.collection.networks.index.names or ["network"]
                ) + first_result.index.names

            elif axis == 1:
                result.columns.names = (
                    self.collection.networks.index.names or ["network"]
                ) + first_result.columns.names
            else:
                msg = "Axis must be 0 or 1"
                raise AssertionError(msg)
            return result

    # -----------------
    # Custom processors
    # Any custom processor are defined below. They need to be added with the same
    # signature and added to the _method_patterns list above.
    # -----------------

    def vertical_concat(self, is_call: bool, *args: Any, **kwargs: Any) -> Any:
        results = {}

        for idx, network in self.collection.networks.items():
            accessor = self.accessor_func(network)
            if is_call:
                result = accessor(*args, **kwargs)
            else:
                result = accessor
            results[idx] = result

        return self._do_concat(results, axis=0)

    def horizontal_concat(self, is_call: bool, *args: Any, **kwargs: Any) -> Any:
        results = {}
        for idx, network in self.collection.networks.items():
            accessor = self.accessor_func(network)
            if is_call:
                result = accessor(*args, **kwargs)
            else:
                result = accessor
            results[idx] = result

        return self._do_concat(results, axis=1)

    def return_from_first(self, is_call: bool, *args: Any, **kwargs: Any) -> Any:
        """
        Return the result from the first network in the collection.

        This is used for properties that are expected to be the same across all networks.
        """
        # Get the first network
        first_network = self.collection.networks.iloc[0]
        accessor = self.accessor_func(first_network)
        if is_call:
            result = accessor(*args, **kwargs)
        else:
            result = accessor

        return result

    def statistics_plot(self, is_call: bool, *args: Any, **kwargs: Any) -> Any:
        if not is_call:
            msg = "Arguments are not allowed for property accessors"
            raise ValueError(msg)

        # self.collection._statistics.installed_capacity.plot()

    # wrapped_callable = MethodHandlerWrapper(
    #     handler_class=StatisticHandler, inject_attrs={"n": "_n"}
    # )(your_callable)
