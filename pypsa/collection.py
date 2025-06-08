"""NetworkCollection class for handling multiple PyPSA networks."""

import logging
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pandas as pd

from pypsa.definitions.structures import Dict
from pypsa.networks import Network
from pypsa.statistics.expressions import StatisticsAccessor

logger = logging.getLogger(__name__)


class NetworkCollection:
    """A collection of networks that can be accessed like a single network.

    **Supported Methods and Properties**
    The following Network methods and properties are supported for NetworkCollections:

    Components Data:
        All components data (static and dynamic) can be accessed and is returned as a
        concatenated pandas DataFrame (e.g. `nc.buses`, `nc.buses_t`, `nc.generators`)

    Statistics:
        All statistics expressions can be accessed in the same way as for a single
        network. This includes dataframes and plots. E.g. `nc.statistics.energy_balance()`,
        `nc.statistics.energy_balance.plot()`, `nc.statistics.energy_balance.plot.bar()`.

    Examples
    --------
    Create a collection from file paths:

    >>> nc = pypsa.NetworkCollection(["network1.nc", "network2.nc"]) # doctest: +SKIP

    Create a collection from Network objects:

    >>> n1 = pypsa.examples.ac_dc_meshed()
    >>> n1.name = "network1"
    >>> n2 = pypsa.examples.model_energy()
    >>> n2.name = "network2"
    >>> nc = pypsa.NetworkCollection([n1, n2])
    >>> nc
    NetworkCollection
    -----------------
    Networks: 2
    Index name: 'network'
    Entries: ['network1', 'network2']

    Access component data across all networks:

    >>> nc.generators
                                      bus control  ... weight  p_nom_opt
    network  Generator                             ...
    network1 Manchester Wind   Manchester      PQ  ...    1.0        0.0
             Manchester Gas    Manchester      PQ  ...    1.0        0.0
             Norway Wind           Norway      PQ  ...    1.0        0.0
             Norway Gas            Norway      PQ  ...    1.0        0.0
             Frankfurt Wind     Frankfurt      PQ  ...    1.0        0.0
             Frankfurt Gas      Frankfurt      PQ  ...    1.0        0.0
    network2 load shedding    electricity      PQ  ...    1.0        0.0
             wind             electricity      PQ  ...    1.0        0.0
             solar            electricity      PQ  ...    1.0        0.0
    <BLANKLINE>
    [9 rows x 37 columns]


    >>> nc.statistics.installed_capacity()
    component  network   carrier
    Generator  network1  gas              150000.00
                         wind                290.00
               network2  load shedding     10901.16
    Line       network1  AC               280000.00
    Link       network1  DC                 4000.00
    Name: value, dtype: float64

    Use custom index:

    >>> import pandas as pd
    >>> index = pd.Index(["scenario_A", "scenario_B"])
    >>> nc = pypsa.NetworkCollection([n1, n2], index=index)
    >>> nc
    NetworkCollection
    -----------------
    Networks: 2
    Index name: 'network'
    Entries: ['scenario_A', 'scenario_B']

    Notes
    -----
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
        networks: pd.Series | Sequence[Network | str],
        index: pd.Index | pd.MultiIndex | Sequence | None = None,
    ) -> None:
        """Initialize the NetworkCollection with one or more networks.

        Parameters
        ----------
        networks : pd.Series | Sequence[Network | str]
            Sequence or pd.Series of Network objects or strings (file paths/ urls) to
            include in the collection. If strings are provided, they will be passed to
            pypsa.Network() to create Network objects.
        index : pd.Index, pd.MultiIndex, Sequence, or None, optional
            The index to use for the collection. If `networks` is of type `pd.Series`,
            no index is allowed and it will be retrieved from the Series. If None, a
            default index based on the network names will be created.

        """
        if isinstance(networks, pd.Series) and index is not None:
            msg = (
                "When passing a pandas Series, the index must be None, "
                "as the Series index is used as the collection index."
            )
            raise ValueError(msg)

        # Validate that networks is not a single string (which would iterate char by char)
        if isinstance(networks, str):
            msg = "Single strings are not supported. Pass a list of strings or Network objects."
            raise TypeError(msg)

        def _convert_to_network(item: Any) -> Network:
            if isinstance(item, Network):
                return item
            elif isinstance(item, str):
                return Network(item)
            else:
                msg = f"All values must be PyPSA Network objects or strings, got {type(item)}."
                raise TypeError(msg)

        if isinstance(networks, pd.Series):
            self.networks = networks.map(_convert_to_network)
        else:
            self.networks = [_convert_to_network(n) for n in networks]

        if not isinstance(self.networks, pd.Series):
            # Format and validate index
            if index is None and not isinstance(self.networks, pd.Series):
                names = ["network" if not n.name else n.name for n in self.networks]

                # Check for duplicate names
                if len(names) != len(set(names)):
                    duplicates = [name for name in set(names) if names.count(name) > 1]
                    msg = (
                        f"Duplicate network names found: {duplicates}. "
                        "Please provide a custom index or ensure all networks have unique names."
                    )
                    raise ValueError(msg)
                index = pd.Index(names, name="network")
            elif isinstance(index, Sequence):
                if len(index) != len(self.networks):
                    msg = "The length of the index must match the number of networks provided."
                    raise ValueError(msg)
                index = pd.Index(index)
            elif not isinstance(index, pd.Index | pd.MultiIndex):
                msg = (
                    "The index must be a pandas Index or a sequence of names matching the "
                    "number of networks provided."
                )
                raise TypeError(msg)

            self.networks = pd.Series(self.networks, index=index)

        # Only set default index name for non-MultiIndex
        if (
            not isinstance(self.networks.index, pd.MultiIndex)
            and self.networks.index.name is None
        ):
            self.networks.index.name = "network"

        # Validate index names
        if isinstance(self.networks.index, pd.MultiIndex):  # noqa: SIM102
            if any(name is None for name in self.networks.index.names):
                msg = "All levels of MultiIndex must have names"
                raise ValueError(msg)

        self._validate_network_compatibility()

        # Initialize accessors which support NetworkCollections and don't need a proxy
        # member
        self.statistics = StatisticsAccessor(self)

    def __getattr__(self, name: str) -> Any:
        """Get attribute from all networks in the collection.

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
        try:
            if isinstance(key, slice | pd.Series):
                selected = self.networks[key]
                if len(selected) == 0:
                    msg = f"Selection with key {key} resulted in empty collection"
                    raise ValueError(msg)
                return NetworkCollection(selected)
            return self.networks[key]
        except KeyError as e:
            msg = f"Key '{key}' not found in NetworkCollection index: {list(self.networks.index)}"
            raise KeyError(msg) from e

    def __len__(self) -> int:
        """Get the number of networks in the collection."""
        return len(self.networks)

    def __iter__(self) -> Iterator[Network]:
        """Iterate over the Network objects in the container."""
        return iter(self.networks)

    @property
    def index(self) -> pd.Index:
        """Get the index of the NetworkCollection.

        Returns
        -------
        pd.Index
            The index of the NetworkCollection.

        """
        return self.networks.index

    @property
    def _index_names(self) -> list[str]:
        """Get the names of the index of the NetworkCollection.

        Returns
        -------
        list[str]
            The names of the index of the NetworkCollection.

        """
        return self.index.names or [self.index.name]

    def __repr__(self) -> str:
        """Return a string representation of the NetworkCollection.

        Returns
        -------
        str
            A string representation showing the number of networks and index information.

        """
        n_networks = len(self.networks)

        # Show index information
        if isinstance(self.networks.index, pd.MultiIndex):
            index_info = f"MultiIndex with {self.networks.index.nlevels} levels: {list(self.networks.index.names)}"
            # Show first few entries of the MultiIndex
            if n_networks > 0:
                sample_size = min(5, n_networks)
                sample_entries = list(self.networks.index[:sample_size])
                index_info += f"\n  First {sample_size} entries: {sample_entries}"
                if n_networks > sample_size:
                    index_info += f"\n  ... and {n_networks - sample_size} more"
        else:
            index_name = self.networks.index.name or "network"
            index_info = f"Index name: '{index_name}'"
            if n_networks > 0:
                sample_size = min(5, n_networks)
                sample_entries = list(self.networks.index[:sample_size])
                index_info += f"\nEntries: {sample_entries}"
                if n_networks > sample_size:
                    index_info += f" ... and {n_networks - sample_size} more"

        return f"NetworkCollection\n-----------------\nNetworks: {n_networks}\n{index_info}"

    @property
    def carriers(self) -> pd.DataFrame:
        """Get a unique DataFrame of carriers across all contained networks.

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

    def _validate_network_compatibility(self) -> None:
        """Validate basic compatibility between networks in the collection.

        Raises
        ------
        ValueError
            If networks have incompatible structures that would prevent
            meaningful aggregation.

        """
        if len(self.networks) <= 1:
            return  # No validation needed for single network or empty collection

        # TODO: Implement basic validation of network compatibility


class MemberProxy:
    """Wrapper for network accessors that combines results from multiple networks.

    This class handles arbitrary nesting of accessor methods and properties,
    dynamically proxying calls to the underlying network objects.
    """

    collection: NetworkCollection

    # Method patterns for special handling
    _method_patterns = {
        # run_per_network
        # ---------------
        "run_per_network": r"^"
        r"(consistency_check_plots)+"
        r"$",
        # ---------------
        "vertical_concat": r"^"
        # Static component dataframes
        r"(sub_networks|buses|carriers|global_constraints|lines|line_types|"
        r"transformers|transformer_types|links|loads|generators|storage_units|"
        r"stores|shunt_impedances|shapes|"
        r"static|"
        r"get_active_assets|"
        # statistics and all statistics expressions
        r"statistics|"
        r"statistics\.[^\.\s]+)"
        r"$",
        # ---------------
        "horizontal_concat": r"^"
        r"(sub_networks|buses|carriers|global_constraints|lines|line_types|"
        r"transformers|transformer_types|links|loads|generators|storage_units|"
        r"stores|shunt_impedances|shapes)_t|"
        r"dynamic|"
        r"get_switchable_as_dense"
        r"$",
        # ---------------
        "return_from_first": r"^"
        r"\S+_components|"
        r"snapshots|"
        r"snapshot_weightings|"
        r"bus_carrier_unit"
        r"$",
        # ---------------
        "index_concat": r"^"
        r"get_committable_i",
    }

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create new instance of MemberProxy.

        If no Wrapper is needed, since the return value is not a callable, immediately
        return the result of the default processor function.
        """
        instance = super().__new__(cls)
        cls.__init__(instance, *args, **kwargs)

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
        """Initialize the wrapper.

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
        """Handle direct calls to the accessor.

        For methods with custom implementations defined by regex patterns,
        uses the matching handler method. Otherwise uses the default behavior
        of collecting results from each network.
        """
        processor = self.get_processor()
        return processor(True, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Handle attribute access on the accessor.

        This method handles three cases:
        1. The attribute is another accessor object (returns a new MemberProxy)
        2. The attribute is a method (returns a function that aggregates results)
        3. The attribute is a property (returns a ResultWrapper of property values)
        """
        # Get the attribute from the first accessor to determine its type
        if len(self.collection.networks) == 0:
            msg = f"Cannot access attribute '{name}' on empty collection"
            raise AttributeError(msg)

        # Create the new accessor path by appending the attribute name
        new_path = f"{self.accessor_path}.{name}" if self.accessor_path else name

        # For any attribute, create a new accessor function that chains the attribute access
        return MemberProxy(
            self.collection, lambda n: getattr(self.accessor_func(n), name), new_path
        )

    def get_processor(self) -> Any:
        """Determine the appropriate processor for the current accessor path."""
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
        # Build names list based on the network collection index and result index names
        network_names = self.collection._index_names

        # Get the names from the first result index (they should all be the same)
        first_result = next(iter(results.values()))
        result_names = [first_result.name] if first_result.name else []

        # Combine network names with result names
        all_names = network_names + result_names

        # Flatten the combinations to include network index values
        flattened_combinations = []
        for idx, result_idx in results.items():
            # idx can be a string or tuple depending on whether collection has MultiIndex
            idx_values = (idx,) if isinstance(idx, str) else idx
            flattened_combinations.extend([idx_values + (val,) for val in result_idx])

        return pd.MultiIndex.from_tuples(flattened_combinations, names=all_names)

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

    def run_per_network(self, is_call: bool, *args: Any, **kwargs: Any) -> Any:
        """Run the accessor function for each network in the collection."""
        results = []
        for _, network in self.collection.networks.items():  # noqa: PERF102
            accessor = self.accessor_func(network)
            if is_call:
                result = accessor(*args, **kwargs)
            else:
                result = accessor
            results.append(result)
        return results

    def vertical_concat(self, is_call: bool, *args: Any, **kwargs: Any) -> Any:
        """Concatenate results vertically (axis=0) from all networks in the collection."""
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
        """Concatenate results horizontally across networks."""
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
        """Return the result from the first network in the collection.

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

    def index_concat(self, is_call: bool, *args: Any, **kwargs: Any) -> Any:
        """Concatenate indexes from all networks in the collection into a MultiIndex.

        This function collects indexes from each network and combines them into a
        MultiIndex where the network identifier(s) form the first level(s) and the
        original index values form the subsequent levels.
        """
        results = {}

        for idx, network in self.collection.networks.items():
            accessor = self.accessor_func(network)
            if is_call:
                result = accessor(*args, **kwargs)
            else:
                result = accessor
            results[idx] = result

        # Use the _concat_indexes helper which already handles MultiIndex creation
        return self._concat_indexes(results)
