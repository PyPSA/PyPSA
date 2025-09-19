import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class InvalidOptionError(AttributeError):
    """Custom exception for invalid options."""

    def __init__(
        self,
        message: str | None = None,
        option_path: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if message is None:
            if option_path is None:
                msg = "'option_path' must be provided if 'message' is not."
                raise ValueError(msg)
            message = f"Invalid option '{option_path}'. Check 'describe_options()' for valid options."
        self.message = message

        super().__init__(self.message, *args, **kwargs)

    def __str__(self) -> str:
        return self.message


class Option:
    def __init__(self, value: Any = None, default: Any = None, docs: str = "") -> None:
        self._value = value if value is not None else default
        self._default = default
        self._docs = docs

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        self._value = value

    def reset(self) -> None:
        """Reset the value to its default."""
        self._value = self._default

    @property
    def docs(self) -> str:
        """Get the documentation for this option."""
        return self._docs


class OptionsNode:
    """PyPSA package options.

    This class provides a hierarchical structure for managing package options and
    the functionality can be accessed via `pypsa.options`.

    """

    def __init__(self, name: str = "") -> None:
        self._name = name
        self._children: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """Get the value of an option at the specified path.

        Examples
        --------
        >>> pypsa.options.general.allow_network_requests
        True

        >>> pypsa.options.params.statistics.round
        5

        """
        if name not in self._children:
            raise InvalidOptionError(option_path=name)

        child = self._children[name]
        if isinstance(child, Option):
            return child.value
        return child

    def __setattr__(self, name: str, value: Any) -> None:
        """Set the value of an option at the specified path.

        Examples
        --------
        Set the option to False:
        >>> pypsa.options.general.allow_network_requests = False
        >>> pypsa.options.general.allow_network_requests
        False

        Reset back to default:
        >>> pypsa.options.reset_all()

        """
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        if name not in self._children:
            raise InvalidOptionError(option_path=name)

        child = self._children[name]
        if isinstance(child, OptionsNode):
            msg = f"Cannot set value for category '{name}'."
            raise InvalidOptionError(msg)
        child.value = value

    def _add_option(self, path: str, default: Any = None, docs: str = "") -> None:
        """Add an option at the specified path."""
        parts = path.split(".")
        node = self

        # Navigate/create the path
        for _i, part in enumerate(parts[:-1]):
            if part not in node._children:
                node._children[part] = OptionsNode(part)
            elif not isinstance(node._children[part], OptionsNode):
                msg = f"Cannot add category '{part}' because an option already exists at this path."
                raise ValueError(msg)
            node = node._children[part]

        # Add the option at the leaf
        leaf_name = parts[-1]
        if leaf_name in node._children and isinstance(
            node._children[leaf_name], OptionsNode
        ):
            msg = f"Cannot add option '{leaf_name}' because a category already exists at this path."
            raise ValueError(msg)

        node._children[leaf_name] = Option(default, default, docs)

    def get_option(self, path: str) -> Any:
        """Get the value of an option at the specified path.

        Parameters
        ----------
        path : str
            Path to the option. Must be in the format "category.option_name" or "category.subcategory.option_name"

        Returns
        -------
        Any
            The value of the option.

        Examples
        --------
        >>> pypsa.options.get_option("general.allow_network_requests")
        True
        >>> pypsa.options.get_option("params.statistics.round")
        5

        """
        parts = path.split(".")
        node = self

        # Navigate to the parent
        for part in parts[:-1]:
            if part not in node._children or not isinstance(
                node._children[part], OptionsNode
            ):
                raise InvalidOptionError(option_path=part)
            node = node._children[part]

        # Get the option value
        leaf_name = parts[-1]
        if leaf_name not in node._children or isinstance(
            node._children[leaf_name], OptionsNode
        ):
            raise InvalidOptionError(option_path=leaf_name)

        return node._children[leaf_name].value

    def set_option(self, path: str, value: Any) -> None:
        """Set the value of an option at the specified path.

        Parameters
        ----------
        path : str
            Path to the option. Must be in the format "category.option_name" or "category.subcategory.option_name"
        value : Any
            Value to set for the option.

        Examples
        --------
        Set the option to False:
        >>> pypsa.options.set_option("general.allow_network_requests", False)
        >>> pypsa.options.general.allow_network_requests
        False

        Reset back to default:
        >>> pypsa.options.reset_all()
        >>> pypsa.options.general.allow_network_requests
        True

        """
        parts = path.split(".")
        node = self

        # Navigate to the parent
        for part in parts[:-1]:
            if part not in node._children or not isinstance(
                node._children[part], OptionsNode
            ):
                raise InvalidOptionError(option_path=part)
            node = node._children[part]

        # Set the option value
        leaf_name = parts[-1]
        if leaf_name not in node._children or isinstance(
            node._children[leaf_name], OptionsNode
        ):
            raise InvalidOptionError(option_path=leaf_name)

        node._children[leaf_name].value = value

    def reset_option(self, path: str) -> None:
        """Reset a single option to its default value.

        Parameters
        ----------
        path : str
            Path to the option. Must be in the format "category.option_name" or "category.subcategory.option_name"

        Examples
        --------
        Set an option to a non-default value:
        >>> pypsa.options.set_option("general.allow_network_requests", False)
        >>> pypsa.options.general.allow_network_requests
        False

        Reset just that option:
        >>> pypsa.options.reset_option("general.allow_network_requests")
        >>> pypsa.options.general.allow_network_requests
        True

        """
        parts = path.split(".")
        node = self

        # Navigate to the parent
        for part in parts[:-1]:
            if part not in node._children or not isinstance(
                node._children[part], OptionsNode
            ):
                raise InvalidOptionError(option_path=part)
            node = node._children[part]

        # Reset the option value
        leaf_name = parts[-1]
        if leaf_name not in node._children or isinstance(
            node._children[leaf_name], OptionsNode
        ):
            raise InvalidOptionError(option_path=leaf_name)

        node._children[leaf_name].reset()

    def reset_all(self) -> None:
        """Reset all options to their default values.

        Examples
        --------
        Define some options:
        >>> pypsa.options.general.allow_network_requests = False
        >>> pypsa.options.params.statistics.round = 4

        Reset all options:
        >>> pypsa.options.reset_all()
        >>> pypsa.options.general.allow_network_requests
        True
        >>> pypsa.options.params.statistics.round
        5

        """
        for child in self._children.values():
            if isinstance(child, Option):
                child.reset()
            else:
                child.reset_all()

    def describe_options(self, prefix: str = "") -> None:
        """Print documentation for a subset of options.

        Parameters
        ----------
        prefix : str
            Prefix for the option path. Used for nested options.
            If empty, the root options are printed.

        Examples
        --------
        Print only params.statistics options:

        >>> pypsa.options.params.statistics.describe_options()
        PyPSA Options
        =============
        drop_zero:
            Default: True
            Description: Default value for the 'drop_zero' parameter in statistics module.
        nice_names:
            Default: True
            Description: Default value for the 'nice_names' parameter in statistics module.
        round:
            Default: 5
            Description: Default value for the 'round' parameter in statistics module.

        Or print all options:
        >>> pypsa.options.describe_options() # doctest: +ELLIPSIS
        PyPSA Options
        =============
        api.new_components_api:
            Default: False
            Description: Activate the new components API, which replaces the static components data access with the more flexible components class. This will just change the api and not any functionality. Components class features are always available. See https://go.pypsa.org/new-components-api for more details.
        debug.runtime_verification:
            Default: False
            Description: Enable runtime verification of PyPSA's internal state. This is useful for debugging and development purposes. This will lead to overhead in performance and should not be used in production.
        general.allow_network_requests:
            Default: True
            Description: Allow PyPSA to make network requests. When False, all network requests (such as checking for version updates) are disabled. This may be needed in restricted environments, offline usage, or for security/privacy reasons. This only controls PyPSA's own network requests, dependencies may still make network requests independently.
        params.statistics.drop_zero:
            Default: True
            Description: Default value for the 'drop_zero' parameter in statistics module.
        params.statistics.nice_names:
            Default: True
            Description: Default value for the 'nice_names' parameter in statistics module.
        ...

        """
        if not prefix:
            print("PyPSA Options\n=============")  # noqa: T201

        for name, child in sorted(self._children.items()):
            path = f"{prefix}.{name}" if prefix else name

            if isinstance(child, Option):
                print(f"{path}:")  # noqa: T201
                print(f"    Default: {child._default}")  # noqa: T201
                print(f"    Description: {child._docs}")  # noqa: T201
            else:
                child.describe_options(path)

    def describe(self) -> None:
        """Print documentation for all options.

        This is a convenience method to call describe_options() without a prefix.

        Examples
        --------
        >>> pypsa.options.describe() # doctest: +ELLIPSIS
        PyPSA Options
        =============
        api.new_components_api:
            Default: False
            Description: Activate the new components API, which replaces the static components data access with the more flexible components class. This will just change the api and not any functionality. Components class features are always available. See https://go.pypsa.org/new-components-api for more details.
        debug.runtime_verification:
            Default: False
            Description: Enable runtime verification of PyPSA's internal state. This is useful for debugging and development purposes. This will lead to overhead in performance and should not be used in production.
        general.allow_network_requests:
            Default: True
            Description: Allow PyPSA to make network requests. When False, all network requests (such as checking for version updates) are disabled. This may be needed in restricted environments, offline usage, or for security/privacy reasons. This only controls PyPSA's own network requests, dependencies may still make network requests independently.
        params.statistics.drop_zero:
            Default: True
            Description: Default value for the 'drop_zero' parameter in statistics module.
        params.statistics.nice_names:
            Default: True
            Description: Default value for the 'nice_names' parameter in statistics module.
        ...

        """
        self.describe_options()


options = OptionsNode()


@contextmanager
def option_context(*args: Any) -> Generator[None, None, None]:
    """Context manager to temporarily set options.

    Parameters
    ----------
    *args : str, Any
        Must be passed in pairs of option_name and value.
        Option_name must be in the format "category.option_name" or "category.subcategory.option_name"

    """
    if len(args) % 2 != 0:
        msg = "Arguments must be paired option_names and values"
        raise ValueError(msg)

    # Get the original values and set the temporary ones
    pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
    original_values: dict[str, Any] = {}

    try:
        # Store original values and set new ones
        for option_name, value in pairs:
            original_values[option_name] = options.get_option(option_name)
            options.set_option(option_name, value)
        yield
    finally:
        # Restore original values
        for option_name, original_value in original_values.items():
            options.set_option(option_name, original_value)


# Setup options
# =============

# General category
options._add_option(
    "general.allow_network_requests",
    True,
    "Allow PyPSA to make network requests. When False, all network requests "
    "(such as checking for version updates) are disabled. This may be needed "
    "in restricted environments, offline usage, or for security/privacy reasons. "
    "This only controls PyPSA's own network requests, dependencies may still "
    "make network requests independently.",
)

# Debugging category
options._add_option(
    "debug.runtime_verification",
    False,
    "Enable runtime verification of PyPSA's internal state. This is useful for "
    "debugging and development purposes. This will lead to overhead in "
    "performance and should not be used in production.",
)

# API
options._add_option(
    "api.new_components_api",
    False,
    "Activate the new components API, which replaces the static components data access "
    "with the more flexible components class. This will just change the api and not any "
    "functionality. Components class features are always available. "
    "See https://go.pypsa.org/new-components-api for more details.",
)

# Warnings category
options._add_option(
    "warnings.components_store_iter",
    True,
    "If False, suppresses the deprecatio warning when iterating over components. ",
)

# Parameters category
options._add_option(
    "params.statistics.nice_names",
    True,
    "Default value for the 'nice_names' parameter in statistics module.",
)
options._add_option(
    "params.statistics.drop_zero",
    True,
    "Default value for the 'drop_zero' parameter in statistics module.",
)
options._add_option(
    "params.statistics.round",
    5,
    "Default value for the 'round' parameter in statistics module.",
)
