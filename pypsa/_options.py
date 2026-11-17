# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import ast
import logging
import os
import warnings
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
            message = f"Invalid option '{option_path}'. Check 'options.describe()' for valid options."
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


class FutureOption(Option):
    """Flag for a PyPSA 2.0 behaviour change: None (unset), True or False."""

    def __init__(self, docs: str = "", warning: str | None = None) -> None:
        super().__init__(None, None, docs)
        self.warning = warning

    @property
    def value(self) -> bool | None:
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        if value is not None and not isinstance(value, bool):
            msg = f"Future flags accept only True, False or None, got {value!r}."
            raise ValueError(msg)
        self._value = value


class AliasOption(Option):
    """Deprecated key that reads and writes another option, optionally inverted."""

    def __init__(
        self, root: "OptionsNode", old_path: str, new_path: str, invert: bool
    ) -> None:
        super().__init__(None, None, f"Deprecated alias of '{new_path}'.")
        self._root = root
        self._old_path = old_path
        self._new_path = new_path
        self._invert = invert

    def _map(self, value: Any) -> Any:
        return (not value) if self._invert and isinstance(value, bool) else value

    @property
    def value(self) -> Any:
        return self._map(self._root.get_option(self._new_path))

    @value.setter
    def value(self, value: Any) -> None:
        warnings.warn(
            f"'{self._old_path}' is deprecated, use '{self._new_path}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        self._root.set_option(self._new_path, self._map(value))

    def reset(self) -> None:
        self._root.reset_option(self._new_path)


class OptionsNode:
    """PyPSA package options.

    <!-- md:badge-version v0.33.0 --> | <!-- md:guide options.md -->

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

    def _get_node(self, path: str) -> Option:
        """Return the `Option` object stored at `path`."""
        node = self
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in node._children or not isinstance(
                node._children[part], OptionsNode
            ):
                raise InvalidOptionError(option_path=part)
            node = node._children[part]
        leaf = node._children.get(parts[-1])
        if not isinstance(leaf, Option):
            raise InvalidOptionError(option_path=parts[-1])
        return leaf

    def _replace_leaf(self, path: str, option: Option) -> None:
        self._add_option(path)
        parent, _, name = path.rpartition(".")
        node = self
        for part in parent.split("."):
            node = node._children[part]
        node._children[name] = option

    def _add_future_flag(self, name: str, docs: str, warning: str | None) -> None:
        """Register the flag `future.<name>`, unset (None) by default.

        `warning` is shown while the flag is unset: what changes in 2.0 and how to
        prepare for it. With `warning=None` the unset flag stays silent.
        """
        self._replace_leaf(f"future.{name}", FutureOption(docs, warning))

    def _add_alias(self, old_path: str, new_path: str, *, invert: bool) -> None:
        """Register `old_path` as a deprecated alias of `new_path`."""
        self._get_node(new_path)  # target must exist
        self._replace_leaf(old_path, AliasOption(self, old_path, new_path, invert))

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
        return self._get_node(path).value

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
        self._get_node(path).value = value

    def reset_option(self, path: str) -> None:
        """Reset a single option to its default value.

        Parameters
        ----------
        path : str
            Path to the option. Must be in the format "category.option_name" or "category.subcategory.option_name"

        Examples
        --------
        Set an option to a non-default value:
        >>> pypsa.set_option("general.allow_network_requests", False)
        >>> pypsa.options.general.allow_network_requests
        False

        Reset just that option:
        >>> pypsa.reset_option("general.allow_network_requests")
        >>> pypsa.options.general.allow_network_requests
        True

        """
        self._get_node(path).reset()

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

    def _load_from_env(self) -> None:
        """Load options from PYPSA_* environment variables.

        `PYPSA_FUTURE` sets the default of every `future.*` flag. It is read first
        and `PYPSA_FUTURE__<NAME>` last, so a per-flag variable wins.
        """
        prefix = "PYPSA_"
        env = sorted(
            (item for item in os.environ.items() if item[0].startswith(prefix)),
            key=lambda item: (item[0] != "PYPSA_FUTURE", "FUTURE__" in item[0]),
        )

        for env_var, env_value in env:
            # PYPSA_GENERAL__ALLOW_NETWORK_REQUESTS -> general.allow_network_requests
            option_path = env_var.removeprefix(prefix).replace("__", ".").lower()

            lower_value = env_value.lower()
            # Handle common booleans
            if lower_value in ("true", "1"):
                parsed_value: Any = True
            elif lower_value in ("false", "0"):
                parsed_value = False
            # Otherwise use literal_eval for Python literals
            else:
                try:
                    parsed_value = ast.literal_eval(env_value)
                except (ValueError, SyntaxError):
                    parsed_value = env_value
                    logger.debug(
                        "Could not parse '%s' as literal, using string", env_var
                    )

            if env_var == "PYPSA_FUTURE":
                if isinstance(parsed_value, bool):
                    for flag in self._children["future"]._children.values():
                        flag._default = flag._value = parsed_value
                elif env_value.strip():
                    logger.warning(
                        "Invalid value '%s' for env var 'PYPSA_FUTURE', "
                        "use true or false.",
                        env_value,
                    )
                continue

            try:
                self.set_option(option_path, parsed_value)
                logger.debug("Set option '%s' from env var '%s'", option_path, env_var)
            except InvalidOptionError:
                logger.warning(
                    "Unknown option '%s' from env var '%s'. "
                    "Use pypsa.options.describe() to see valid options.",
                    option_path,
                    env_var,
                )
            except ValueError as e:
                logger.warning("Invalid value from env var '%s': %s", env_var, e)

    def _describe_options(self, prefix: str = "") -> None:
        """Print documentation for options via path.

        Parameters
        ----------
        prefix : str
            Prefix for the option path. Used for nested options.
            If empty, the root options are printed.



        """
        if not prefix:
            print("PyPSA Options\n=============")  # noqa: T201

        for name, child in self._children.items():
            path = f"{prefix}.{name}" if prefix else name

            if isinstance(child, AliasOption):
                continue
            if isinstance(child, Option):
                print(f"{path}:")  # noqa: T201
                print(f"    Default: {child._default}")  # noqa: T201
                print(f"    Description: {child._docs}")  # noqa: T201
            else:
                child._describe_options(path)

    def describe(self) -> None:
        """Print documentation for options node via attribute access.

        Examples
        --------
        Print all options:

        >>> pypsa.options.describe() # doctest: +ELLIPSIS
        PyPSA Options
        =============
        general.allow_network_requests:
            Default: True
            Description: Allow PyPSA to make network requests...
        ...

        """
        self._describe_options()


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
    "Allow PyPSA to make network requests. When False, all network requests\n\t"
    "(such as checking for version updates) are disabled. This may be needed\n\t"
    "in restricted environments, offline usage, or for security/privacy reasons.\n\t"
    "This only controls PyPSA's own network requests, dependencies may still\n\t"
    "make network requests independently.",
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

options._add_option(
    "params.add.return_names",
    False,
    "Default value for the 'return_names' parameter in Network.add method.\n\t"
    "If True, the add method returns the names of added components.\n\t"
    "If False, it returns None.",
)

options._add_option(
    "params.optimize.model_kwargs",
    {},
    "Default value for the 'model_kwargs' parameter in optimization module.",
)
options._add_option(
    "params.optimize.solver_name",
    "highs",
    "Default value for the 'solver_name' parameter in optimization module.",
)
options._add_option(
    "params.optimize.solver_options",
    {},
    "Default value for the 'solver_options' parameter in optimization module.",
)
options._add_option(
    "params.optimize.log_to_console",
    None,
    "Whether to print solver output to console. Passed as a solver option\n\t"
    "to linopy's Model.solve(). When None, solver default behavior is used.\n\t"
    "Note: not all solvers support this option (e.g. HiGHS does, CPLEX does not).",
)

options._add_option(
    "params.consistency.numerical_tolerance",
    1e-9,
    "Tolerance for numerical comparisons in consistency checks (e.g. p_min_pu > p_max_pu). "
    "Values within this tolerance of the boundary are not flagged as inconsistent. "
    "Set to 0 to disable tolerance and flag any violation.",
)
# Optimization category
options._add_option(
    "optimization.model_snapshot_index",
    "auto",
    "Snapshot dimension of multi-period models: 'multiindex' keeps the "
    "(period, timestep) MultiIndex, 'flat' uses tuple labels "
    "with period/timestep auxiliary coordinates, 'auto' picks by linopy's active "
    "semantics ('flat' under v1, else 'multiindex'). Applies to newly built "
    "models only.",
)
# Warnings category
options._add_option(
    "warnings.components_store_iter",
    True,
    "If False, suppresses the deprecation warning when iterating over components.",
)
options._add_option(
    "warnings.attribute_typos",
    True,
    "If False, suppresses warnings about potential typos in component attribute names. "
    "Note: warnings about unintended attributes (standard attributes for other components) "
    "will still be shown.",
)

# Future flags: None keeps the current behaviour and warns, True adopts 2.0
options._add_future_flag(
    "string_dtype",
    "Keep pandas' `str` dtype for string data on import instead of converting it to object.",
    "pandas reads strings as `str` dtype since 3.0. PyPSA converts them back to "
    "object dtype on import, but keeps `str` from PyPSA 2.0 on.",
)
options._add_alias("api.legacy_string_dtype", "future.string_dtype", invert=True)
options._add_future_flag(
    "components_api",
    "`n.<name>` returns the `Components` object instead of the static DataFrame, "
    "and `n.<name>_t` is deprecated. See `https://go.pypsa.org/new-components-api`.",
    # Silent while unset: `n.<name>` is the most common access path, so a warning
    # there would reach nearly every user.
    None,
)
options._add_alias("api.new_components_api", "future.components_api", invert=False)
options._add_future_flag(
    "exclude_objective_constant",
    "Leave the capital costs of existing capacity on extendable assets out of the "
    "objective, so `n.objective_constant` is zero and the LP is better conditioned.",
    "From PyPSA 2.0 on, the objective leaves out the capital costs of existing "
    "capacity on extendable assets, which improves LP conditioning. Pass "
    "`include_objective_constant` to `n.optimize` to choose per call.",
)
options._add_alias(
    "params.optimize.include_objective_constant",
    "future.exclude_objective_constant",
    invert=True,
)
options._add_future_flag(
    "inds_order",
    "`get_switchable_as_dense` with `inds` returns columns in `inds` order, keeps "
    "duplicates and raises a `KeyError` for unknown names.",
    "From PyPSA 2.0 on, `get_switchable_as_dense` returns columns in `inds` order, "
    "keeps duplicates and raises a `KeyError` for unknown names. Unique, known "
    "names in component order give the same result in both.",
)

# Debugging category
options._add_option(
    "debug.runtime_verification",
    False,
    "Enable runtime verification of PyPSA's internal state. This is useful for\n\t"
    "debugging and development purposes. This will lead to overhead in\n\t"
    "performance and should not be used in production.",
)

# Load environment variables from .env file if python-dotenv is installed
try:
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        logger.debug("Loaded environment variables from '%s'", dotenv_path)
except ImportError:
    pass

# Load options from environment variables
options._load_from_env()
