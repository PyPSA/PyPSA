from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


class Option:
    def __init__(self, value: Any = None) -> None:
        self._value = value
        self._default = None
        self._docs = ""

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


class OptionCategory:
    _valid_options: set = set()
    _option_defaults: dict = {}
    _option_docs: dict = {}

    def __init__(self) -> None:
        self._options: dict = {}

    def __getattr__(self, name: str) -> Any:
        if name not in self._valid_options:
            raise AttributeError(
                f"Invalid option '{name}' for this category. "
                f"Valid options are: {', '.join(sorted(self._valid_options))}"
            )

        if name not in self._options:
            option = Option()
            option._default = self._option_defaults.get(name)
            option._value = option._default
            option._docs = self._option_docs.get(name, "")
            self._options[name] = option
        # Just return the raw value
        return self._options[name]._value

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        if name not in self._valid_options:
            raise AttributeError(
                f"Invalid option '{name}' for this category. "
                f"Valid options are: {', '.join(sorted(self._valid_options))}"
            )

        if name not in self._options:
            option = Option()
            option._default = self._option_defaults.get(name)
            option._docs = self._option_docs.get(name, "")
            self._options[name] = option
        self._options[name]._value = value

    def reset_all(self) -> None:
        """Reset all options in this category to their defaults."""
        for option in self._options.values():
            option.reset()


class WarningsCategory(OptionCategory):
    _valid_options = {"components_store_iter"}

    # Define defaults and docs using a setup method
    @classmethod
    def setup(cls) -> None:
        cls._option_defaults = {"components_store_iter": True}
        cls._option_docs = {"components_store_iter": "Some Description"}


class _Options:
    _valid_categories = {"warnings": WarningsCategory}

    def __init__(self) -> None:
        self._categories: dict = {}
        # Setup all categories
        for category_cls in self._valid_categories.values():
            category_cls.setup()

    def __getattr__(self, category_name: str) -> None:
        if category_name not in self._valid_categories:
            raise AttributeError(
                f"Invalid category '{category_name}'. "
                f"Valid categories are: {', '.join(sorted(self._valid_categories.keys()))}"
            )

        if category_name not in self._categories:
            self._categories[category_name] = self._valid_categories[category_name]()
        return self._categories[category_name]

    def _describe_options(self) -> None:
        """Print documentation for all options."""
        print("PyPSA Options\n=============")
        for category_name, category_cls in self._valid_categories.items():
            for option_name in sorted(category_cls._valid_options):
                docs = category_cls._option_docs.get(
                    option_name, "No documentation available"
                )
                default = category_cls._option_defaults.get(
                    option_name, "No default value"
                )
                print(f"{category_name}.{option_name}:")
                print(f"    Default: {default}")
                print(f"    Description: {docs}")


# Create a global options instance
options = _Options()


def get_option(option_name: str) -> Any:
    """
    Get value for option

    Parameters
    ----------
    option_name : str
        The name of the option in the form "category.option_name"

    Returns
    -------
    Any
        The value of the option

    See Also
    --------
    set_option : Set value for option
    describe_options : Print documentation for all options

    Examples
    --------
    >>> import pypsa
    >>> pypsa.get_option("warnings.components_store_iter")
    True

    or

    >>> pypsa.options.warnings.components_store_iter
    True
    """
    category, name = option_name.split(".")
    try:
        return getattr(getattr(options, category), name)
    except AttributeError:
        msg = f"Invalid option '{option_name}'. Check options via pypsa.options.describe_options()"
        raise AttributeError(msg)


def set_option(option_name: str, value: Any) -> None:
    """
    Set value for option

    Parameters
    ----------
    option_name : str
        The name of the option in the form "category.option_name"
    value : Any
        The value to set

    Returns
    -------
    None

    See Also
    --------
    get_option : Get value for option
    describe_options : Print documentation for all options

    Examples
    --------
    >>> import pypsa
    >>> pypsa.set_option("warnings.components_store_iter", False)

    or

    >>> pypsa.options.warnings.components_store_iter = False
    """

    category, name = option_name.split(".")
    try:
        setattr(getattr(options, category), name, value)
    except AttributeError:
        msg = f"Invalid option '{option_name}'. Check options via pypsa.options.describe_options()"
        raise AttributeError(msg)


def describe_options() -> None:
    """
    Print documentation for all options.

    See Also
    --------
    get_option : Get value for option
    set_option : Set value for option

    Examples
    --------
    >>> import pypsa
    >>> pypsa.describe_options()
    PyPSA Options
    =============
    warnings.components_store_iter:
        Default: True
        Description: Some Description
    """
    options._describe_options()


@contextmanager
def option_context(*args: Any) -> Generator[None, None, None]:
    """
    Context manager to temporarily set options.

    Parameters
    ----------
    *args : str, Any
        Must be passed in pairs of option_name and value.
        Option_name must be in the format "category.option_name"

    Returns
    -------
    None

    See Also
    --------
    get_option : Get value for option
    set_option : Set value for option
    describe_options : Print documentation for all options

    Examples
    --------
    >>> import pypsa
    >>> with pypsa.option_context("warnings.components_store_iter", False):
    ...     # do something with temporary option value
    ...     print(pypsa.get_option("warnings.components_store_iter"))
    False
    >>> # options are reverted to original values
    >>> print(pypsa.get_option("warnings.components_store_iter"))
    True
    """
    if len(args) % 2 != 0:
        raise ValueError("Arguments must be paired option_names and values")

    # Get the original values and set the temporary ones
    pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
    original_values: dict[str, Any] = {}

    try:
        # Store original values and set new ones
        for option_name, value in pairs:
            original_values[option_name] = get_option(option_name)
            set_option(option_name, value)

        yield

    finally:
        # Restore original values
        for option_name, original_value in original_values.items():
            set_option(option_name, original_value)
