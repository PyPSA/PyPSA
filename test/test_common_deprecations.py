import warnings
from unittest.mock import patch

import pytest

from pypsa.common import (
    deprecated_common_kwargs,
    deprecated_in_next_major,
    deprecated_kwargs,
    deprecated_namespace,
    rename_deprecated_kwargs,
)


@pytest.fixture
def mock_version_semver():
    with patch("pypsa.common.__version_semver__", "1.0.0"):
        yield


def test_rename_deprecated_kwargs_basic(mock_version_semver):
    """Test basic functionality of rename_deprecated_kwargs."""
    func_name = "test_func"
    kwargs = {"old_arg": "value"}
    aliases = {"old_arg": "new_arg"}
    deprecated_in = "0.9.0"
    removed_in = "1.1.0"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rename_deprecated_kwargs(func_name, kwargs, aliases, deprecated_in, removed_in)

        # Check that a warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "`old_arg` is deprecated as an argument to `test_func`" in str(
            w[0].message
        )
        assert "Deprecated in version 0.9.0" in str(w[0].message)
        assert "Will be removed in version 1.1.0" in str(w[0].message)

    # Check that the argument was renamed
    assert "new_arg" in kwargs
    assert "old_arg" not in kwargs
    assert kwargs["new_arg"] == "value"


def test_rename_deprecated_kwargs_both_args_present(mock_version_semver):
    """Test that an error is raised when both old and new arguments are provided."""
    func_name = "test_func"
    kwargs = {"old_arg": "old_value", "new_arg": "new_value"}
    aliases = {"old_arg": "new_arg"}
    deprecated_in = "0.9.0"
    removed_in = "1.1.0"

    with pytest.raises(DeprecationWarning) as excinfo:
        rename_deprecated_kwargs(func_name, kwargs, aliases, deprecated_in, removed_in)

    assert "received both old_arg and new_arg as arguments" in str(excinfo.value)


def test_rename_deprecated_kwargs_no_deprecated_args(mock_version_semver):
    """Test that no warnings are raised when no deprecated arguments are used."""
    func_name = "test_func"
    kwargs = {"new_arg": "value"}
    aliases = {"old_arg": "new_arg"}
    deprecated_in = "0.9.0"
    removed_in = "1.1.0"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rename_deprecated_kwargs(func_name, kwargs, aliases, deprecated_in, removed_in)

        # Check that no warnings were raised
        assert len(w) == 0

    # Check that the arguments remain unchanged
    assert "new_arg" in kwargs
    assert kwargs["new_arg"] == "value"


def test_rename_deprecated_kwargs_multiple_aliases(mock_version_semver):
    """Test handling multiple aliases at once."""
    func_name = "test_func"
    kwargs = {"old_arg1": "value1", "old_arg2": "value2"}
    aliases = {"old_arg1": "new_arg1", "old_arg2": "new_arg2"}
    deprecated_in = "0.9.0"
    removed_in = "1.1.0"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rename_deprecated_kwargs(func_name, kwargs, aliases, deprecated_in, removed_in)

        # Check that warnings were raised
        assert len(w) == 2

    # Check that the arguments were renamed
    assert "new_arg1" in kwargs
    assert "new_arg2" in kwargs
    assert "old_arg1" not in kwargs
    assert "old_arg2" not in kwargs
    assert kwargs["new_arg1"] == "value1"
    assert kwargs["new_arg2"] == "value2"


def test_deprecated_kwargs_decorator(mock_version_semver):
    """Test the deprecated_kwargs decorator."""

    @deprecated_kwargs(deprecated_in="0.9.0", removed_in="1.1.0", old_arg="new_arg")
    def test_func(new_arg):
        return new_arg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = test_func(old_arg="value")

        # Check that a warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "`old_arg` is deprecated as an argument to `test_func`" in str(
            w[0].message
        )

    # Check that the function worked correctly
    assert result == "value"


def test_deprecated_common_kwargs(mock_version_semver):
    """Test the deprecated_common_kwargs decorator."""

    @deprecated_common_kwargs
    def test_func(n):
        return n

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = test_func(network="value")

        # Check that a warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "`network` is deprecated as an argument to `test_func`" in str(
            w[0].message
        )

    # Check that the function worked correctly
    assert result == "value"


def test_deprecated_in_next_major(mock_version_semver):
    """Test the deprecated_in_next_major decorator."""

    @deprecated_in_next_major("This function will be removed in version 2.0")
    def test_func():
        return "value"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = test_func()

        # Check that a warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "is deprecated as of 1.0" in str(w[0].message).lower()
        assert "will be removed in 2.0" in str(w[0].message).lower()

    # Check that the function worked correctly
    assert result == "value"


def test_deprecated_namespace(mock_version_semver):
    """Test the deprecated_namespace decorator."""

    def test_func():
        return "value"

    decorated_func = deprecated_namespace(
        test_func,
        previous_module="old.module",
        deprecated_in="0.9.0",
        removed_in="1.1.0",
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = decorated_func()

        # Check that a warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "old.module" in str(w[0].message)
        assert "Deprecated since version 0.9.0" in str(w[0].message)
        assert "Will be removed in version 1.1.0" in str(w[0].message)

    # Check that the function worked correctly
    assert result == "value"
