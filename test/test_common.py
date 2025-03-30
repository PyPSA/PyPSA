import warnings

import numpy as np
import pandas as pd
import pytest

from pypsa.common import (
    MethodHandlerWrapper,
    as_index,
    check_pypsa_version,
    deprecated_common_kwargs,
    deprecated_kwargs,
    equals,
    future_deprecation,
    list_as_string,
    rename_kwargs,
)


def test_decorator_with_arguments():
    """Test the decorator when used with arguments: @MethodHandlerWrapper(handler_class=...)"""

    class ResultHandler:
        def __init__(self, method):
            self.method = method

        def __call__(self, *args, **kwargs):
            result = self.method(*args, **kwargs)
            return f"Processed: {result}"

    class TestClass:
        def __init__(self, value=10):
            self.value = value

        @MethodHandlerWrapper(handler_class=ResultHandler)
        def method_with_decorator_args(self, x):
            """Test method with decorator args"""
            return self.value + x

    test_instance = TestClass()
    result = test_instance.method_with_decorator_args(5)
    assert result == "Processed: 15"


def test_decorator_without_arguments():
    """Test the decorator when used without arguments: @MethodHandlerWrapper"""

    class ResultHandler:
        def __init__(self, method):
            self.method = method

        def __call__(self, *args, **kwargs):
            result = self.method(*args, **kwargs)
            return f"Processed: {result}"

    wrapper = MethodHandlerWrapper(handler_class=ResultHandler)

    class TestClass:
        def __init__(self, value=10):
            self.value = value

        @wrapper
        def method_with_simple_decorator(self, x):
            """Test method with simple decorator"""
            return self.value + x

    test_instance = TestClass()
    result = test_instance.method_with_simple_decorator(5)
    assert result == "Processed: 15"


def test_class_method_access():
    """Test accessing the decorated method at the class level"""

    class ResultHandler:
        def __init__(self, method):
            self.method = method

        def __call__(self, *args, **kwargs):
            result = self.method(*args, **kwargs)
            return f"Processed: {result}"

    class TestClass:
        def __init__(self, value=10):
            self.value = value

        @MethodHandlerWrapper(handler_class=ResultHandler)
        def method(self, x):
            return self.value + x

    # Should return the wrapper itself, not the handler instance
    assert isinstance(TestClass.method, MethodHandlerWrapper)


@pytest.mark.parametrize(
    "attr, expected_name",
    [
        ("snapshots", "snapshot"),
        ("investment_periods", "period"),
    ],
)
def test_as_index(ac_dc_network_mi, attr, expected_name):
    n = ac_dc_network_mi

    # Test with None values
    result = as_index(n, None, attr)
    assert isinstance(result, pd.Index)
    assert result.equals(getattr(n, attr))
    assert result.name == expected_name

    # Test with valid values
    values = getattr(n, attr)[:3]
    result = as_index(n, values, attr)
    assert isinstance(result, pd.Index)
    assert result.equals(pd.Index(values))
    assert result.name == expected_name

    # Test with different levels
    # with pytest.raises(ValueError):
    #     as_index(n, n.snapshots, attr)

    # Test with invalid values
    with pytest.raises(ValueError):
        as_index(n, ["invalid"], attr)

    # # Test with scalar value
    # scalar_result = as_index(n, getattr(n, attr)[0], attr, expected_name)
    # assert isinstance(scalar_result, pd.Index)
    # assert scalar_result.equals(pd.Index([getattr(n, attr)[0]]))
    # assert scalar_result.name == expected_name


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 1, True),
        (1, 2, False),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), True),
        (np.array([1, 2, 3]), np.array([1, 2, 4]), False),
        (pd.DataFrame({"A": [1, 2]}), pd.DataFrame({"A": [1, 2]}), True),
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}, False),
        ([1, 2, 3], [1, 2, 3], True),
        (np.nan, np.nan, True),
    ],
)
def test_equals(a, b, expected):
    assert equals(a, b) == expected


def test_equals_ignored_classes():
    class IgnoredClass:
        def __init__(self, value=1):
            self.value = value

    assert equals(
        IgnoredClass(value=1), IgnoredClass(value=2), ignored_classes=[IgnoredClass]
    )


def test_equals_type_mismatch():
    with pytest.raises(AssertionError):
        equals(1, "1")


@pytest.fixture
def warning_catcher():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


def test_deprecated_kwargs(warning_catcher):
    @deprecated_kwargs(old_arg="new_arg")
    def test_func(new_arg):
        return new_arg

    result = test_func(old_arg="value")
    assert result == "value"
    assert len(warning_catcher) == 1
    assert issubclass(warning_catcher[0].category, DeprecationWarning)
    assert "old_arg" in str(warning_catcher[0].message)


def test_rename_kwargs():
    kwargs = {"old_arg": "value"}
    aliases = {"old_arg": "new_arg"}
    with pytest.warns(DeprecationWarning):
        rename_kwargs("test_func", kwargs, aliases)
    assert "new_arg" in kwargs
    assert "old_arg" not in kwargs
    assert kwargs["new_arg"] == "value"


def test_deprecated_common_kwargs(warning_catcher):
    @deprecated_common_kwargs
    def test_func(n):
        return n

    result = test_func(network="value")
    assert result == "value"
    assert len(warning_catcher) == 1
    assert issubclass(warning_catcher[0].category, DeprecationWarning)
    assert "network" in str(warning_catcher[0].message)


def test_future_deprecation(warning_catcher):
    @future_deprecation(activate=False)
    def test_func_inactive():
        return "not deprecated"

    result = test_func_inactive()
    assert result == "not deprecated"
    assert len(warning_catcher) == 0

    @future_deprecation(activate=True)
    def test_func_active():
        return "deprecated"

    result = test_func_active()
    assert result == "deprecated"
    assert len(warning_catcher) == 1
    assert issubclass(warning_catcher[0].category, DeprecationWarning)


def test_list_as_string():
    # Test comma-separated (default)
    assert list_as_string(["a", "b", "c"]) == "a, b, c"
    # Test bullet-list
    expected_bullet = "  - x\n  - y\n  - z"
    assert (
        list_as_string(["x", "y", "z"], prefix="  ", style="bullet-list")
        == expected_bullet
    )
    # Test dict input
    assert list_as_string({"a": 1, "b": 2, "c": 3}) == "a, b, c"
    # Test empty list
    assert list_as_string([]) == ""
    # Test single item
    assert list_as_string(["a"]) == "a"
    # Test invalid style
    with pytest.raises(ValueError):
        list_as_string(["a", "b"], style="invalid")
    # Test prefix
    assert list_as_string(["a", "b"], prefix="-> ") == "-> a, b"
    # Test empty lists
    assert list_as_string([]) == ""
    assert list_as_string([], style="bullet-list") == ""


def test_version_check(caplog):
    check_pypsa_version("0.20.0")
    assert caplog.text == ""

    check_pypsa_version("0.0")
    assert "The correct version of PyPSA could not be resolved" in caplog.text
