import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pypsa.common import (
    MethodHandlerWrapper,
    UnexpectedError,
    _check_for_update,
    as_index,
    equals,
    list_as_string,
)


def test_unexpected_error_message_formatting():
    """Test that UnexpectedError correctly formats the error message with the issue tracker URL."""
    # Test with a custom message
    custom_message = "Something unexpected happened"
    error = UnexpectedError(custom_message)

    # Verify the error message contains both the custom message and the URL
    assert custom_message in str(error)
    assert UnexpectedError.URL_CREATE_ISSUE in str(error)
    assert "Please track this issue in our issue tracker:" in str(error)

    # Test with an empty message
    empty_error = UnexpectedError()
    assert "Please track this issue in our issue tracker:" in str(empty_error)
    assert UnexpectedError.URL_CREATE_ISSUE in str(empty_error)


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
    ("attr", "expected_name"),
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


# Tests for the Comparator class
class TestEquals:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (1, 1, True),
            (1, 2, False),
            (np.array([1, 2, 3]), np.array([1, 2, 3]), True),
            (np.array([1, 2, 3]), np.array([1, 2, 4]), False),
            (pd.DataFrame({"A": [1, 2]}), pd.DataFrame({"A": [1, 2]}), True),
            (pd.DataFrame({"A": [1, 2]}), pd.DataFrame({"A": [1, 4]}), False),
            ({"a": 1, "b": 2}, {"a": 1, "b": 3}, False),
            ([1, 2, 3], [1, 2, 3], True),
            (np.nan, np.nan, True),
            # Additional test cases
            ("string", "string", True),
            ("string", "different", False),
            (None, None, True),
            (True, True, True),
            (True, False, False),
            ([], [], True),
            ({}, {}, True),
            ((1, 2), (1, 2), True),
            ((1, 2), (1, 3), False),
            ({1, 2}, {1, 2}, True),
            ({1, 2}, {1, 3}, False),
            # Same object identity
            (lambda x: x, lambda x: x, False),  # Functions with different identity
        ],
    )
    def test_equals(self, a, b, expected):
        assert equals(a, b) == expected

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (1, 2),
            ("a", "b"),
            (np.array([1, 2, 3]), np.array([1, 2, 4])),
            (pd.DataFrame({"A": [1, 3]}), pd.DataFrame({"A": [1, 2]})),
        ],
    )
    def test_equals_logs(self, a, b, caplog):
        assert equals(a, b, log_mode="silent") is False
        assert caplog.text == ""

        assert equals(a, b, log_mode="verbose") is False
        assert caplog.text != ""

        with pytest.raises(ValueError):
            equals(a, b, log_mode="strict")

        with pytest.raises(ValueError):
            equals(a, b, log_mode="invalid")

    def test_equals_ignored_classes(self):
        class IgnoredClass:
            def __init__(self, value=1):
                self.value = value

        assert equals(
            IgnoredClass(value=1), IgnoredClass(value=2), ignored_classes=[IgnoredClass]
        )

    def test_equals_type_mismatch(self):
        with pytest.raises(ValueError):
            equals(1, "1", log_mode="strict")

    def test_invalid_log_mode_type(self):
        with pytest.raises(ValueError, match="'log_mode' must be one of"):
            equals(1, 1, log_mode=123)

    def test_nested_structures(self):
        a = {"level1": {"level2": [1, 2, {"level3": "value"}]}}
        b = {"level1": {"level2": [1, 2, {"level3": "different"}]}}

        assert equals(a, a) is True
        assert equals(a, b) is False

    def test_pandas_series(self):
        a = pd.Series([1, 2, 3])
        b = pd.Series([1, 2, 3])
        c = pd.Series([1, 2, 4])

        assert equals(a, b) is True
        assert equals(a, c) is False

    def test_pandas_empty_dataframes(self):
        a = pd.DataFrame()
        b = pd.DataFrame()

        assert equals(a, b) is True

    def test_numpy_arrays_with_nan(self):
        a = np.array([1, 2, np.nan])
        b = np.array([1, 2, np.nan])

        assert equals(a, b) is True

    def test_same_object_identity(self):
        obj = {"complex": "object"}
        assert equals(obj, obj) is True

    def test_dict_with_missing_keys(self):
        a = {"key1": 1, "key2": 2}
        b = {"key1": 1}
        c = {"key1": 1, "key2": 2, "key3": 3}

        assert equals(a, b) is False
        assert equals(a, c) is False


@pytest.fixture
def warning_catcher():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


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


@pytest.fixture
def mock_response():
    """Create a mock response object."""

    class MockResponse:
        def __init__(self, data):
            self._data = data

        def read(self):
            return f'{{"tag_name": "v{self._data}"}}'.encode()

    return MockResponse


@pytest.mark.parametrize(
    ("current", "latest", "expected_message"),
    [
        # Test case format: (current_version, latest_version, expected_message)
        (
            (1, 0, 0),
            "2.0.0",
            "New version 2.0.0 available! (Current: 1.0.0)",
        ),  # newer version
        ((1, 0, 0), "1.0.0", ""),  # same version
        ((2, 0, 0), "1.0.0", ""),  # current is newer
        (
            (1, 2, 2),
            "1.2.3",
            "New version 1.2.3 available! (Current: 1.2.2)",
        ),  # minor update
    ],
)
def test_check_for_update(mock_response, current, latest, expected_message):
    """Test version comparison scenarios."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = mock_response(latest)
        _check_for_update.cache_clear()
        result = _check_for_update(current, "test_owner", "test_repo")
        assert result == expected_message


def test_check_for_update_error_handling():
    """Test error handling scenarios."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = Exception("Connection failed")

        _check_for_update.cache_clear()
        result = _check_for_update((1, 0, 0), "test_owner", "test_repo")
        assert result == ""


def test_check_for_update_respects_network_option(mock_response):
    """Test that version check respects the allow_network_requests option."""
    import pypsa

    # Test that version check is skipped when network requests are disabled
    with pypsa.option_context("general.allow_network_requests", False):
        result = _check_for_update((1, 0, 0), "test_owner", "test_repo")
        assert result == ""

    # Test that version check works when network requests are allowed
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = mock_response("2.0.0")

        _check_for_update.cache_clear()
        with pypsa.option_context("general.allow_network_requests", True):
            result = _check_for_update((1, 0, 0), "test_owner", "test_repo")
            assert "New version 2.0.0 available!" in result


def test_url_loading_respects_network_option():
    """Test that URL loading respects the allow_network_requests option."""
    import pypsa
    from pypsa.network.io import _retrieve_from_url

    # Test that URL loading is blocked when network requests are disabled
    def dummy_function(x):
        return None

    with pypsa.option_context("general.allow_network_requests", False):
        with pytest.raises(ValueError, match="Network requests are disabled"):
            _retrieve_from_url("https://example.com/test.nc", dummy_function)

    # Test that URL loading works when network requests are allowed
    with pypsa.option_context("general.allow_network_requests", True):
        with patch("pypsa.network.io.urlretrieve") as mock_urlretrieve:
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test"

                def mock_io_function(x):
                    return "test_result"

                result = _retrieve_from_url(
                    "https://example.com/test.nc", mock_io_function
                )
                assert result == "test_result"
                mock_urlretrieve.assert_called_once()


def test_network_constructor_respects_network_option():
    """Test that Network constructor respects the allow_network_requests option when loading from URLs."""
    import pypsa

    # Test that Network loading is blocked when network requests are disabled
    with pypsa.option_context("general.allow_network_requests", False):
        with pytest.raises(ValueError, match="Network requests are disabled"):
            pypsa.Network("https://example.com/test.nc")
