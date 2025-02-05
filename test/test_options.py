import pytest

import pypsa


def test_getter():
    # Default init and get
    pypsa.options.warnings.components_store_iter = True
    assert pypsa.options.warnings.components_store_iter is True
    with pytest.raises(AttributeError):
        pypsa.options.warnings.invalid_option
    with pytest.raises(AttributeError):
        pypsa.options.invalid_category.invalid_option


def test_setter():
    pypsa.options.warnings.components_store_iter = False
    assert pypsa.options.warnings.components_store_iter is False
    with pytest.raises(AttributeError):
        pypsa.options.warnings.invalid_option = False
    with pytest.raises(AttributeError):
        pypsa.options.invalid_category.invalid_option = False


def test_getter_method():
    pypsa.options.warnings.components_store_iter = True
    assert pypsa.get_option("warnings.components_store_iter") is True
    pypsa.options.warnings.components_store_iter = False
    assert pypsa.get_option("warnings.components_store_iter") is False

    with pytest.raises(
        AttributeError,
        match="Invalid option 'warnings.invalid_option'. Check options via pypsa.options.describe_options()",
    ):
        pypsa.get_option("warnings.invalid_option")
    with pytest.raises(
        AttributeError,
        match="Invalid option 'invalid_warning.invalid_option'. Check options via pypsa.options.describe_options()",
    ):
        pypsa.get_option("invalid_warning.invalid_option")


def test_setter_method():
    pypsa.set_option("warnings.components_store_iter", False)
    assert pypsa.options.warnings.components_store_iter is False
    assert pypsa.get_option("warnings.components_store_iter") is False

    with pytest.raises(
        AttributeError,
        match="Invalid option 'warnings.invalid_option'. Check options via pypsa.options.describe_options()",
    ):
        pypsa.set_option("warnings.invalid_option", False)

    with pytest.raises(
        AttributeError,
        match="Invalid option 'invalid_warning.invalid_option'. Check options via pypsa.options.describe_options()",
    ):
        pypsa.set_option("invalid_warning.invalid_option", False)


def test_describe_method():
    pypsa.describe_options()


def test_option_context():
    """Test option_context functionality."""
    # Basic usage
    pypsa.options.warnings.components_store_iter = True
    assert pypsa.options.warnings.components_store_iter is True
    with pypsa.option_context("warnings.components_store_iter", False):
        assert pypsa.options.warnings.components_store_iter is False
    assert pypsa.options.warnings.components_store_iter is True

    # Nested contexts
    with pypsa.option_context("warnings.components_store_iter", False):
        assert pypsa.options.warnings.components_store_iter is False
        with pypsa.option_context("warnings.components_store_iter", True):
            assert pypsa.options.warnings.components_store_iter is True
        assert pypsa.options.warnings.components_store_iter is False

    # Exception handling
    with pytest.raises(ValueError):
        with pypsa.option_context("warnings.components_store_iter", False):
            raise ValueError()
    assert pypsa.options.warnings.components_store_iter is True

    # Invalid arguments
    with pytest.raises(ValueError, match="Arguments must be paired"):
        with pypsa.option_context("warnings.components_store_iter"):
            pass

    with pytest.raises(AttributeError):
        with pypsa.option_context("invalid.option", True):
            pass

    # Different value types
    test_values = [1, "test", None, 3.14, [1, 2, 3]]
    for val in test_values:
        with pypsa.option_context("warnings.components_store_iter", val):
            assert pypsa.options.warnings.components_store_iter == val
        assert pypsa.options.warnings.components_store_iter is True
