import pytest

import pypsa

# Add options for testing
pypsa.options._add_option("test.test_option", True, "Test option")
pypsa.options._add_option("test.nested.test_option", False, "Another test option")


def test_getter():
    # Default init and get
    pypsa.options.test.test_option = True
    assert pypsa.options.test.test_option is True
    with pytest.raises(AttributeError):
        pypsa.options.test.invalid_option
    with pytest.raises(AttributeError):
        pypsa.options.invalid_category.invalid_option

    # Nested
    pypsa.options.test.nested.test_option = False
    assert pypsa.options.test.nested.test_option is False
    with pytest.raises(AttributeError):
        pypsa.options.test.nested.invalid_option
    with pytest.raises(AttributeError):
        pypsa.options.invalid_category.test.nested.invalid_option


def test_setter():
    pypsa.options.test.test_option = False
    assert pypsa.options.test.test_option is False
    with pytest.raises(AttributeError):
        pypsa.options.test.invalid_option = False
    with pytest.raises(AttributeError):
        pypsa.options.invalid_category.invalid_option = False

    # Nested
    pypsa.options.test.nested.test_option = False
    assert pypsa.options.test.nested.test_option is False
    with pytest.raises(AttributeError):
        pypsa.options.test.test_options.invalid_option = False
    with pytest.raises(AttributeError):
        pypsa.options.invalid_category.some_stuff.invalid_option = False


def test_getter_method():
    pypsa.options.test.test_option = True
    assert pypsa.get_option("test.test_option") is True
    pypsa.options.test.test_option = False
    assert pypsa.get_option("test.test_option") is False

    with pytest.raises(AttributeError, match="Invalid option"):
        pypsa.get_option("test.invalid_option")
    with pytest.raises(AttributeError, match="Invalid option"):
        pypsa.get_option("test.invalid_option")

    # Nested
    pypsa.options.test.nested.test_option = True
    assert pypsa.get_option("test.nested.test_option") is True
    pypsa.options.test.nested.test_option = False
    assert pypsa.get_option("test.nested.test_option") is False
    with pytest.raises(AttributeError, match="Invalid option"):
        pypsa.get_option("test.test_options.invalid_option")
    with pytest.raises(AttributeError, match="Invalid option"):
        pypsa.get_option("invalid_warning.some_stuff.invalid_option")


def test_setter_method():
    pypsa.set_option("test.test_option", False)
    assert pypsa.options.test.test_option is False
    assert pypsa.get_option("test.test_option") is False

    with pytest.raises(AttributeError, match="Invalid option"):
        pypsa.set_option("test.invalid_option", False)

    with pytest.raises(AttributeError, match="Invalid option"):
        pypsa.set_option("test.invalid_option", False)

    # Nested
    pypsa.set_option("test.nested.test_option", False)
    assert pypsa.options.test.nested.test_option is False
    assert pypsa.get_option("test.nested.test_option") is False
    with pytest.raises(AttributeError, match="Invalid option"):
        pypsa.set_option("test.test_options.invalid_option", False)


def test_describe_method(capsys):
    pypsa.describe_options()
    all_options = capsys.readouterr().out

    assert all_options.startswith("PyPSA Options")
    assert "test.test_option" in all_options
    assert "test.nested.test_option" in all_options

    pypsa.options.describe_options()
    all_options_module = capsys.readouterr().out
    assert all_options == all_options_module

    # Test options with no description
    pypsa.options.test.nested.describe_options()
    nested_options = capsys.readouterr().out
    assert "test.nested.test_option" not in nested_options
    assert "test.test_option" not in nested_options
    assert "test_option" in nested_options


def test_option_context():
    """Test option_context functionality."""
    # Basic usage
    pypsa.options.test.test_option = True
    assert pypsa.options.test.test_option is True
    with pypsa.option_context("test.test_option", False):
        assert pypsa.options.test.test_option is False
    assert pypsa.options.test.test_option is True

    # Nested contexts
    with pypsa.option_context("test.test_option", False):
        assert pypsa.options.test.test_option is False
        with pypsa.option_context("test.test_option", True):
            assert pypsa.options.test.test_option is True
        assert pypsa.options.test.test_option is False

    # Exception handling
    with pytest.raises(ValueError):
        with pypsa.option_context("test.test_option", False):
            raise ValueError()
    assert pypsa.options.test.test_option is True

    # Invalid arguments
    with pytest.raises(ValueError, match="Arguments must be paired"):
        with pypsa.option_context("test.test_option"):
            pass

    with pytest.raises(AttributeError):
        with pypsa.option_context("invalid.option", True):
            pass

    # Different value types
    test_values = [1, "test", None, 3.14, [1, 2, 3]]
    for val in test_values:
        with pypsa.option_context("test.test_option", val):
            assert pypsa.options.test.test_option == val
        assert pypsa.options.test.test_option is True


def test_nested_option_context():
    """Test nested option_context functionality."""
    # Basic usage
    pypsa.options.test.nested.test_option = True
    assert pypsa.options.test.nested.test_option is True
    with pypsa.option_context("test.nested.test_option", False):
        assert pypsa.options.test.nested.test_option is False
    assert pypsa.options.test.nested.test_option is True

    # Nested contexts
    with pypsa.option_context("test.nested.test_option", False):
        assert pypsa.options.test.nested.test_option is False
        with pypsa.option_context("test.nested.test_option", True):
            assert pypsa.options.test.nested.test_option is True
        assert pypsa.options.test.nested.test_option is False

    # Exception handling
    with pytest.raises(ValueError):
        with pypsa.option_context("test.nested.test_option", False):
            raise ValueError()
    assert pypsa.options.test.nested.test_option is True

    # Invalid arguments
    with pytest.raises(ValueError, match="Arguments must be paired"):
        with pypsa.option_context("test.nested.test_option"):
            pass

    with pytest.raises(AttributeError):
        with pypsa.option_context("invalid.option", True):
            pass

    # Different value types
    test_values = [1, "test", None, 3.14, [1, 2, 3]]
    for val in test_values:
        with pypsa.option_context("test.nested.test_option", val):
            assert pypsa.options.test.nested.test_option == val
        assert pypsa.options.test.nested.test_option is True
