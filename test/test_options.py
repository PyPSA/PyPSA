import pytest


@pytest.fixture
def mocked_pypsa():
    import pypsa

    # Add options for testing
    pypsa.options._add_option("test.test_option", True, "Test option")
    pypsa.options._add_option("test.nested.test_option", False, "Another test option")

    return pypsa


def test_getter(mocked_pypsa):
    # Default init and get
    mocked_pypsa.options.test.test_option = True
    assert mocked_pypsa.options.test.test_option is True
    with pytest.raises(AttributeError):
        mocked_pypsa.options.test.invalid_option
    with pytest.raises(AttributeError):
        mocked_pypsa.options.invalid_category.invalid_option

    # Nested
    mocked_pypsa.options.test.nested.test_option = False
    assert mocked_pypsa.options.test.nested.test_option is False
    with pytest.raises(AttributeError):
        mocked_pypsa.options.test.nested.invalid_option
    with pytest.raises(AttributeError):
        mocked_pypsa.options.invalid_category.test.nested.invalid_option


def test_setter(mocked_pypsa):
    mocked_pypsa.options.test.test_option = False
    assert mocked_pypsa.options.test.test_option is False
    with pytest.raises(AttributeError):
        mocked_pypsa.options.test.invalid_option = False
    with pytest.raises(AttributeError):
        mocked_pypsa.options.invalid_category.invalid_option = False

    # Nested
    mocked_pypsa.options.test.nested.test_option = False
    assert mocked_pypsa.options.test.nested.test_option is False
    with pytest.raises(AttributeError):
        mocked_pypsa.options.test.test_options.invalid_option = False
    with pytest.raises(AttributeError):
        mocked_pypsa.options.invalid_category.some_stuff.invalid_option = False


def test_getter_method(mocked_pypsa):
    mocked_pypsa.options.test.test_option = True
    assert mocked_pypsa.get_option("test.test_option") is True
    mocked_pypsa.options.test.test_option = False
    assert mocked_pypsa.get_option("test.test_option") is False

    with pytest.raises(AttributeError, match="Invalid option"):
        mocked_pypsa.get_option("test.invalid_option")
    with pytest.raises(AttributeError, match="Invalid option"):
        mocked_pypsa.get_option("test.invalid_option")

    # Nested
    mocked_pypsa.options.test.nested.test_option = True
    assert mocked_pypsa.get_option("test.nested.test_option") is True
    mocked_pypsa.options.test.nested.test_option = False
    assert mocked_pypsa.get_option("test.nested.test_option") is False
    with pytest.raises(AttributeError, match="Invalid option"):
        mocked_pypsa.get_option("test.test_options.invalid_option")
    with pytest.raises(AttributeError, match="Invalid option"):
        mocked_pypsa.get_option("invalid_warning.some_stuff.invalid_option")


def test_setter_method(mocked_pypsa):
    mocked_pypsa.set_option("test.test_option", False)
    assert mocked_pypsa.options.test.test_option is False
    assert mocked_pypsa.get_option("test.test_option") is False

    with pytest.raises(AttributeError, match="Invalid option"):
        mocked_pypsa.set_option("test.invalid_option", False)

    with pytest.raises(AttributeError, match="Invalid option"):
        mocked_pypsa.set_option("test.invalid_option", False)

    # Nested
    mocked_pypsa.set_option("test.nested.test_option", False)
    assert mocked_pypsa.options.test.nested.test_option is False
    assert mocked_pypsa.get_option("test.nested.test_option") is False
    with pytest.raises(AttributeError, match="Invalid option"):
        mocked_pypsa.set_option("test.test_options.invalid_option", False)


def test_describe_method(capsys, mocked_pypsa):
    mocked_pypsa.describe_options()
    all_options = capsys.readouterr().out

    assert all_options.startswith("PyPSA Options")
    assert "test.test_option" in all_options
    assert "test.nested.test_option" in all_options

    mocked_pypsa.options.describe_options()
    all_options_module = capsys.readouterr().out
    assert all_options == all_options_module

    # Test options with no description
    mocked_pypsa.options.test.nested.describe_options()
    nested_options = capsys.readouterr().out
    assert "test.nested.test_option" not in nested_options
    assert "test.test_option" not in nested_options
    assert "test_option" in nested_options


def test_option_context(mocked_pypsa):
    """Test option_context functionality."""
    # Basic usage
    mocked_pypsa.options.test.test_option = True
    assert mocked_pypsa.options.test.test_option is True
    with mocked_pypsa.option_context("test.test_option", False):
        assert mocked_pypsa.options.test.test_option is False
    assert mocked_pypsa.options.test.test_option is True

    # Nested contexts
    with mocked_pypsa.option_context("test.test_option", False):
        assert mocked_pypsa.options.test.test_option is False
        with mocked_pypsa.option_context("test.test_option", True):
            assert mocked_pypsa.options.test.test_option is True
        assert mocked_pypsa.options.test.test_option is False

    # Exception handling
    with pytest.raises(ValueError):
        with mocked_pypsa.option_context("test.test_option", False):
            raise ValueError()
    assert mocked_pypsa.options.test.test_option is True

    # Invalid arguments
    with pytest.raises(ValueError, match="Arguments must be paired"):
        with mocked_pypsa.option_context("test.test_option"):
            pass

    with pytest.raises(AttributeError):
        with mocked_pypsa.option_context("invalid.option", True):
            pass

    # Different value types
    test_values = [1, "test", None, 3.14, [1, 2, 3]]
    for val in test_values:
        with mocked_pypsa.option_context("test.test_option", val):
            assert mocked_pypsa.options.test.test_option == val
        assert mocked_pypsa.options.test.test_option is True


def test_nested_option_context(mocked_pypsa):
    """Test nested option_context functionality."""
    # Basic usage
    mocked_pypsa.options.test.nested.test_option = True
    assert mocked_pypsa.options.test.nested.test_option is True
    with mocked_pypsa.option_context("test.nested.test_option", False):
        assert mocked_pypsa.options.test.nested.test_option is False
    assert mocked_pypsa.options.test.nested.test_option is True

    # Nested contexts
    with mocked_pypsa.option_context("test.nested.test_option", False):
        assert mocked_pypsa.options.test.nested.test_option is False
        with mocked_pypsa.option_context("test.nested.test_option", True):
            assert mocked_pypsa.options.test.nested.test_option is True
        assert mocked_pypsa.options.test.nested.test_option is False

    # Exception handling
    with pytest.raises(ValueError):
        with mocked_pypsa.option_context("test.nested.test_option", False):
            raise ValueError()
    assert mocked_pypsa.options.test.nested.test_option is True

    # Invalid arguments
    with pytest.raises(ValueError, match="Arguments must be paired"):
        with mocked_pypsa.option_context("test.nested.test_option"):
            pass

    with pytest.raises(AttributeError):
        with mocked_pypsa.option_context("invalid.option", True):
            pass

    # Different value types
    test_values = [1, "test", None, 3.14, [1, 2, 3]]
    for val in test_values:
        with mocked_pypsa.option_context("test.nested.test_option", val):
            assert mocked_pypsa.options.test.nested.test_option == val
        assert mocked_pypsa.options.test.nested.test_option is True


def test_general_allow_network_requests():
    """Test the general.allow_network_requests option."""
    import pypsa

    # Test default value
    assert pypsa.get_option("general.allow_network_requests") is True

    # Test setting to False
    pypsa.set_option("general.allow_network_requests", False)
    assert pypsa.get_option("general.allow_network_requests") is False

    # Test setting back to True
    pypsa.set_option("general.allow_network_requests", True)
    assert pypsa.get_option("general.allow_network_requests") is True

    # Test using option_context
    with pypsa.option_context("general.allow_network_requests", False):
        assert pypsa.get_option("general.allow_network_requests") is False
    assert pypsa.get_option("general.allow_network_requests") is True
