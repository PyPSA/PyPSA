# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pytest

import pypsa


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
    mocked_pypsa.options._describe_options()
    all_options = capsys.readouterr().out

    assert all_options.startswith("PyPSA Options")
    assert "test.test_option" in all_options
    assert "test.nested.test_option" in all_options

    mocked_pypsa.options._describe_options()
    all_options_module = capsys.readouterr().out
    assert all_options == all_options_module

    # Test options with no description
    mocked_pypsa.options.test.nested.describe()
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


def test_add_return_names_option():
    """Test the params.add.return_names option."""
    import pandas as pd

    import pypsa

    n = pypsa.Network()

    # Default: option is False, returns None
    assert pypsa.get_option("params.add.return_names") is False
    assert n.add("Bus", "bus1") is None

    # Set option to True, now returns Index
    pypsa.set_option("params.add.return_names", True)
    result = n.add("Bus", "bus2")
    assert isinstance(result, pd.Index)
    assert result[0] == "bus2"

    # Explicit parameter overrides option
    assert n.add("Bus", "bus3", return_names=False) is None
    pypsa.set_option("params.add.return_names", False)
    result = n.add("Bus", "bus4", return_names=True)
    assert isinstance(result, pd.Index)
    assert result[0] == "bus4"

    # Test with option_context
    with pypsa.option_context("params.add.return_names", True):
        result = n.add("Bus", "bus5")
        assert isinstance(result, pd.Index)
        assert result[0] == "bus5"
    assert n.add("Bus", "bus6") is None  # Back to False


def test_params_optimize():
    n = pypsa.examples.ac_dc_meshed()

    n.optimize()
    assert n.model.solver_name == "highs"

    n.optimize.create_model()
    n.optimize.solve_model()
    assert n.model.solver_name == "highs"

    with pypsa.option_context("params.optimize.solver_name", "gurobi"):
        n.optimize()
        assert n.model.solver_name == "gurobi"

        n.optimize.create_model()
        n.optimize.solve_model()
        assert n.model.solver_name == "gurobi"


@pytest.mark.parametrize(
    ("env_var", "env_value", "option_path", "expected"),
    [
        # String
        (
            "PYPSA_PARAMS__OPTIMIZE__SOLVER_NAME",
            "gurobi",
            "params.optimize.solver_name",
            "gurobi",
        ),
        # Booleans
        (
            "PYPSA_GENERAL__ALLOW_NETWORK_REQUESTS",
            "true",
            "general.allow_network_requests",
            True,
        ),
        (
            "PYPSA_GENERAL__ALLOW_NETWORK_REQUESTS",
            "false",
            "general.allow_network_requests",
            False,
        ),
        (
            "PYPSA_GENERAL__ALLOW_NETWORK_REQUESTS",
            "1",
            "general.allow_network_requests",
            True,
        ),
        (
            "PYPSA_GENERAL__ALLOW_NETWORK_REQUESTS",
            "0",
            "general.allow_network_requests",
            False,
        ),
        (
            "PYPSA_GENERAL__ALLOW_NETWORK_REQUESTS",
            "TRUE",
            "general.allow_network_requests",
            True,
        ),
        (
            "PYPSA_GENERAL__ALLOW_NETWORK_REQUESTS",
            "FALSE",
            "general.allow_network_requests",
            False,
        ),
        # Integer
        ("PYPSA_PARAMS__STATISTICS__ROUND", "3", "params.statistics.round", 3),
        # Dict
        (
            "PYPSA_PARAMS__OPTIMIZE__SOLVER_OPTIONS",
            "{'threads': 4}",
            "params.optimize.solver_options",
            {"threads": 4},
        ),
        # Malformed dict falls back to string
        (
            "PYPSA_PARAMS__OPTIMIZE__SOLVER_NAME",
            "{'threads': 4",
            "params.optimize.solver_name",
            "{'threads': 4",
        ),
    ],
)
def test_env_var_parsing(monkeypatch, env_var, env_value, option_path, expected):
    """Test parsing of a couple of different environment variables."""
    import importlib

    from pypsa import _options

    monkeypatch.setenv(env_var, env_value)
    importlib.reload(_options)

    assert _options.options.get_option(option_path) == expected

    monkeypatch.delenv(env_var)
    importlib.reload(_options)


def test_invalid_env_var_warns(monkeypatch, caplog):
    """Test that invalid env vars log a warning."""
    import importlib
    import logging

    from pypsa import _options

    monkeypatch.setenv("PYPSA_INVALID__OPTION", "value")

    with caplog.at_level(logging.WARNING):
        importlib.reload(_options)

    assert "Unknown option" in caplog.text

    monkeypatch.delenv("PYPSA_INVALID__OPTION")
    importlib.reload(_options)


def test_option_priority(monkeypatch):
    """Test that priority holds: function args > runtime > env vars > defaults."""
    import importlib

    from pypsa import _options

    assert _options.options.params.optimize.solver_name == "highs"

    # Env var overrides default
    monkeypatch.setenv("PYPSA_PARAMS__OPTIMIZE__SOLVER_NAME", "gurobi")
    importlib.reload(_options)
    assert _options.options.params.optimize.solver_name == "gurobi"

    # Runtime overrides env var
    _options.options.set_option("params.optimize.solver_name", "cplex")
    assert _options.options.params.optimize.solver_name == "cplex"

    monkeypatch.delenv("PYPSA_PARAMS__OPTIMIZE__SOLVER_NAME")
    importlib.reload(_options)
