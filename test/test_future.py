# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import importlib
import os
import warnings

import pytest

import pypsa
from pypsa import _options
from pypsa._future import PyPSA2Warning, future_flag

pytestmark = pytest.mark.usefixtures("restore_options_module")

FLAG = "test_flag"
DOCS = "A throwaway flag."
WARNING = "From PyPSA 2.0 on, the throwaway flag is on."


@pytest.fixture
def flag():
    """Register a throwaway flag, unset whatever PYPSA_FUTURE is."""
    _options.options._add_future_flag(FLAG, DOCS, WARNING)
    return FLAG


def _reload_with_env(monkeypatch, **env):
    for key in os.environ:
        if key.startswith(("PYPSA_FUTURE", "PYPSA_API__")):
            monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    importlib.reload(_options)
    return _options.options


def test_warning_class():
    assert issubclass(PyPSA2Warning, FutureWarning)
    assert pypsa.PyPSA2Warning is PyPSA2Warning


def test_unset_warns_once_and_keeps_current_behaviour(flag):
    assert pypsa.options.future.test_flag is None
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert future_flag(FLAG) is False
        assert future_flag(FLAG) is False
    assert [w.category for w in rec] == [PyPSA2Warning]
    assert str(rec[0].message) == (
        f"{WARNING} Set `pypsa.options.future.{FLAG}` to True to opt in now or "
        "to False to keep the current behaviour."
    )
    assert rec[0].filename == __file__


def test_unset_flag_without_warning_is_silent():
    _options.options._add_future_flag(FLAG, DOCS, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert future_flag(FLAG) is False


@pytest.mark.parametrize("value", [True, False])
def test_set_flag_is_silent(flag, value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pypsa.option_context(f"future.{FLAG}", value):
            assert future_flag(FLAG) is value


@pytest.mark.parametrize("value", ["no", "False", 1, 0, "x"])
def test_invalid_python_value_rejected(flag, value):
    with pytest.raises(ValueError, match="True, False or None"):
        pypsa.options.future.test_flag = value
    assert pypsa.options.future.test_flag is None


def test_per_flag_env(monkeypatch):
    opts = _reload_with_env(monkeypatch, PYPSA_FUTURE__INDS_ORDER="true")
    assert opts.future.inds_order is True


def test_umbrella_env_sets_defaults(monkeypatch):
    opts = _reload_with_env(monkeypatch, PYPSA_FUTURE="1")
    assert opts.future.inds_order is True
    assert opts.future.string_dtype is True

    opts.future.inds_order = False
    opts.reset_all()
    assert opts.future.inds_order is True


def test_per_flag_beats_alias_beats_umbrella(monkeypatch):
    with pytest.warns(DeprecationWarning, match="future.string_dtype"):
        opts = _reload_with_env(
            monkeypatch,
            PYPSA_FUTURE="false",
            PYPSA_API__LEGACY_STRING_DTYPE="false",
            PYPSA_FUTURE__STRING_DTYPE="false",
        )
    assert opts.future.string_dtype is False

    with pytest.warns(DeprecationWarning, match="future.string_dtype"):
        opts = _reload_with_env(
            monkeypatch, PYPSA_FUTURE="false", PYPSA_API__LEGACY_STRING_DTYPE="false"
        )
    assert opts.future.string_dtype is True


@pytest.mark.parametrize(("value", "logged"), [("maybe", True), ("", False)])
def test_invalid_umbrella_logs(monkeypatch, caplog, value, logged):
    with caplog.at_level("WARNING"):
        opts = _reload_with_env(monkeypatch, PYPSA_FUTURE=value)
    assert ("PYPSA_FUTURE" in caplog.text) is logged
    assert opts.future.inds_order is None


def test_invalid_flag_env_logs(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        opts = _reload_with_env(monkeypatch, PYPSA_FUTURE__INDS_ORDER="no")
    assert "PYPSA_FUTURE__INDS_ORDER" in caplog.text
    assert opts.future.inds_order is None


def test_alias(flag, capsys):
    opts = _options.options
    opts._add_alias("api.old", f"future.{FLAG}", invert=True)

    assert opts.api.old is None
    with pytest.warns(DeprecationWarning, match=f"future.{FLAG}") as rec:
        opts.api.old = False
    assert rec[0].filename == __file__
    assert opts.future.test_flag is True
    assert opts.api.old is False
    with pytest.warns(DeprecationWarning, match=f"future.{FLAG}"):
        opts.set_option("api.old", True)
    assert opts.future.test_flag is False

    with (
        pytest.warns(DeprecationWarning, match="api.old"),
        pytest.raises(ValueError, match="True, False or None"),
    ):
        opts.set_option("api.old", "maybe")

    opts.reset_option("api.old")
    assert opts.future.test_flag is None
    with (
        pytest.warns(DeprecationWarning, match="api.old"),
        _options.option_context("api.old", True),
    ):
        assert opts.future.test_flag is False
    assert opts.future.test_flag is None

    opts.describe()
    assert "api.old" not in capsys.readouterr().out
