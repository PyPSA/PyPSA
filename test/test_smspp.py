# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Tests for the optional SMS++ accessor."""

import importlib.machinery
import sys
import types

import pytest

import pypsa


class FakeSMSppResult:
    """Minimal pypsa2smspp result object."""

    solution = object()
    objective_value = 12.0
    status = "10 (Success)"


class FakeSMSNetwork:
    """Minimal SMS++ network proxy."""


class FakeTransformation:
    """Fake pypsa2smspp Transformation recording the pipeline order."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.sms_network = None
        self.result = None
        self.instances.append(self)

    def create_model(self, n, verbose=True):
        self.calls.append("create_model")
        self.sms_network = FakeSMSNetwork()
        return self.sms_network

    def optimize(self, verbose=True):
        self.calls.append("optimize")
        self.result = FakeSMSppResult()
        return self.result

    def retrieve_solution(self, n, verbose=True):
        self.calls.append("retrieve_solution")
        n._smspp_inverse_objective = self.result.objective_value
        return n


def install_fake_smspp(monkeypatch, tmp_path):
    """Install fake pypsa2smspp and pysmspp modules for unit tests."""
    package_dir = tmp_path / "pypsa2smspp"
    data_dir = package_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "config_default.yaml").write_text("smspp: {}\n", encoding="utf-8")

    fake_pypsa2smspp = types.ModuleType("pypsa2smspp")
    fake_pypsa2smspp.__file__ = str(package_dir / "__init__.py")
    fake_pypsa2smspp.__path__ = [str(package_dir)]
    fake_pypsa2smspp.__spec__ = importlib.machinery.ModuleSpec(
        "pypsa2smspp",
        loader=None,
        is_package=True,
    )
    fake_pypsa2smspp.Transformation = FakeTransformation

    fake_pysmspp = types.ModuleType("pysmspp")

    def is_smspp_installed():
        return True

    fake_pysmspp.is_smspp_installed = is_smspp_installed

    FakeTransformation.instances = []
    monkeypatch.setitem(sys.modules, "pypsa2smspp", fake_pypsa2smspp)
    monkeypatch.setitem(sys.modules, "pysmspp", fake_pysmspp)


def test_smspp_accessor_runs_staged_pipeline(monkeypatch, tmp_path):
    install_fake_smspp(monkeypatch, tmp_path)

    n = pypsa.Network()

    status, condition = n.optimize.smspp(solver_options={"name": "custom"})

    assert (status, condition) == ("ok", "10 (Success)")
    assert FakeTransformation.instances[0].calls == [
        "create_model",
        "optimize",
        "retrieve_solution",
    ]
    assert FakeTransformation.instances[0].kwargs == {"name": "custom"}
    assert n._smspp_inverse_objective == 12.0


def test_optimize_smspp_returns_status_tuple(monkeypatch, tmp_path):
    install_fake_smspp(monkeypatch, tmp_path)

    n = pypsa.Network()

    status, condition = n.optimize(solver_name="smspp", solver_options={})

    assert (status, condition) == ("ok", "10 (Success)")
    assert FakeTransformation.instances[0].kwargs == {}


def test_smspp_solving():
    pytest.importorskip("pypsa2smspp")
    pysmspp = pytest.importorskip("pysmspp")
    if not pysmspp.is_smspp_installed():
        pytest.skip("SMS++ not installed")

    nsms = pypsa.examples.ac_dc_meshed()

    # global constraints not yet supported
    nsms.remove("GlobalConstraint", nsms.global_constraints.index)

    # create base case on a fresh copy
    norig = nsms.copy()
    norig.optimize(solver_name="highs")

    nsms.optimize(solver_name="smspp")

    assert nsms.objective + nsms.objective_constant == pytest.approx(
        norig.objective + norig.objective_constant, rel=1e-3
    )
