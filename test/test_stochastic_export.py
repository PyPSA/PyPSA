# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Test the mpi-sppy file exporter and driver (pypsa.optimization.stochastic_mpisppy).

Most tests cover the dependency-free phases -- ``write_stochastic_problem_mpisppy``,
``read_stochastic_solution_mpisppy`` and the command-building / orchestration parts of
``solve_stochastic_mpisppy`` (with the solver subprocess mocked out) -- and therefore
need neither mpi-sppy nor a solver. The end-to-end ``solve_stochastic_mpisppy`` test is
skipped unless mpi-sppy and Gurobi are installed.
"""

import importlib.util
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from linopy import available_solvers

import pypsa
from pypsa.descriptors import nominal_attrs
from pypsa.optimization import stochastic_mpisppy as stochastic

# Build models without the existing-infrastructure objective constant (avoids a
# FutureWarning) and without the consistency check (avoids carrier warnings).
MODEL_KWARGS = {"include_objective_constant": False, "consistency_check": False}

_HAS_MPISPPY = importlib.util.find_spec("mpisppy") is not None
_HAS_GUROBI = "gurobi" in available_solvers
_HAS_MPIEXEC = shutil.which("mpiexec") is not None


@pytest.fixture
def stochastic_export_network():
    """A tiny two-bus, three-scenario network with four first-stage capacities."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=4, freq="h"))
    n.add("Bus", "elec")
    n.add("Bus", "heat")
    n.add("Load", "elec-load", bus="elec", p_set=[100, 120, 90, 110])
    n.add("Load", "heat-load", bus="heat", p_set=[40, 30, 50, 45])
    n.add(
        "Generator",
        "wind",
        bus="elec",
        p_nom_extendable=True,
        capital_cost=1000,
        marginal_cost=0,
        p_max_pu=[0.3, 0.6, 0.2, 0.5],
    )
    n.add(
        "Generator",
        "gas",
        bus="elec",
        p_nom_extendable=True,
        capital_cost=500,
        marginal_cost=50,
    )
    n.add(
        "Link",
        "p2h",
        bus0="elec",
        bus1="heat",
        p_nom_extendable=True,
        capital_cost=300,
        efficiency=0.9,
    )
    n.add(
        "Generator",
        "boiler",
        bus="heat",
        p_nom_extendable=True,
        capital_cost=200,
        marginal_cost=30,
    )
    n.set_scenarios({"low": 0.3, "med": 0.4, "high": 0.3})
    for scenario, gas_mc in [("low", 40), ("med", 50), ("high", 70)]:
        n.generators.loc[(scenario, "gas"), "marginal_cost"] = gas_mc
    return n


def _read_json(path):
    return json.loads(Path(path).read_text())


def test_write_creates_scenario_files(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    manifest = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, model_kwargs=MODEL_KWARGS
    )

    stems = [sf["stem"] for sf in manifest["scenario_files"]]
    assert stems == ["scenario0", "scenario1", "scenario2"]
    for stem in stems:
        for suffix in (".lp", "_nonants.json", "_rho.csv"):
            assert (tmp_path / f"{stem}{suffix}").exists()
    assert (tmp_path / "pypsa_stochastic_manifest.json").exists()
    # Human-readable PyPSA names are preserved in the metadata.
    assert [sf["name"] for sf in manifest["scenario_files"]] == ["low", "med", "high"]


def test_nonants_json_schema(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    n.optimize.write_stochastic_problem_mpisppy(tmp_path, model_kwargs=MODEL_KWARGS)

    payload = _read_json(tmp_path / "scenario0_nonants.json")
    assert set(payload) == {"scenarioData", "treeData"}
    assert payload["scenarioData"]["scenProb"] == 0.3
    assert payload["scenarioData"]["name"] == "low"
    root = payload["treeData"]["nodes"]["ROOT"]
    assert root["condProb"] == 1.0
    assert root["serialNumber"] == 0
    # Implicit linopy names: opaque x{label}, no special characters.
    assert root["nonAnts"] == ["x0", "x1", "x2", "x3"]


def test_nonant_lists_identical_across_scenarios(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    n.optimize.write_stochastic_problem_mpisppy(tmp_path, model_kwargs=MODEL_KWARGS)

    nonants = {
        stem: _read_json(tmp_path / f"{stem}_nonants.json")["treeData"]["nodes"][
            "ROOT"
        ]["nonAnts"]
        for stem in ("scenario0", "scenario1", "scenario2")
    }
    assert nonants["scenario0"] == nonants["scenario1"] == nonants["scenario2"]


def test_cost_proportional_rho(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    manifest = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, model_kwargs=MODEL_KWARGS
    )

    # rho_i = |capital_cost_i| for the default alpha=1.0.
    assert manifest["rho"] == {"x0": 1000.0, "x1": 500.0, "x2": 200.0, "x3": 300.0}
    # rho is scenario-invariant: every scenario's _rho.csv is byte-identical.
    rho_texts = {
        (tmp_path / f"{stem}_rho.csv").read_text()
        for stem in ("scenario0", "scenario1", "scenario2")
    }
    assert len(rho_texts) == 1
    assert next(iter(rho_texts)).splitlines()[0] == "varname,rho"


def test_nonant_map(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    manifest = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, model_kwargs=MODEL_KWARGS
    )

    assert manifest["nonant_map"] == {
        "x0": {"component": "Generator", "attr": "p_nom", "asset": "wind"},
        "x1": {"component": "Generator", "attr": "p_nom", "asset": "gas"},
        "x2": {"component": "Generator", "attr": "p_nom", "asset": "boiler"},
        "x3": {"component": "Link", "attr": "p_nom", "asset": "p2h"},
    }


def test_flat_and_mapping_rho(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    flat = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, rho=2.5, model_kwargs=MODEL_KWARGS
    )
    assert set(flat["rho"].values()) == {2.5}

    mapped = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, rho={"x0": 42.0}, rho_floor=0.1, model_kwargs=MODEL_KWARGS
    )
    assert mapped["rho"]["x0"] == 42.0
    assert mapped["rho"]["x1"] == 0.1  # falls back to rho_floor


def test_explicit_first_stage(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    manifest = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, first_stage=["Generator-p_nom"], model_kwargs=MODEL_KWARGS
    )
    assert manifest["first_stage"] == ["Generator-p_nom"]
    assert manifest["nonants"] == ["x0", "x1", "x2"]  # the three generators only


def test_clean_removes_phantom_scenarios(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    n.optimize.write_stochastic_problem_mpisppy(tmp_path, model_kwargs=MODEL_KWARGS)
    # A larger prior run would have left a higher-numbered scenario behind.
    (tmp_path / "scenario9.lp").write_text("\\* phantom *\\\nend\n")
    (tmp_path / "scenario9_nonants.json").write_text("{}")

    n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, clean=True, model_kwargs=MODEL_KWARGS
    )
    assert not (tmp_path / "scenario9.lp").exists()
    assert sorted(p.name for p in tmp_path.glob("*.lp")) == [
        "scenario0.lp",
        "scenario1.lp",
        "scenario2.lp",
    ]


def test_mps_format(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    manifest = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, file_format="mps", model_kwargs=MODEL_KWARGS
    )
    assert manifest["file_format"] == "mps"
    assert (tmp_path / "scenario0.mps").exists()


def test_manifest_solve_command(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    manifest = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, model_kwargs=MODEL_KWARGS
    )
    cmd = manifest["solve_command"]
    assert "mpisppy.generic_cylinders" in cmd
    assert f"--mps-files-directory {tmp_path}" in cmd
    assert "--lagrangian" in cmd
    assert "--xhatshuffle" in cmd
    assert "--write-xhat-file" in cmd
    # One rank per cylinder plus the PH hub.
    assert manifest["mpi_ranks"] == 3
    assert "sbatch" in manifest["sbatch_template"]


def test_manifest_solve_command_rho_setter(stochastic_export_network, tmp_path):
    """A rho setter passed via ``mpisppy_args`` is baked into the recorded command."""
    n = stochastic_export_network
    manifest = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, mpisppy_args=["--coeff-rho"], model_kwargs=MODEL_KWARGS
    )
    cmd = manifest["solve_command"]
    assert "--coeff-rho" in cmd
    # mpisppy_args are appended verbatim after the built-in flags.
    assert cmd.rstrip().endswith("--coeff-rho")


def test_sbatch_template_structure(stochastic_export_network, tmp_path):
    """The decoupled SLURM template wires the three phases together correctly."""
    n = stochastic_export_network
    manifest = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, model_kwargs=MODEL_KWARGS
    )
    tpl = manifest["sbatch_template"]

    # DIR points at the transfer directory (shared across all three jobs).
    assert f"DIR={tmp_path}" in tpl
    # Directory hygiene: one rm covering all four transfer-file globs (§13.6).
    assert (
        'rm -f "$DIR"/*.lp "$DIR"/*.mps "$DIR"/*_nonants.json "$DIR"/*_rho.csv' in tpl
    )
    # Three jobs: write -> solve -> read, chained with afterok dependencies.
    assert "write.sbatch" in tpl
    assert "solve.sbatch" in tpl
    assert "read.sbatch" in tpl
    assert "--dependency=afterok:$j1" in tpl
    assert "--dependency=afterok:$j2" in tpl
    # The solve job's rank annotation matches the manifest's rank count.
    assert f"-n {manifest['mpi_ranks']}," in tpl
    # The solve step runs the mpi-sppy driver via srun over the shared DIR.
    assert "srun" in tpl
    assert "mpisppy.generic_cylinders --mps-files-directory $DIR" in tpl


def test_requires_scenarios(tmp_path):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "b")
    n.add("Generator", "g", bus="b", p_nom_extendable=True)
    with pytest.raises(ValueError, match="requires a stochastic network"):
        n.optimize.write_stochastic_problem_mpisppy(tmp_path)


def test_invalid_file_format(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    with pytest.raises(ValueError, match="Invalid file_format"):
        n.optimize.write_stochastic_problem_mpisppy(tmp_path, file_format="nc")


def test_read_solution_sets_nom_opt(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    manifest = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, model_kwargs=MODEL_KWARGS
    )

    # Synthesize the incumbent xhat mpi-sppy would write.
    values = {"x0": 0.0, "x1": 120.0, "x2": 50.0, "x3": 5.0}
    lines = ["# mpi-sppy xhat", "# node_name, variable_name, value"]
    lines += [f"ROOT, {name}, {val}" for name, val in values.items()]
    Path(manifest["solution_file"]).write_text("\n".join(lines) + "\n")

    applied = n.optimize.read_stochastic_solution_mpisppy(tmp_path)
    assert applied == values

    # First-stage capacities are shared, so every scenario row gets the value.
    for scenario in n.scenarios:
        assert n.generators.loc[(scenario, "gas"), "p_nom_opt"] == 120.0
        assert n.generators.loc[(scenario, "boiler"), "p_nom_opt"] == 50.0
        assert n.links.loc[(scenario, "p2h"), "p_nom_opt"] == 5.0


def test_read_solution_missing_files(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    # No manifest yet.
    with pytest.raises(FileNotFoundError, match="manifest"):
        n.optimize.read_stochastic_solution_mpisppy(tmp_path)

    # Manifest present, but the solve has not produced a solution file.
    n.optimize.write_stochastic_problem_mpisppy(tmp_path, model_kwargs=MODEL_KWARGS)
    with pytest.raises(FileNotFoundError, match="Solution file"):
        n.optimize.read_stochastic_solution_mpisppy(tmp_path)


# --- dispatch re-solve (dispatch="resolve") -----------------------------------

# The re-solve needs an LP solver but not mpi-sppy; pick whatever is installed.
_SOLVER = next((s for s in ("gurobi", "highs") if s in available_solvers), None)


def _carry_optimal_capacities(n, n_ef):
    """Copy a native solve's ``*_nom_opt`` onto ``n`` (as the xhat read-back would)."""
    for c, attr in nominal_attrs.items():
        static = n.components[c].static
        if not static.empty:
            static[f"{attr}_opt"] = n_ef.components[c].static[f"{attr}_opt"]


def test_resolve_dispatch_unknown_scenario(stochastic_export_network):
    # Scenario names are validated before any solve, so no solver is needed.
    with pytest.raises(ValueError, match="Unknown scenario"):
        stochastic._resolve_dispatch(stochastic_export_network, scenarios=["nope"])


def test_read_solution_dispatch_modes(stochastic_export_network, tmp_path):
    n = stochastic_export_network
    manifest = n.optimize.write_stochastic_problem_mpisppy(
        tmp_path, model_kwargs=MODEL_KWARGS
    )
    values = {"x0": 0.0, "x1": 120.0, "x2": 50.0, "x3": 5.0}
    lines = ["# xhat"] + [f"ROOT, {k}, {v}" for k, v in values.items()]
    Path(manifest["solution_file"]).write_text("\n".join(lines) + "\n")

    # "read" is documented but not yet implemented; an unknown mode is an error.
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        n.optimize.read_stochastic_solution_mpisppy(tmp_path, dispatch="read")
    with pytest.raises(ValueError, match="Unknown dispatch mode"):
        n.optimize.read_stochastic_solution_mpisppy(tmp_path, dispatch="bogus")


@pytest.mark.skipif(_SOLVER is None, reason="needs an LP solver")
def test_resolve_dispatch_matches_native_ef(stochastic_export_network):
    n = stochastic_export_network
    n_ef = n.copy()
    n_ef.optimize(solver_name=_SOLVER, include_objective_constant=False)
    _carry_optimal_capacities(n, n_ef)

    stochastic._resolve_dispatch(n, solver_name=_SOLVER)

    # Primal dispatch reproduces the monolithic EF exactly (capacities fixed).
    ef_p = n_ef.generators_t.p
    rs_p = n.generators_t.p.reindex(columns=ef_p.columns)
    assert np.nanmax(np.abs(rs_p.to_numpy() - ef_p.to_numpy())) < 1e-6
    # Scenario-conditional duals are populated for every scenario.
    price = n.buses_t.marginal_price
    assert not price.empty
    assert set(price.columns.get_level_values("scenario")) == set(n.scenarios)


@pytest.mark.skipif(_SOLVER is None, reason="needs an LP solver")
def test_resolve_dispatch_scenario_subset(stochastic_export_network):
    n = stochastic_export_network
    n_ef = n.copy()
    n_ef.optimize(solver_name=_SOLVER, include_objective_constant=False)
    _carry_optimal_capacities(n, n_ef)

    stochastic._resolve_dispatch(n, scenarios=["low"], solver_name=_SOLVER)

    # Only the requested scenario's dispatch and duals are populated.
    assert set(n.generators_t.p.columns.get_level_values("scenario")) == {"low"}
    assert set(n.buses_t.marginal_price.columns.get_level_values("scenario")) == {"low"}


# --- solve_stochastic_mpisppy: command building (dependency-free) ---------------------


def test_solve_command_ph(tmp_path):
    argv, nranks = stochastic._solve_command(tmp_path, method="ph")
    assert nranks == 3  # PH hub + lagrangian + xhatshuffle
    assert argv[0] == "mpiexec"
    assert argv[argv.index("-np") + 1] == "3"
    assert "mpisppy.generic_cylinders" in argv
    assert "--lagrangian" in argv
    assert "--xhatshuffle" in argv
    assert "--default-rho" in argv
    assert "--max-iterations" in argv
    # PH caps solver threads per rank by default (2); None removes the cap.
    assert argv[argv.index("--max-solver-threads") + 1] == "2"
    assert (
        "--max-solver-threads"
        not in stochastic._solve_command(
            tmp_path, method="ph", max_solver_threads=None
        )[0]
    )
    assert argv[argv.index("--write-xhat-file") + 1] == str(tmp_path / "xhat.csv")


def test_solve_command_ef(tmp_path):
    argv, nranks = stochastic._solve_command(
        tmp_path, method="ef", solver_name="gurobi"
    )
    assert nranks == 1  # extensive form is monolithic, no MPI
    assert "mpiexec" not in argv
    assert "--EF" in argv
    assert argv[argv.index("--EF-solver-name") + 1] == "gurobi"
    # The default cap is PH-only: our max_solver_threads param never adds a cap
    # for the single-process EF...
    assert (
        "--max-solver-threads"
        not in stochastic._solve_command(tmp_path, method="ef", max_solver_threads=2)[0]
    )
    # ...but a user who explicitly asks to cap EF threads must be let through.
    ef_user_capped = stochastic._solve_command(
        tmp_path, method="ef", mpisppy_args=["--max-solver-threads", "1"]
    )[0]
    assert ef_user_capped[ef_user_capped.index("--max-solver-threads") + 1] == "1"
    assert argv[argv.index("--write-xhat-file") + 1] == str(tmp_path / "xhat.csv")


def test_solve_command_nprocs_options_args(tmp_path):
    argv, nranks = stochastic._solve_command(
        tmp_path,
        method="ph",
        nprocs=12,
        mpisppy_options={"rel_gap": 0.01, "intra_hub_conv_thresh": True, "off": False},
        mpisppy_args=["--max-stalled-iters", "5"],
        config_file="cfg.txt",
    )
    assert nranks == 12  # nprocs overrides the cylinder count
    assert argv[argv.index("-np") + 1] == "12"
    assert argv[argv.index("--rel-gap") + 1] == "0.01"  # underscore -> dash
    assert "--intra-hub-conv-thresh" in argv  # True -> bare flag
    assert "--off" not in argv  # False -> skipped
    assert argv[argv.index("--config-file") + 1] == "cfg.txt"
    assert argv[-2:] == ["--max-stalled-iters", "5"]  # extra args appended last


def test_solve_command_invalid_method(tmp_path):
    with pytest.raises(ValueError, match="Unknown method"):
        stochastic._solve_command(tmp_path, method="bogus")


def test_options_to_args():
    assert stochastic._options_to_args(None) == []
    assert stochastic._options_to_args({}) == []
    assert stochastic._options_to_args({"flag": True}) == ["--flag"]
    assert stochastic._options_to_args({"skip": False, "none": None}) == []
    assert stochastic._options_to_args({"max_iterations": 100}) == [
        "--max-iterations",
        "100",
    ]


# --- solve_stochastic_mpisppy: orchestration (solver subprocess mocked) ---------------


def _fake_xhat_writer(values):
    """Return a fake ``_run_solver`` that writes ``values`` as an xhat CSV."""

    def fake_run(argv, command, tee):
        xhat = Path(argv[argv.index("--write-xhat-file") + 1])
        lines = ["# fake mpi-sppy xhat"]
        lines += [f"ROOT, {name}, {val}" for name, val in values.items()]
        xhat.write_text("\n".join(lines) + "\n")

    return fake_run


def test_solve_stochastic_orchestration(
    stochastic_export_network, tmp_path, monkeypatch
):
    n = stochastic_export_network
    # Bypass the optional-dependency gate and the real solver subprocess.
    monkeypatch.setattr("pypsa.common.check_optional_dependency", lambda *a, **k: None)
    values = {"x0": 0.0, "x1": 120.0, "x2": 50.0, "x3": 5.0}
    monkeypatch.setattr(stochastic, "_run_solver", _fake_xhat_writer(values))

    applied = n.optimize.solve_stochastic_mpisppy(
        tmp_path, method="ef", model_kwargs=MODEL_KWARGS
    )
    assert applied == values
    # First-stage capacities are shared, so every scenario row gets the value.
    for scenario in n.scenarios:
        assert n.generators.loc[(scenario, "gas"), "p_nom_opt"] == 120.0
        assert n.generators.loc[(scenario, "boiler"), "p_nom_opt"] == 50.0
        assert n.links.loc[(scenario, "p2h"), "p_nom_opt"] == 5.0
    # An explicit working_dir is never auto-deleted.
    assert (tmp_path / "pypsa_stochastic_manifest.json").exists()


def test_solve_stochastic_tempdir_cleanup(
    stochastic_export_network, tmp_path, monkeypatch
):
    n = stochastic_export_network
    monkeypatch.setattr("pypsa.common.check_optional_dependency", lambda *a, **k: None)
    monkeypatch.setattr(stochastic, "_run_solver", _fake_xhat_writer({"x0": 1.0}))

    target = tmp_path / "work"

    def fake_mkdtemp(*args, **kwargs):
        target.mkdir(exist_ok=True)
        return str(target)

    monkeypatch.setattr(stochastic.tempfile, "mkdtemp", fake_mkdtemp)

    # working_dir=None + keep_files=False -> the temporary directory is removed.
    n.optimize.solve_stochastic_mpisppy(method="ef", model_kwargs=MODEL_KWARGS)
    assert not target.exists()

    # keep_files=True -> the temporary directory survives.
    n.optimize.solve_stochastic_mpisppy(
        method="ef", keep_files=True, model_kwargs=MODEL_KWARGS
    )
    assert target.exists()


def test_solve_stochastic_requires_mpiexec(
    stochastic_export_network, tmp_path, monkeypatch
):
    n = stochastic_export_network
    monkeypatch.setattr("pypsa.common.check_optional_dependency", lambda *a, **k: None)
    monkeypatch.setattr(stochastic.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="mpiexec"):
        n.optimize.solve_stochastic_mpisppy(
            tmp_path, method="ph", model_kwargs=MODEL_KWARGS
        )


def test_solve_stochastic_gates_on_mpisppy(
    stochastic_export_network, tmp_path, monkeypatch
):
    def boom(module_name, install_message):
        raise ImportError(install_message)

    monkeypatch.setattr("pypsa.common.check_optional_dependency", boom)
    n = stochastic_export_network
    with pytest.raises(ImportError, match="Missing optional dependency"):
        n.optimize.solve_stochastic_mpisppy(
            tmp_path, method="ef", model_kwargs=MODEL_KWARGS
        )


def test_solve_stochastic_invalid_method(
    stochastic_export_network, tmp_path, monkeypatch
):
    monkeypatch.setattr("pypsa.common.check_optional_dependency", lambda *a, **k: None)
    n = stochastic_export_network
    with pytest.raises(ValueError, match="Unknown method"):
        n.optimize.solve_stochastic_mpisppy(
            tmp_path, method="bogus", model_kwargs=MODEL_KWARGS
        )


# --- solve_stochastic_mpisppy: end-to-end through real mpi-sppy -----------------------


@pytest.mark.skipif(
    not (_HAS_MPISPPY and _HAS_GUROBI), reason="needs mpi-sppy and Gurobi"
)
def test_solve_stochastic_ef_matches_native_ef(stochastic_export_network, tmp_path):
    """The mpi-sppy EF first stage must match PyPSA's native EF first stage."""
    n = stochastic_export_network
    m = n.copy()

    # Native monolithic EF (the correctness oracle). `optimize` takes
    # `include_objective_constant` directly; `consistency_check` is a
    # `create_model` kwarg, so it is not forwarded here.
    n.optimize(solver_name="gurobi", include_objective_constant=False)
    native = {
        ("Generator", asset): n.generators.loc[(n.scenarios[0], asset), "p_nom_opt"]
        for asset in ("wind", "gas", "boiler")
    }
    native[("Link", "p2h")] = n.links.loc[(n.scenarios[0], "p2h"), "p_nom_opt"]

    # Decomposition path: write -> mpi-sppy EF subprocess -> read back.
    applied = m.optimize.solve_stochastic_mpisppy(
        tmp_path,
        method="ef",
        solver_name="gurobi",
        mpisppy_args=["--max-solver-threads", "2"],
        tee=False,
        model_kwargs=MODEL_KWARGS,
    )
    assert set(applied) == {"x0", "x1", "x2", "x3"}
    for asset in ("wind", "gas", "boiler"):
        assert m.generators.loc[(m.scenarios[0], asset), "p_nom_opt"] == pytest.approx(
            native[("Generator", asset)], abs=1e-3
        )
    assert m.links.loc[(m.scenarios[0], "p2h"), "p_nom_opt"] == pytest.approx(
        native[("Link", "p2h")], abs=1e-3
    )


@pytest.mark.skipif(
    not (_HAS_MPISPPY and _HAS_GUROBI and _HAS_MPIEXEC),
    reason="needs mpi-sppy, Gurobi and mpiexec",
)
def test_decoupled_workflow_runs_recorded_command(stochastic_export_network, tmp_path):
    """The decoupled write -> (recorded solve_command) -> read round-trip (§13.6).

    Unlike the inline ``solve_stochastic_mpisppy`` test, this executes the *exact* PH
    ``solve_command`` the manifest hands a SLURM user (rather than a command the
    driver rebuilds internally) and reads back a genuine mpi-sppy ``xhat.csv`` --
    so it exercises the recorded command string and the real solution file.
    """
    n = stochastic_export_network
    m = n.copy()

    # Native monolithic EF (the oracle the PH first stage must reproduce).
    n.optimize(solver_name="gurobi", include_objective_constant=False)
    s0 = n.scenarios[0]
    native = {
        a: n.generators.loc[(s0, a), "p_nom_opt"] for a in ("wind", "gas", "boiler")
    }
    native["p2h"] = n.links.loc[(s0, "p2h"), "p_nom_opt"]

    # Step 1 (write.sbatch): PyPSA writes the per-scenario files + manifest.
    manifest = m.optimize.write_stochastic_problem_mpisppy(
        tmp_path,
        solver_name="gurobi",
        max_iterations=200,
        mpisppy_args=["--max-solver-threads", "2"],
        model_kwargs=MODEL_KWARGS,
    )

    # Step 2 (solve.sbatch): run the recorded PH command verbatim.
    completed = subprocess.run(
        shlex.split(manifest["solve_command"]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert Path(manifest["solution_file"]).exists()

    # Step 3 (read.sbatch): PyPSA reads the incumbent first stage back.
    applied = m.optimize.read_stochastic_solution_mpisppy(tmp_path)
    assert set(applied) == {"x0", "x1", "x2", "x3"}
    for asset in ("wind", "gas", "boiler"):
        assert m.generators.loc[(s0, asset), "p_nom_opt"] == pytest.approx(
            native[asset], abs=1e-2
        )
    assert m.links.loc[(s0, "p2h"), "p_nom_opt"] == pytest.approx(
        native["p2h"], abs=1e-2
    )


@pytest.mark.skipif(
    not (_HAS_MPISPPY and _HAS_GUROBI and _HAS_MPIEXEC),
    reason="needs mpi-sppy, Gurobi and mpiexec",
)
def test_solve_stochastic_ph_coeff_rho_matches_native_ef(
    stochastic_export_network, tmp_path
):
    """PH with the ``--coeff-rho`` rho setter must still reproduce the native EF.

    Exercises the documented decoupled-workflow rho setter end to end: the
    coefficient-based rho is passed through ``mpisppy_args`` (so it lands in the
    driver command), and the PH first stage is checked against the monolithic EF.
    """
    n = stochastic_export_network
    m = n.copy()

    n.optimize(solver_name="gurobi", include_objective_constant=False)
    s0 = n.scenarios[0]
    native = {
        a: n.generators.loc[(s0, a), "p_nom_opt"] for a in ("wind", "gas", "boiler")
    }
    native["p2h"] = n.links.loc[(s0, "p2h"), "p_nom_opt"]

    m.optimize.solve_stochastic_mpisppy(
        tmp_path,
        method="ph",
        solver_name="gurobi",
        max_iterations=200,
        mpisppy_args=["--coeff-rho", "--max-solver-threads", "2"],
        tee=False,
        model_kwargs=MODEL_KWARGS,
    )
    for asset in ("wind", "gas", "boiler"):
        assert m.generators.loc[(s0, asset), "p_nom_opt"] == pytest.approx(
            native[asset], abs=1e-2
        )
    assert m.links.loc[(s0, "p2h"), "p_nom_opt"] == pytest.approx(
        native["p2h"], abs=1e-2
    )
