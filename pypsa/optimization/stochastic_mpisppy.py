# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Export PyPSA stochastic problems for decomposition with mpi-sppy.

A PyPSA stochastic network (built with :meth:`pypsa.Network.set_scenarios`) can
be solved by *decomposition* -- Progressive Hedging with bounding cylinders --
using `mpi-sppy <https://github.com/Pyomo/mpi-sppy>`__ as an alternative to
PyPSA's native monolithic Extensive Form solve. The two programs meet only at a
**file boundary**: PyPSA writes one optimisation file per scenario plus small
metadata files, and mpi-sppy's file-based scenario loader
(``mpisppy.problem_io.mps_module``) reads them.

This module therefore imports neither mpi-sppy nor Pyomo. It contains the two
dependency-free phases of the workflow:

- :func:`write_stochastic_problem_mpisppy` -- slice each scenario into a standalone
  single-scenario network, build its model, and write ``{s}.lp`` (or ``.mps``),
  ``{s}_nonants.json`` (first-stage variable names + probability) and
  ``{s}_rho.csv`` (per-nonant Progressive-Hedging rho);
- :func:`read_stochastic_solution_mpisppy` -- read mpi-sppy's incumbent first-stage
  solution back onto the network as ``*_nom_opt``.

The intervening solve (``mpisppy.generic_cylinders`` under MPI) is run
separately, either as a blocking subprocess (inline) or as its own scheduler
job (decoupled / SLURM). See the design doc in the mpi-sppy repository
(``doc/designs/pypsa_stochastic_design.md``) for the full rationale.
"""

from __future__ import annotations

import itertools
import json
import logging
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.descriptors import nominal_attrs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from linopy import Model

    from pypsa import Network

logger = logging.getLogger(__name__)


# File names that ``clean=True`` removes and that the exporter (re)writes. The
# scenario-model glob (``*.lp`` / ``*.mps``) is how mpi-sppy discovers
# scenarios, so stale files would otherwise become phantom scenarios.
_SCENARIO_SUFFIXES = ("lp", "mps")
_METADATA_SUFFIXES = ("_nonants.json", "_rho.csv")
_MANIFEST_NAME = "pypsa_stochastic_manifest.json"
_SOLUTION_NAME = "xhat.csv"


def _default_first_stage(model: Model) -> list[str]:
    """Return the default first-stage variable names present in ``model``.

    The default first stage is every extendable capacity variable
    (``Generator-p_nom``, ``Line-s_nom``, ``Link-p_nom``, ``Store-e_nom``,
    ``StorageUnit-p_nom``, ``Transformer-s_nom``, ...) plus modular module
    counts (``*-n_mod``), restricted to those actually defined in the model.
    """
    candidates = []
    for component, attr in nominal_attrs.items():
        candidates.append(f"{component}-{attr}")
        candidates.append(f"{component}-n_mod")
    return [name for name in candidates if name in model.variables]


def _nonant_records(model: Model, first_stage: Sequence[str]) -> list[dict[str, Any]]:
    """Map first-stage linopy variables to their on-file nonant names.

    Returns one record per scalar nonant, in deterministic order (the order of
    ``first_stage``, then each variable's coordinate order), with keys
    ``name`` (the on-file ``x{label}`` name), ``component``, ``attr`` and
    ``asset`` (the component element name).

    linopy assigns variable *labels* by creation order, so structurally
    identical scenario networks built with different data yield identical
    ``x{label}`` names -- which is what lets mpi-sppy match nonants by name.
    """
    records: list[dict[str, Any]] = []
    for varname in first_stage:
        component, attr = varname.split("-", 1)
        labels = model.variables[varname].labels
        # C-order product of the coordinate values lines up with ``ravel``.
        coord_lists = [labels.coords[d].values for d in labels.dims]
        flat_labels = labels.values.ravel()
        for combo, label in zip(
            itertools.product(*coord_lists), flat_labels, strict=True
        ):
            label = int(label)
            if label < 0:  # masked / inactive element
                continue
            asset = (
                combo[labels.dims.index("name")] if "name" in labels.dims else combo[-1]
            )
            records.append(
                {
                    "name": f"x{label}",
                    "label": label,
                    "component": component,
                    "attr": attr,
                    "asset": str(asset),
                }
            )
    return records


def _objective_coefficients(model: Model) -> dict[int, float]:
    """Return ``{variable label: objective coefficient}`` for ``model``.

    A label appearing in several objective terms has its coefficients summed.
    """
    flat = model.objective.flat
    coeffs = pd.Series(
        flat["coeffs"].astype(float).to_numpy(),
        index=flat["vars"].astype(int).to_numpy(),
    )
    return coeffs.groupby(level=0).sum().to_dict()


def _compute_rho(
    model: Model,
    records: Sequence[dict[str, Any]],
    rho: str | float | dict[str, float],
    rho_alpha: float,
    rho_floor: float,
) -> dict[str, float]:
    """Compute the per-nonant Progressive-Hedging rho.

    ``rho`` selects the policy:

    - ``"cost-proportional"`` (default): ``rho_i = max(rho_floor, rho_alpha *
      |c_i|)`` where ``c_i`` is the objective coefficient of nonant ``i`` (the
      ``capital_cost`` on the ``*_nom`` variable);
    - a number: that flat value for every nonant;
    - a mapping ``{on-file name: rho}``: explicit per-nonant values, with
      ``rho_floor`` for any nonant the mapping omits.

    rho is **scenario-invariant** (capital costs are first-stage), so this is
    computed once and replicated to every scenario's ``_rho.csv`` -- the
    consistency mpi-sppy requires of the writer. The coefficients come from the
    single ``model`` passed in (scenario 0 at the call site); a network with
    genuinely per-scenario ``capital_cost`` would silently use only that
    scenario's values.
    """
    if isinstance(rho, str):
        if rho != "cost-proportional":
            msg = f"Unknown rho policy {rho!r}; expected 'cost-proportional', a number or a mapping."
            raise ValueError(msg)
        coeffs = _objective_coefficients(model)
        return {
            r["name"]: max(rho_floor, rho_alpha * abs(coeffs.get(r["label"], 0.0)))
            for r in records
        }
    if isinstance(rho, dict):
        return {r["name"]: float(rho.get(r["name"], rho_floor)) for r in records}
    if isinstance(rho, (int, float)) and not isinstance(rho, bool):
        return {r["name"]: float(rho) for r in records}
    msg = (
        f"Invalid `rho` of type {type(rho).__name__}; expected str, number or mapping."
    )
    raise TypeError(msg)


def _write_nonants_json(
    path: Path, scenario: str, probability: float, nonants: Sequence[str]
) -> None:
    """Write the ``{s}_nonants.json`` consumed by ``mps_module.scenario_creator``."""
    payload = {
        "scenarioData": {"name": scenario, "scenProb": float(probability)},
        "treeData": {
            "globalNodeCount": 1,
            "nodes": {
                "ROOT": {
                    "serialNumber": 0,
                    "condProb": 1.0,
                    "nonAnts": list(nonants),
                }
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_rho_csv(path: Path, rho: dict[str, float], order: Sequence[str]) -> None:
    """Write the ``{s}_rho.csv`` (header ``varname,rho``, one row per nonant)."""
    lines = ["varname,rho"]
    lines += [f"{name},{rho[name]}" for name in order]
    path.write_text("\n".join(lines) + "\n")


def _clean_directory(directory: Path) -> None:
    """Remove stale scenario + metadata files so discovery sees only this run."""
    stale: list[Path] = []
    for suffix in _SCENARIO_SUFFIXES:
        stale += directory.glob(f"*.{suffix}")
    for suffix in _METADATA_SUFFIXES:
        stale += directory.glob(f"*{suffix}")
    stale.append(directory / _MANIFEST_NAME)
    # Also drop a stale solution file so an aborted prior solve is not read as fresh.
    stale.append(directory / _SOLUTION_NAME)
    for path in stale:
        if path.exists():
            path.unlink()


def _options_to_args(options: dict[str, Any] | None) -> list[str]:
    """Translate an ``mpisppy_options`` dict into mpi-sppy CLI arguments.

    ``True`` becomes a bare ``--flag``; ``False`` / ``None`` is skipped; any
    other value becomes ``--flag value``. Underscores in keys are converted to
    dashes to match mpi-sppy's CLI (``{"max_iterations": 100}`` ->
    ``--max-iterations 100``).
    """
    if not options:
        return []
    args: list[str] = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        else:
            args += [flag, str(value)]
    return args


def _solve_command(
    directory: Path,
    *,
    method: str = "ph",
    cylinders: Sequence[str] = ("lagrangian", "xhatshuffle"),
    solver_name: str = "gurobi",
    default_rho: float = 1.0,
    max_iterations: int = 50,
    config_file: str | None = None,
    mpisppy_options: dict[str, Any] | None = None,
    mpisppy_args: Sequence[str] | None = None,
    nprocs: int | None = None,
    max_solver_threads: int | None = 2,
) -> tuple[list[str], int]:
    """Build the mpi-sppy driver command (an argv list) and its MPI rank count.

    ``method="ph"`` runs ``generic_cylinders`` under ``mpiexec`` with one rank
    per cylinder plus the Progressive-Hedging hub (e.g. ``("lagrangian",
    "xhatshuffle")`` -> 3 ranks), overridable with ``nprocs``. ``method="ef"``
    solves the extensive form directly in a single process -- the correctness
    oracle, no MPI. Both write the incumbent first stage with
    ``--write-xhat-file`` so :func:`read_stochastic_solution_mpisppy` can read it back.

    For PH, the per-rank subproblem solver is capped at ``max_solver_threads``
    threads (default 2) so that ranks sharing a node do not oversubscribe cores;
    pass ``None`` to remove the cap. If the caller supplies their own
    ``--max-solver-threads`` through ``mpisppy_options``/``mpisppy_args``, that
    replaces the default (the flag is never emitted twice). ``max_solver_threads``
    applies to PH only -- the single-process EF is not capped by this default,
    but a user can still limit EF threads via ``mpisppy_args`` / ``mpisppy_options``.
    """
    solution_file = directory / _SOLUTION_NAME
    # Assemble the user-supplied tail (config file + options dict + raw args) up
    # front so the PH thread-cap default can defer to an explicit
    # --max-solver-threads the user passed, rather than emitting it twice.
    tail: list[str] = []
    if config_file is not None:
        tail += ["--config-file", str(config_file)]
    tail += _options_to_args(mpisppy_options)
    if mpisppy_args:
        tail += list(mpisppy_args)
    user_capped_threads = any(
        t == "--max-solver-threads" or t.startswith("--max-solver-threads=")
        for t in tail
    )
    if method == "ef":
        nranks = 1
        argv = [
            "python",
            "-m",
            "mpisppy.generic_cylinders",
            "--mps-files-directory",
            str(directory),
            "--EF",
            "--EF-solver-name",
            solver_name,
            "--write-xhat-file",
            str(solution_file),
        ]
    elif method == "ph":
        nranks = nprocs if nprocs is not None else 1 + len(cylinders)
        argv = [
            "mpiexec",
            "-np",
            str(nranks),
            "python",
            "-m",
            "mpi4py",
            "-m",
            "mpisppy.generic_cylinders",
            "--mps-files-directory",
            str(directory),
            "--solver-name",
            solver_name,
        ]
        for cylinder in cylinders:
            argv.append(f"--{cylinder}")
        argv += [
            "--default-rho",
            str(default_rho),
            "--max-iterations",
            str(max_iterations),
        ]
        # Cap each rank's subproblem solver so co-located ranks do not
        # oversubscribe cores. This default is PH-only; EF (above) is left
        # uncapped. Skip it when the user already set --max-solver-threads (via
        # mpisppy_options/mpisppy_args) so the flag is never emitted twice.
        if max_solver_threads is not None and not user_capped_threads:
            argv += ["--max-solver-threads", str(max_solver_threads)]
        argv += ["--write-xhat-file", str(solution_file)]
    else:
        msg = f"Unknown method {method!r}; expected 'ph' or 'ef'."
        raise ValueError(msg)
    argv += tail
    return argv, nranks


def _sbatch_template(directory: Path, nranks: int) -> str:
    """Return a copy-paste SLURM dependency chain (write -> solve -> read)."""
    return (
        "# Decoupled SLURM workflow (edit envs/partitions/accounts to taste).\n"
        f"DIR={directory}\n"
        f'rm -f "$DIR"/*.lp "$DIR"/*.mps "$DIR"/*_nonants.json "$DIR"/*_rho.csv "$DIR"/{_SOLUTION_NAME}  # hygiene\n'
        "j1=$(sbatch --parsable write.sbatch)                           # -n 1,   PyPSA env\n"
        f"j2=$(sbatch --parsable --dependency=afterok:$j1 solve.sbatch)  # -n {nranks}, mpi-sppy env\n"
        "sbatch          --dependency=afterok:$j2 read.sbatch          # -n 1,   PyPSA env\n"
        "# solve.sbatch runs, e.g.:\n"
        "#   srun python -m mpi4py -m mpisppy.generic_cylinders --mps-files-directory $DIR ...\n"
    )


def write_stochastic_problem_mpisppy(
    n: Network,
    directory: str | Path,
    *,
    clean: bool = True,
    first_stage: Sequence[str] | None = None,
    rho: str | float | dict[str, float] = "cost-proportional",
    rho_alpha: float = 1.0,
    rho_floor: float = 1e-3,
    file_format: str = "lp",
    cylinders: Sequence[str] = ("lagrangian", "xhatshuffle"),
    solver_name: str = "gurobi",
    default_rho: float = 1.0,
    max_iterations: int = 50,
    config_file: str | None = None,
    mpisppy_args: Sequence[str] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a stochastic network as per-scenario files for mpi-sppy.

    For each scenario the network is sliced into a standalone single-scenario
    network (:meth:`pypsa.Network.get_scenario`), a linopy model is built, and
    three files are written into ``directory``:

    - ``{s}.lp`` (or ``{s}.mps``) -- the scenario subproblem, with implicit
      ``x{label}`` variable names;
    - ``{s}_nonants.json`` -- the scenario probability and the ordered list of
      first-stage (nonanticipative) variable names;
    - ``{s}_rho.csv`` -- the per-nonant Progressive-Hedging rho.

    A ``pypsa_stochastic_manifest.json`` is also written (and returned),
    carrying the exact phase-2 mpi-sppy command, an ``sbatch`` template and the
    nonant-to-component map that :func:`read_stochastic_solution_mpisppy` uses. This
    function is dependency-free -- it imports neither mpi-sppy nor Pyomo.

    Parameters
    ----------
    n : pypsa.Network
        A stochastic network (``n.has_scenarios`` must be ``True``).
    directory : str or pathlib.Path
        Output directory; created if missing.
    clean : bool, default True
        Remove stale ``*.lp`` / ``*.mps`` / ``*_nonants.json`` / ``*_rho.csv``
        (and the manifest) from ``directory`` first. mpi-sppy discovers
        scenarios by globbing the directory, so leftovers from a larger prior
        run would otherwise be picked up as phantom scenarios.
    first_stage : sequence of str, optional
        Linopy variable names to treat as first-stage nonants. ``None`` (the
        default) uses every extendable capacity variable plus ``*-n_mod``.
    rho : {"cost-proportional"}, number or mapping, default "cost-proportional"
        Per-nonant Progressive-Hedging rho policy. See :func:`_compute_rho`.
    rho_alpha : float, default 1.0
        Scale factor for cost-proportional rho.
    rho_floor : float, default 1e-3
        Lower bound on rho so zero-cost nonants still get a positive value.
    file_format : {"lp", "mps"}, default "lp"
        Scenario file format. LP is preferred (linopy controls the on-file
        names directly and the files are human-readable); MPS also works.
    cylinders : sequence of str, default ("lagrangian", "xhatshuffle")
        mpi-sppy bounding spokes; one MPI rank per cylinder plus the PH hub.
        Only used to assemble the manifest command.
    solver_name : str, default "gurobi"
        Subproblem solver recorded in the manifest command.
    default_rho : float, default 1.0
        mpi-sppy ``--default-rho`` fallback recorded in the manifest command.
    max_iterations : int, default 50
        Progressive-Hedging ``--max-iterations`` recorded in the manifest command.
    config_file : str, optional
        mpi-sppy ``--config-file`` to reference in the manifest command.
    mpisppy_args : sequence of str, optional
        Extra mpi-sppy CLI arguments appended to the manifest command.
    model_kwargs : dict, optional
        Keyword arguments forwarded to ``create_model`` for each scenario.

    Returns
    -------
    dict
        The manifest (also written to ``directory/pypsa_stochastic_manifest.json``).

    Raises
    ------
    ValueError
        If ``n`` has no scenarios, ``file_format`` is invalid, or the
        first-stage nonant name-lists disagree across scenarios.

    """
    if not n.has_scenarios:
        msg = (
            "write_stochastic_problem_mpisppy requires a stochastic network. "
            "Call n.set_scenarios({...}) first."
        )
        raise ValueError(msg)
    if file_format not in _SCENARIO_SUFFIXES:
        msg = f"Invalid file_format {file_format!r}; expected one of {_SCENARIO_SUFFIXES}."
        raise ValueError(msg)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if clean:
        _clean_directory(directory)

    model_kwargs = dict(model_kwargs or {})
    weights = n.scenario_weightings["weight"]
    scenarios = list(n.scenarios)
    # mpi-sppy uses each scenario file's stem as the scenario name and requires
    # it to end in a (zero-based) integer; it also sorts stems lexicographically.
    # Use a zero-padded index stem so the order matches and discovery is happy;
    # the human-readable PyPSA name is preserved in the metadata.
    width = max(1, len(str(len(scenarios) - 1)))
    stems = [f"scenario{i:0{width}d}" for i in range(len(scenarios))]

    reference_nonants: list[str] | None = None
    rho_values: dict[str, float] = {}
    nonant_map: dict[str, dict[str, str]] = {}
    first_stage_resolved: list[str] = []
    scenario_files: list[dict[str, Any]] = []

    for i, (scenario, stem) in enumerate(zip(scenarios, stems, strict=True)):
        ns = n.get_scenario(scenario)
        model = ns.optimize.create_model(**model_kwargs)

        if first_stage is None:
            scen_first_stage = _default_first_stage(model)
        else:
            missing = [v for v in first_stage if v not in model.variables]
            if missing:
                msg = f"first_stage variables not found in model for scenario {scenario!r}: {missing}"
                raise ValueError(msg)
            scen_first_stage = list(first_stage)

        records = _nonant_records(model, scen_first_stage)
        names = [r["name"] for r in records]

        # The necessary condition: identical nonant names AND order everywhere.
        if reference_nonants is None:
            reference_nonants = names
            first_stage_resolved = scen_first_stage
            rho_values = _compute_rho(model, records, rho, rho_alpha, rho_floor)
            nonant_map = {
                r["name"]: {
                    "component": r["component"],
                    "attr": r["attr"],
                    "asset": r["asset"],
                }
                for r in records
            }
        elif names != reference_nonants:
            msg = (
                f"First-stage nonant names disagree between scenario "
                f"{scenarios[0]!r} and {scenario!r}. mpi-sppy matches nonants "
                "positionally, so every scenario must expose the same first-stage "
                "variables in the same order. Ensure all scenarios share component "
                "structure and snapshots (data-only differences)."
            )
            raise ValueError(msg)

        model.to_file(directory / f"{stem}.{file_format}", io_api=file_format)
        _write_nonants_json(
            directory / f"{stem}_nonants.json", scenario, weights[scenario], names
        )
        _write_rho_csv(directory / f"{stem}_rho.csv", rho_values, reference_nonants)
        scenario_files.append(
            {
                "index": i,
                "name": scenario,
                "stem": stem,
                "probability": float(weights[scenario]),
            }
        )
        logger.info(
            "Wrote scenario %r as %r (%d nonants) to %s",
            scenario,
            stem,
            len(names),
            directory,
        )

    solve_argv, nranks = _solve_command(
        directory,
        method="ph",
        cylinders=cylinders,
        solver_name=solver_name,
        default_rho=default_rho,
        max_iterations=max_iterations,
        config_file=config_file,
        mpisppy_args=mpisppy_args,
    )
    solve_command = shlex.join(solve_argv)
    manifest: dict[str, Any] = {
        "directory": str(directory),
        "file_format": file_format,
        "scenarios": {s: float(weights[s]) for s in scenarios},
        "scenario_files": scenario_files,
        "first_stage": first_stage_resolved,
        "nonants": reference_nonants,
        "nonant_map": nonant_map,
        "rho": rho_values,
        "solution_file": str(directory / _SOLUTION_NAME),
        "solve_command": solve_command,
        "mpi_ranks": nranks,
        "sbatch_template": _sbatch_template(directory, nranks),
    }
    (directory / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Wrote %d scenarios to %s. Phase-2 command:\n  %s",
        len(manifest["scenarios"]),
        directory,
        solve_command,
    )
    return manifest


def _read_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / _MANIFEST_NAME
    if not manifest_path.exists():
        msg = (
            f"No {_MANIFEST_NAME} in {directory}. read_stochastic_solution_mpisppy needs the "
            "manifest written by write_stochastic_problem_mpisppy to map nonants back to components."
        )
        raise FileNotFoundError(msg)
    return json.loads(manifest_path.read_text())


def _read_xhat_csv(path: Path) -> dict[str, float]:
    """Parse mpi-sppy's incumbent xhat CSV into ``{variable_name: value}``.

    The file (written by ``--write-xhat-file``) has ``#`` comment lines and
    data lines ``node_name, variable_name, value``. Only the ``ROOT`` node is
    used (two-stage). mpi-sppy normalises ``(`` / ``)`` to ``_`` in names; our
    implicit ``x{label}`` names contain neither, so they pass through verbatim.
    """
    values: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            msg = (
                f"Malformed line in xhat file {path} (expected "
                f"'node, variable, value'): {line!r}"
            )
            raise ValueError(msg)
        node, varname, value = parts
        if node != "ROOT":
            continue
        values[varname] = float(value)
    return values


def _validate_dispatch(
    n: Network, dispatch: str, scenarios: Sequence[str] | None
) -> None:
    """Validate a ``dispatch``/``scenarios`` request before any expensive work.

    Called up front by both entry points so a typo -- or the not-yet-implemented
    ``dispatch="read"`` -- fails immediately rather than after a full solve.
    """
    if dispatch == "read":
        msg = (
            "dispatch='read' (reading mpi-sppy's full primal tree solution) is "
            "planned but not yet implemented; use dispatch='resolve'."
        )
        raise NotImplementedError(msg)
    if dispatch not in ("none", "resolve"):
        msg = f"Unknown dispatch mode {dispatch!r}; expected 'none' or 'resolve'."
        raise ValueError(msg)
    if dispatch == "resolve" and scenarios is not None:
        unknown = [s for s in scenarios if s not in set(n.scenarios)]
        if unknown:
            msg = (
                f"Unknown scenario(s) {unknown}; network scenarios are "
                f"{list(n.scenarios)}."
            )
            raise ValueError(msg)


def read_stochastic_solution_mpisppy(
    n: Network,
    directory: str | Path,
    *,
    solution_file: str | Path | None = None,
    dispatch: str = "none",
    scenarios: Sequence[str] | None = None,
    solver_name: str = "gurobi",
    **solve_kwargs: Any,
) -> dict[str, float]:
    """Read mpi-sppy's incumbent first stage back onto the network.

    Reads the manifest written by :func:`write_stochastic_problem_mpisppy` and the
    incumbent xhat file written by mpi-sppy (``--write-xhat-file``), then sets
    the corresponding ``*_nom_opt`` capacity values on ``n`` (shared across all
    scenarios, since they are first-stage decisions). Modular module-count
    nonants (``*-n_mod``) are also written back (see Notes). The ``dispatch``
    mode optionally recovers the second stage too (see Notes). Reading the first
    stage is dependency-free; ``dispatch="resolve"`` needs a linopy-supported
    solver (but not mpi-sppy).

    Parameters
    ----------
    n : pypsa.Network
        The stochastic network the problem was written from.
    directory : str or pathlib.Path
        The transfer directory containing the manifest and the solution file.
    solution_file : str or pathlib.Path, optional
        Path to mpi-sppy's xhat CSV. Defaults to the manifest's
        ``solution_file`` (``directory/xhat.csv``).
    dispatch : {"none", "resolve"}, default "none"
        How to recover the second stage (the xhat carries only the first stage):

        - ``"none"`` -- first stage only.
        - ``"resolve"`` -- re-solve each scenario with the optimized capacities
          fixed, recovering the per-scenario operational time series **and
          scenario-conditional duals** (e.g. ``generators_t.p``,
          ``buses_t.marginal_price``) onto ``n``. The per-scenario problems are
          independent LPs but are solved serially.

        (``"read"`` -- parse mpi-sppy's full primal tree solution -- is planned
        but not yet implemented.)
    scenarios : sequence of str, optional
        Restrict the dispatch recovery to this subset of scenarios (default: all).
        Useful for ``dispatch="resolve"`` when only some scenarios' dispatch /
        scenario-conditional duals are of interest, since re-solving is serial.
    solver_name : str, default "gurobi"
        Solver for the dispatch re-solve (only used when ``dispatch="resolve"``).
    **solve_kwargs
        Extra keyword arguments forwarded to each scenario's ``n.optimize`` call
        during the dispatch re-solve.

    Returns
    -------
    dict
        ``{on-file nonant name: value}`` that was applied (the first stage).

    Notes
    -----
    For modular components (extendable with a non-zero ``*_nom_mod`` module
    size), the export carries both the capacity (``*_nom``) and the integer
    module count (``*-n_mod``) as first-stage nonants. The module count is
    written to an ``n_mod_opt`` column and the capacity is set to the clean
    integer-derived value ``*_nom_opt = n_mod_opt * *_nom_mod`` (the two are
    tied by an equality constraint in every subproblem, so this agrees with the
    capacity nonant to solver tolerance).

    The incumbent xhat carries only the *first stage*. With
    ``dispatch="resolve"`` the second stage is recovered by fixing the
    capacities and re-solving each scenario; the per-scenario operational
    results are merged onto ``n``'s ``(scenario, name)`` columns. The duals from
    these standalone re-solves are **conditional on the scenario** (the marginal
    values in that scenario's realised world) -- a distinct object from the
    extensive form's probability-weighted duals -- and are typically what a
    decision maker wants.

    """
    _validate_dispatch(n, dispatch, scenarios)
    directory = Path(directory)
    manifest = _read_manifest(directory)
    nonant_map: dict[str, dict[str, str]] = manifest["nonant_map"]

    xhat_path = (
        Path(solution_file)
        if solution_file is not None
        else Path(manifest["solution_file"])
    )
    if not xhat_path.exists():
        msg = (
            f"Solution file {xhat_path} not found. Run the phase-2 solve first:\n  "
            f"{manifest['solve_command']}"
        )
        raise FileNotFoundError(msg)
    values = _read_xhat_csv(xhat_path)

    applied: dict[str, float] = {}
    nmod_records: list[tuple[str, dict[str, str], float]] = []
    # First pass: plain capacity nonants (``*_nom`` -> ``*_nom_opt``). Module
    # counts are deferred to a second pass so they win over any capacity nonant
    # for the same asset (the integer-derived capacity is the clean value).
    for name, value in values.items():
        info = nonant_map.get(name)
        if info is None:
            logger.warning("xhat nonant %r not in manifest; skipping.", name)
            continue
        if info["attr"] == "n_mod":
            nmod_records.append((name, info, value))
            continue
        static = n.components[info["component"]].static
        # Capacities are shared first-stage decisions: write to every scenario.
        mask = static.index.get_level_values("name") == info["asset"]
        if not mask.any():
            logger.warning(
                "xhat nonant %r maps to %s %r, which is not in the network; skipping.",
                name,
                info["component"],
                info["asset"],
            )
            continue
        static.loc[mask, f"{info['attr']}_opt"] = value
        applied[name] = value

    # Second pass: module-count nonants (``*-n_mod`` -> ``n_mod_opt``), deriving
    # the integer capacity ``*_nom_opt = n_mod_opt * *_nom_mod``.
    for name, info, value in nmod_records:
        component = info["component"]
        nom_attr = nominal_attrs[component]
        static = n.components[component].static
        mask = static.index.get_level_values("name") == info["asset"]
        if not mask.any():
            logger.warning(
                "xhat nonant %r maps to %s %r, which is not in the network; skipping.",
                name,
                component,
                info["asset"],
            )
            continue
        n_modules = round(value)
        static.loc[mask, "n_mod_opt"] = n_modules
        static.loc[mask, f"{nom_attr}_opt"] = (
            n_modules * static.loc[mask, f"{nom_attr}_mod"]
        )
        applied[name] = float(n_modules)

    logger.info("Applied %d first-stage values from %s.", len(applied), xhat_path)
    if nmod_records:
        logger.info(
            "Wrote back %d modular module-count nonant(s) to n_mod_opt.",
            len(nmod_records),
        )

    if dispatch == "resolve":
        _resolve_dispatch(
            n, scenarios=scenarios, solver_name=solver_name, **solve_kwargs
        )

    return applied


def _resolve_dispatch(
    n: Network,
    *,
    scenarios: Sequence[str] | None = None,
    solver_name: str = "gurobi",
    **solve_kwargs: Any,
) -> None:
    """Recover the second stage by re-solving each scenario with capacities fixed.

    Requires the first-stage ``*_nom_opt`` capacities to be set already (the
    xhat write-back) and a linopy-supported solver. For each scenario in
    ``scenarios`` (default: all) the capacities are fixed
    (:meth:`~pypsa.optimization.OptimizationAccessor.fix_optimal_capacities`),
    the resulting pure-dispatch LP is solved, and the per-scenario operational
    time series and (scenario-conditional) duals are merged back onto ``n``. The
    scenarios are independent once capacities are fixed, but the loop runs
    serially -- hence the ``scenarios`` subset, for when only some scenarios'
    dispatch / duals are wanted.
    """
    all_scenarios = list(n.scenarios)
    selected = all_scenarios if scenarios is None else list(scenarios)
    unknown = [s for s in selected if s not in set(all_scenarios)]
    if unknown:
        msg = f"Unknown scenario(s) {unknown}; network scenarios are {all_scenarios}."
        raise ValueError(msg)

    solved: dict[str, Network] = {}
    objectives: list[str] = []
    for scenario in selected:
        ns = n.get_scenario(scenario)
        ns.optimize.fix_optimal_capacities()
        status, condition = ns.optimize(solver_name=solver_name, **solve_kwargs)
        if status != "ok":
            msg = (
                f"Dispatch re-solve for scenario {scenario!r} did not solve to "
                f"optimality (status={status!r}, condition={condition!r}); the "
                "first-stage capacities were still written back to n."
            )
            raise RuntimeError(msg)
        solved[scenario] = ns
        obj = ns.objective  # set now that the solve succeeded above
        objectives.append(f"{scenario}={obj:.6g}" if obj is not None else scenario)

    _merge_scenario_results(n, solved)
    logger.info(
        "Re-solved dispatch for %d of %d scenario(s) [%s].",
        len(selected),
        len(all_scenarios),
        ", ".join(objectives),
    )


def _merge_scenario_results(n: Network, solved: dict[str, Network]) -> None:
    """Write per-scenario dispatch results back onto the stochastic network.

    For each component output time series (``status == "Output"``) that the
    re-solve populated -- primal operations *and* scenario-conditional duals
    (e.g. ``marginal_price``) -- rebuild the ``(scenario, name)`` column
    MultiIndex with ``pd.concat(..., names=["scenario"])`` (the inverse of
    :meth:`~pypsa.Network.get_scenario` and the same construction
    :meth:`~pypsa.Network.set_scenarios` uses) and assign it onto ``n``. When
    only a subset of scenarios was solved, their columns are merged into any
    existing frame rather than replacing it. Inputs are left untouched;
    first-stage capacity outputs (``*_nom_opt``) come from the xhat write-back.
    """
    scenarios = list(solved)
    for c in n.components:
        defaults = c.defaults
        output_series = defaults.index[
            defaults["varying"] & (defaults["status"] == "Output")
        ]
        for attr in output_series:
            frames: dict[str, pd.DataFrame] = {}
            for scenario in scenarios:
                frame = solved[scenario].components[c.name].dynamic.get(attr)
                if frame is None or frame.empty:
                    frames = {}
                    break
                frames[scenario] = frame
            if not frames:
                continue
            block = pd.concat(frames, axis=1, names=["scenario"])
            existing = c.dynamic.get(attr)
            if existing is None or existing.empty:
                c.dynamic[attr] = block
            else:
                merged = existing.reindex(
                    index=existing.index.union(block.index),
                    columns=existing.columns.union(block.columns),
                )
                for col in block.columns:
                    merged[col] = block[col]
                c.dynamic[attr] = merged


def _run_solver(argv: list[str], command: str, tee: bool) -> None:
    """Run the mpi-sppy driver subprocess, raising on a non-zero exit.

    ``tee=True`` streams the driver's output live (inherits the parent's stdout
    / stderr); ``tee=False`` captures it and, on failure, folds it into the
    raised error. PyPSA never joins the MPI communicator -- mpi-sppy runs wholly
    inside this child process.
    """
    if tee:
        # Output already streamed to the terminal; nothing to capture.
        returncode = subprocess.run(argv, check=False).returncode  # noqa: S603
        stdout = stderr = ""
    else:
        completed = subprocess.run(  # noqa: S603
            argv, check=False, capture_output=True, text=True
        )
        returncode, stdout, stderr = (
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )
    if returncode != 0:
        msg = f"mpi-sppy solve failed (exit code {returncode}).\n  Command: {command}"
        if stdout:
            msg += f"\n--- stdout ---\n{stdout}"
        if stderr:
            msg += f"\n--- stderr ---\n{stderr}"
        raise RuntimeError(msg)


def solve_stochastic_mpisppy(
    n: Network,
    working_dir: str | Path | None = None,
    *,
    method: str = "ph",
    solver_name: str = "gurobi",
    first_stage: Sequence[str] | None = None,
    rho: str | float | dict[str, float] = "cost-proportional",
    rho_alpha: float = 1.0,
    rho_floor: float = 1e-3,
    file_format: str = "lp",
    cylinders: Sequence[str] = ("lagrangian", "xhatshuffle"),
    default_rho: float = 1.0,
    max_iterations: int = 50,
    config_file: str | None = None,
    mpisppy_options: dict[str, Any] | None = None,
    mpisppy_args: Sequence[str] | None = None,
    nprocs: int | None = None,
    keep_files: bool = False,
    tee: bool = True,
    model_kwargs: dict[str, Any] | None = None,
    dispatch: str = "none",
    scenarios: Sequence[str] | None = None,
) -> dict[str, float]:
    """Solve a stochastic network inline via mpi-sppy decomposition.

    This is the *inline* convenience path (laptop / single node): it writes the
    per-scenario files (:func:`write_stochastic_problem_mpisppy`), runs the mpi-sppy
    driver as a **blocking subprocess**, and reads the incumbent first stage
    back onto ``n`` (:func:`read_stochastic_solution_mpisppy`). It is the only stochastic
    entry point that needs mpi-sppy installed; the write/read phases are
    dependency-free, so large HPC runs can instead drive the three phases as
    separate scheduler jobs (see the manifest's ``sbatch_template``).

    PyPSA itself never joins the MPI communicator -- mpi-sppy runs entirely
    inside the spawned subprocess and communicates only through the files in
    ``working_dir``.

    Parameters
    ----------
    n : pypsa.Network
        A stochastic network (``n.has_scenarios`` must be ``True``).
    working_dir : str or pathlib.Path, optional
        Directory for the transfer files. ``None`` (default) uses a fresh
        temporary directory that is removed afterwards unless ``keep_files``.
        An explicit directory is cleaned of stale scenario files first but never
        auto-deleted.
    method : {"ph", "ef"}, default "ph"
        ``"ph"`` runs Progressive Hedging with bounding cylinders under
        ``mpiexec`` (the target). ``"ef"`` solves the extensive form directly in
        one process -- a correctness oracle for small problems, no MPI required.
    solver_name : str, default "gurobi"
        Subproblem solver. The PH proximal term is quadratic, so a QP/MIQP-capable
        solver (e.g. Gurobi) is used unless the proximal term is linearized.
    first_stage, rho, rho_alpha, rho_floor, file_format, model_kwargs
        Forwarded to :func:`write_stochastic_problem_mpisppy`.
    cylinders : sequence of str, default ("lagrangian", "xhatshuffle")
        PH bounding spokes; one MPI rank per cylinder plus the hub.
    default_rho : float, default 1.0
        mpi-sppy ``--default-rho`` fallback.
    max_iterations : int, default 50
        PH ``--max-iterations``.
    config_file : str, optional
        mpi-sppy ``--config-file`` forwarded verbatim (the primary way to pass
        mpi-sppy's full option surface).
    mpisppy_options : dict, optional
        Extra mpi-sppy options as a dict; ``True`` -> bare flag, other values ->
        ``--flag value``, underscores in keys become dashes.
    mpisppy_args : sequence of str, optional
        Explicit extra mpi-sppy CLI arguments, appended last (highest precedence).
    nprocs : int, optional
        MPI rank count for the inline PH run. ``None`` uses one rank per cylinder
        plus the hub. Ignored for ``method="ef"``.
    keep_files : bool, default False
        Keep a temporary ``working_dir`` after solving (no effect when
        ``working_dir`` is given).
    tee : bool, default True
        Stream the driver's output live; otherwise capture it and surface it only
        on failure.
    dispatch : {"none", "resolve"}, default "none"
        Second-stage recovery after the decomposed solve, forwarded to
        :func:`read_stochastic_solution_mpisppy`. ``"resolve"`` re-solves each scenario
        with the optimized capacities fixed, recovering the per-scenario dispatch
        and scenario-conditional duals onto ``n`` (using ``solver_name``; serial).
    scenarios : sequence of str, optional
        Restrict ``dispatch="resolve"`` to this subset of scenarios (default: all).

    Returns
    -------
    dict
        ``{on-file nonant name: value}`` applied to ``n`` as ``*_nom_opt``.

    Raises
    ------
    ImportError
        If mpi-sppy is not installed (``pip install pypsa[mpisppy]``).
    RuntimeError
        If ``method="ph"`` but ``mpiexec`` is not on ``PATH``, or the solve exits
        non-zero.

    """
    from pypsa.common import check_optional_dependency  # noqa: PLC0415

    check_optional_dependency(
        "mpisppy",
        "Missing optional dependency 'mpi-sppy' for stochastic decomposition. "
        "Install via `pip install pypsa[mpisppy]`.",
    )
    if method not in ("ph", "ef"):
        msg = f"Unknown method {method!r}; expected 'ph' or 'ef'."
        raise ValueError(msg)
    # Validate the dispatch/scenarios request now, before the (expensive) solve.
    _validate_dispatch(n, dispatch, scenarios)
    if method == "ph" and shutil.which("mpiexec") is None:
        msg = (
            "mpiexec was not found on PATH but is required to run the Progressive-"
            "Hedging cylinders. Install an MPI runtime (e.g. mpich or openmpi), or "
            "use method='ef' for a serial extensive-form solve."
        )
        raise RuntimeError(msg)

    if working_dir is None:
        directory = Path(tempfile.mkdtemp(prefix="pypsa-stochastic-"))
        cleanup = not keep_files
    else:
        directory = Path(working_dir)
        cleanup = False

    try:
        write_stochastic_problem_mpisppy(
            n,
            directory,
            clean=True,
            first_stage=first_stage,
            rho=rho,
            rho_alpha=rho_alpha,
            rho_floor=rho_floor,
            file_format=file_format,
            cylinders=cylinders,
            solver_name=solver_name,
            default_rho=default_rho,
            max_iterations=max_iterations,
            config_file=config_file,
            mpisppy_args=mpisppy_args,
            model_kwargs=model_kwargs,
        )
        argv, nranks = _solve_command(
            directory,
            method=method,
            cylinders=cylinders,
            solver_name=solver_name,
            default_rho=default_rho,
            max_iterations=max_iterations,
            config_file=config_file,
            mpisppy_options=mpisppy_options,
            mpisppy_args=mpisppy_args,
            nprocs=nprocs,
        )
        command = shlex.join(argv)
        logger.info(
            "Solving stochastic problem via mpi-sppy (method=%s, %d rank(s)):\n  %s",
            method,
            nranks,
            command,
        )
        _run_solver(argv, command, tee)
        return read_stochastic_solution_mpisppy(
            n,
            directory,
            dispatch=dispatch,
            scenarios=scenarios,
            solver_name=solver_name,
        )
    finally:
        if cleanup:
            shutil.rmtree(directory, ignore_errors=True)
