<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Stochastic Optimization by Decomposition (mpi-sppy)

## Overview

[Stochastic Optimization](stochastic.md) solves the two-stage problem as a single
monolithic **extensive form** (EF): one linear program that contains every
scenario. This is simple and exact, but the model grows with the number of
scenarios and can become too large to build or solve directly.

As an alternative, PyPSA can solve the same problem by **decomposition** using
[mpi-sppy](https://mpi-sppy.readthedocs.io). Instead of one big LP, each scenario
becomes an independent subproblem, and PH coordinates the shared first-stage
(investment) decisions across them, typically across many parallel MPI ranks on an
HPC cluster. The modelling is unchanged — you still define scenarios with
[`set_scenarios`](stochastic.md#scenario-definition) — only the solve differs.

!!! note "When to use decomposition"

    The monolithic EF (`n.optimize()`) is the right default for small and medium
    stochastic problems. Reach for decomposition when the EF becomes too large to
    build or solve directly and you want to spread the scenario subproblems across
    many cores or cluster nodes.

PyPSA and mpi-sppy meet only at a **file boundary**: PyPSA writes one optimization
file per scenario plus small metadata files, mpi-sppy solves them, and PyPSA reads
the solution back. PyPSA never joins the MPI job, and mpi-sppy never imports PyPSA.

## Installation

On top of a working PyPSA installation, the decomposition path needs three further
pieces: a working MPI with `mpi4py`, `mpi-sppy` itself, and a solver reached through
a Pyomo **persistent** interface. (mpi-sppy will also work with solvers that are not
persistent, just more slowly.) None of this is needed for
the `write`/`read` helpers — only the parallel `solve_stochastic` driver and the
mpi-sppy solve phase of the decoupled workflow use MPI and mpi-sppy.

### MPI and mpi4py

Progressive Hedging runs the scenario subproblems in parallel under MPI, so you need
an MPI runtime providing `mpiexec`/`mpirun` (MPICH or Open MPI) and an `mpi4py` built
against it. Follow the mpi-sppy README's *Install mpi4py* instructions
([readthedocs](https://mpi-sppy.readthedocs.io/en/latest/install_mpi.html)); in
short, either install both through conda

```bash
conda install openmpi    # then, in that order:
conda install mpi4py
```

or, to compile `mpi4py` against an MPI you already have, install it through pip

```bash
pip install mpi4py
```

Then verify the installation the way mpi-sppy recommends — from a clone of the
mpi-sppy repository:

```bash
mpirun -n 2 python -m mpi4py mpi_one_sided_test.py
```

No error messages means you have an MPI installation that should work well. (Even
with an error message mpi-sppy may still run and return correct results, just
potentially much slower.)

!!! warning "MPICH on HPC clusters"

    On some HPC platforms using an MPICH implementation you must
    `export MPICH_ASYNC_PROGRESS=1` before running, or run-times can inflate by a
    factor of 2–4 and large rank counts (≫ 10) can stall once scenarios are created.
    See the *Install mpi4py* section of the mpi-sppy README for details.

### mpi-sppy

Install `mpi-sppy` from GitHub into the **same environment as PyPSA** — the PyPI
release can lag behind the features the interface relies on (such as the MPS-file
loader):

```bash
pip install "git+https://github.com/Pyomo/mpi-sppy.git"
```

or, for a checkout you can edit, clone first and install editable (add the `[mpi]`
extra to pull in `mpi4py` at the same time):

```bash
git clone https://github.com/Pyomo/mpi-sppy.git
pip install -e "./mpi-sppy[mpi]"
```

!!! tip "The `mpisppy` extra"

    `pip install "pypsa[mpisppy]"` does this GitHub install for you — the extra pulls
    `mpi-sppy` straight from `main` (along with `mpi4py` and `mip`, the coin-or parser
    mpi-sppy uses to read the per-scenario LP/MPS files). Still set up MPI and `mpi4py`
    first (above): installing `mpi4py` through the extra just compiles it against
    whatever MPI `pip` happens to find.

### A persistent QP solver

Some decomposition algorithms, such as Progressive Hedging (PH), add a quadratic
proximal term to each scenario subproblem and re-solve it on every iteration, so the
subproblem solver must be reached through a Pyomo
**persistent** interface — that lets mpi-sppy update the objective in place between
iterations instead of rebuilding the model each time. You select the solver by name
with `solver_name=` (see [Passing mpi-sppy options](#passing-mpi-sppy-options)).

The commercial solvers expose persistent interfaces that handle the quadratic
proximal term directly — `gurobi_persistent`, `cplex_persistent`,
`xpress_persistent` — and become available once the solver and its Python bindings
are installed in the environment (for example `pip install gurobipy` enables
`gurobi_persistent`).

For an open-source option, use HiGHS through Pyomo's APPSI (Auto-Persistent Pyomo
Solver Interface) as `solver_name="appsi_highs"`. APPSI is persistent, but HiGHS
cannot take the quadratic proximal term directly, so for PH, you must also
**linearize** it
with `--linearize-proximal-terms` (and `--linearize-binary-proximal-terms` if any
first-stage variables are binary). Pass that through the options channel — inline as
`mpisppy_options={"linearize_proximal_terms": True}`, or, since that channel is
inline-only, as `mpisppy_args=["--linearize-proximal-terms"]` for the decoupled
workflow (see [Passing mpi-sppy options](#passing-mpi-sppy-options)).

The extensive-form oracle (`method="ef"`) is a single solve with no proximal term, so
it needs neither a persistent interface nor linearization — a plain
`solver_name="gurobi"` (or `"appsi_highs"`) is fine there. However, there is no
reason to use the EF option in mpi-sppy because the solver already in PyPSA does the
same thing with less overhead.

## Decoupled workflow (HPC / SLURM)

On a cluster you usually cannot hold a Python process open while a large MPI job
queues and runs, and the three phases have very different resource profiles (one core
to write, many cores to solve, one core to read) and even different software
environments (PyPSA vs mpi-sppy). For this, drive the phases as **separate jobs**
using the two dependency-free helpers.

```python
# Phase 1 — PyPSA: write one file per scenario, the metadata, and a manifest.
manifest = n.optimize.write_stochastic_problem("/shared/run42")
```

`write_stochastic_problem` returns (and writes) a manifest describing the run. Two
keys drive the remaining phases:

- `manifest["solve_command"]` — the exact `mpisppy.generic_cylinders` command to run
  for phase 2 (the same command the inline driver builds; see
  [Passing mpi-sppy options](#passing-mpi-sppy-options) for how to shape it).
- `manifest["sbatch_template"]` — a copy-paste SLURM dependency chain that wires the
  three phases together with `afterok` and includes the directory-hygiene `rm` (see
  the warning below). For `/shared/run42` with the default two cylinders
  (`manifest["mpi_ranks"]` = 3) it expands to:

```bash
# Decoupled SLURM workflow (edit envs/partitions/accounts to taste).
DIR=/shared/run42
rm -f "$DIR"/*.lp "$DIR"/*.mps "$DIR"/*_nonants.json "$DIR"/*_rho.csv  # hygiene
j1=$(sbatch --parsable write.sbatch)                           # -n 1,   PyPSA env
j2=$(sbatch --parsable --dependency=afterok:$j1 solve.sbatch)  # -n 3, mpi-sppy env
sbatch          --dependency=afterok:$j2 read.sbatch          # -n 1,   PyPSA env
# solve.sbatch runs, e.g.:
#   srun python -m mpi4py -m mpisppy.generic_cylinders --mps-files-directory $DIR ...
```

The `sbatch_template` only *submits and chains* the jobs — it does not write them.
You provide the three batch scripts it names (`write.sbatch`, `solve.sbatch`,
`read.sbatch`), since their `#SBATCH` headers, module loads and environment activation
are cluster-specific and PyPSA cannot fill them in. `write.sbatch` and `read.sbatch`
are single-core PyPSA jobs (the `write_stochastic_problem` and
`read_stochastic_solution` calls); `solve.sbatch` runs `manifest["solve_command"]` on
`manifest["mpi_ranks"]` ranks in the mpi-sppy environment, launched with `srun` in
place of the manifest's `mpiexec -np`. A minimal `solve.sbatch` looks like:

```bash
#!/bin/bash
#SBATCH --job-name=stoch-solve
#SBATCH --ntasks=3                    # = manifest["mpi_ranks"]
#SBATCH --time=02:00:00
#SBATCH --partition=<your-partition>  # plus --account etc. for your site

module load mpi             # however your site provides MPI
source activate mpi-sppy    # the environment with mpi-sppy and a solver

# manifest["solve_command"], with srun as the launcher instead of `mpiexec -np`:
srun python -m mpi4py -m mpisppy.generic_cylinders \
    --mps-files-directory /shared/run42 \
    --solver-name gurobi_persistent --lagrangian --xhatshuffle \
    --default-rho 1.0 --max-iterations 50 \
    --write-xhat-file /shared/run42/xhat.csv
```

Phase 3 then reads the solution back:

```python
# Phase 3 — PyPSA: read the optimized first stage (optionally re-solve dispatch).
n.optimize.read_stochastic_solution("/shared/run42", dispatch="resolve")
```

!!! warning "Directory hygiene"

    mpi-sppy discovers scenarios by scanning the transfer directory for model files,
    so a previous run that wrote *more* scenarios would leave stale files that a
    smaller new run silently picks up as phantom scenarios.
    `write_stochastic_problem(clean=True)` (the default) clears them in Python; the
    `sbatch_template` also `rm`s them as a belt-and-braces step covering an aborted
    write.

## Inline solve

For a laptop or a single interactive node, `solve_stochastic` runs the whole
workflow in one blocking call: write the per-scenario files, run the mpi-sppy driver
as a subprocess, and read the optimized first-stage capacities back onto the network
as `*_nom_opt`.

```python
import pypsa

n = pypsa.Network()
# ... build the network ...
n.set_scenarios({"low": 0.3, "med": 0.4, "high": 0.3})
# ... set per-scenario data ...

n.optimize.solve_stochastic(method="ph", solver_name="gurobi_persistent")
```

`method="ph"` (the default) runs Progressive Hedging with bounding cylinders under
`mpiexec`. `method="ef"` instead builds and solves the extensive form through
mpi-sppy in a single process; it needs no MPI and is mainly useful as a small-scale
correctness oracle (it should match `n.optimize()`).

By default only the first-stage capacities are written back. To recover the second
stage, use `dispatch=` (see
[Recovering the second stage](#recovering-the-second-stage-dispatch-and-prices)).

## Passing mpi-sppy options

Both entry points — the inline `solve_stochastic` driver and the `solve_command`
baked into the decoupled manifest — ultimately build a single
`mpisppy.generic_cylinders` command line. You shape it at three levels, from most to
least convenient.

**1. Named keyword arguments** for the common Progressive-Hedging controls, accepted
by both `solve_stochastic` and `write_stochastic_problem`:

- `solver_name=` — the persistent subproblem solver, e.g. `"gurobi_persistent"`
  (→ `--solver-name`); see [A persistent QP solver](#a-persistent-qp-solver).
- `cylinders=` — the bounding spokes, default `("lagrangian", "xhatshuffle")`
  (→ one `--<cylinder>` flag each).
- `default_rho=`, `max_iterations=` (→ `--default-rho`, `--max-iterations`), plus the
  per-variable `rho=` policy (default `"cost-proportional"`) written into the
  per-scenario `_rho.csv` files.

`solve_stochastic` additionally takes `nprocs=` to override the MPI rank count
(default `1 + len(cylinders)`: a Progressive-Hedging hub rank plus one per cylinder).

**2. An options dict**, `mpisppy_options=` (inline `solve_stochastic` only). Each
entry becomes a CLI flag: the key gets a `--` prefix with underscores turned to
dashes, `True` becomes a bare flag, and `False`/`None` are dropped. For example,

```python
n.optimize.solve_stochastic(
    solver_name="gurobi_persistent",
    mpisppy_options={"rel_gap": 0.01, "max_solver_threads": 4, "presolve": True},
)
# appends:  --rel-gap 0.01 --max-solver-threads 4 --presolve
```

**3. Escape hatches** for anything the helpers don't model: `mpisppy_args=` (a list
appended to the command verbatim) and `config_file=` (→ `--config-file`), both
accepted by `solve_stochastic` and `write_stochastic_problem`. Use these to reach any
option in `python -m mpisppy.generic_cylinders --help`.

!!! note "Tuning the decoupled solve"

    `write_stochastic_problem` bakes the named arguments, `config_file=` and
    `mpisppy_args=` into `manifest["solve_command"]` — but **not** `mpisppy_options=`,
    which is inline-only. On a cluster, drive your tuning through the named arguments
    and `mpisppy_args=`/`config_file=` so it survives into the command `solve.sbatch`
    actually runs.

## Recovering the second stage (dispatch and prices)

The decomposed solve returns the optimized **first-stage** capacities. The
per-scenario **dispatch** (operational time series) and **marginal prices** are
recovered separately via the `dispatch` argument of `solve_stochastic` (and of
`read_stochastic_solution`):

```python
# recover per-scenario dispatch + scenario-conditional duals for all scenarios
n.optimize.solve_stochastic(method="ph", dispatch="resolve")

# ... or only for the scenarios you care about (e.g. the stressed one)
n.optimize.solve_stochastic(method="ph", dispatch="resolve", scenarios=["high"])
```

`dispatch="resolve"` fixes the optimized capacities and re-solves each scenario as
an independent dispatch problem, writing the operational results
(`n.generators_t.p`, `n.storage_units_t.state_of_charge`, …) and the marginal prices
(`n.buses_t.marginal_price`) onto the network's `(scenario, name)` columns. Because
the re-solves are independent, you can restrict them to a subset with `scenarios=`.

!!! note "Scenario-conditional duals"

    The marginal prices from `dispatch="resolve"` are **conditional on the realized
    scenario** — the marginal values in that scenario's world. They differ from the
    extensive form's probability-weighted duals (the EF dual is the
    scenario-conditional dual scaled by the scenario probability) and are usually
    what a decision maker wants. The re-solves run serially, which is a further
    reason to narrow them with `scenarios=` when only some scenarios are of interest.

## Performance note

The file interface is not fast — each run writes and re-parses one LP/MPS file per
scenario. For the large, hard problems decomposition targets this overhead is
negligible (wall time is dominated by the subproblem solves), but for small problems
the monolithic `n.optimize()` is usually quicker. Decomposition pays off when the
extensive form is too large to handle directly and you can spread the scenario
subproblems across many cores.

## See also

- [Stochastic Optimization](stochastic.md) — the modelling layer (`set_scenarios`,
  the extensive form, CVaR risk preferences).
- [Optimization Methods](../../api/networks/optimize.md) — API reference for
  `solve_stochastic`, `write_stochastic_problem` and `read_stochastic_solution`.
