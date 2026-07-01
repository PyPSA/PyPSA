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
the `write`/`read` helpers — only the parallel `solve_stochastic_mpisppy` driver and the
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

    `pip install "pypsa[mpisppy]"` pulls the PyPI-installable helpers — `mpi4py` and
    `mip` (the coin-or parser mpi-sppy uses to read the per-scenario LP/MPS files) —
    but **not `mpi-sppy` itself**, which you install with the GitHub command above
    until a PyPI release ships the MPS-file loader. Set up MPI and `mpi4py` first
    (above): installing `mpi4py` through the extra just compiles it against whatever
    MPI `pip` happens to find.

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
environments (PyPSA vs mpi-sppy). For this, drive the phases as **separate SLURM
jobs** using the two dependency-free helpers. The worked example below stages a run
in the shared directory `/shared/run42`; throughout, `pypsa` and `mpi-sppy` name two
conda environments (they may be the same one if you installed everything together).

### Phase 1 — write (PyPSA, one core)

A single-core PyPSA job builds the network and calls `write_stochastic_problem_mpisppy`. Any
mpi-sppy options you want in the solve — here the `--coeff-rho` rho setter — are
passed now through `mpisppy_args=` so they are baked into the recorded
`solve_command`:

```python
# write_step.py
from my_model import build_network  # your code: returns a network with scenarios set

n = build_network()
manifest = n.optimize.write_stochastic_problem_mpisppy(
    "/shared/run42",
    solver_name="gurobi_persistent",
    max_iterations=200,
    mpisppy_args=["--coeff-rho"],  # coefficient-based rho setter (see below)
)
print(manifest["solve_command"])   # the exact phase-2 command, --coeff-rho included
```

run it from the one-task batch script `write.sbatch`:

```bash
#!/bin/bash
# write.sbatch
#SBATCH --job-name=stoch-write
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --partition=<your-partition>

source activate pypsa   # the environment with PyPSA
python write_step.py
```

`write_stochastic_problem_mpisppy` returns (and writes to
`/shared/run42/pypsa_stochastic_manifest.json`) a manifest whose keys drive the rest:

- `manifest["solve_command"]` — the exact `mpisppy.generic_cylinders` command for
  phase 2, with your `--coeff-rho` already in it (see
  [Passing mpi-sppy options](#passing-mpi-sppy-options)).
- `manifest["mpi_ranks"]` — how many ranks that command needs (here 3: a
  Progressive-Hedging hub plus one per cylinder).
- `manifest["sbatch_template"]` — a copy-paste driver that *submits and chains* the
  three jobs with `afterok` (and does the directory-hygiene `rm`). For `/shared/run42`
  with the default two cylinders it expands to:

```bash
# Decoupled SLURM workflow (edit envs/partitions/accounts to taste).
DIR=/shared/run42
rm -f "$DIR"/*.lp "$DIR"/*.mps "$DIR"/*_nonants.json "$DIR"/*_rho.csv "$DIR"/xhat.csv  # hygiene
j1=$(sbatch --parsable write.sbatch)                           # -n 1,   PyPSA env
j2=$(sbatch --parsable --dependency=afterok:$j1 solve.sbatch)  # -n 3, mpi-sppy env
sbatch          --dependency=afterok:$j2 read.sbatch          # -n 1,   PyPSA env
# solve.sbatch runs, e.g.:
#   srun python -m mpi4py -m mpisppy.generic_cylinders --mps-files-directory $DIR ...
```

The driver only *references* the three job scripts — you supply them, because their
`#SBATCH` headers, module loads and environment activation are cluster-specific and
PyPSA cannot fill them in.

### Phase 2 — solve (mpi-sppy, many ranks)

`solve.sbatch` runs `manifest["solve_command"]` on `manifest["mpi_ranks"]` ranks in
the mpi-sppy environment, launched with `srun` in place of the manifest's
`mpiexec -np`:

```bash
#!/bin/bash
# solve.sbatch
#SBATCH --job-name=stoch-solve
#SBATCH --ntasks=3                    # manifest["mpi_ranks"]: 1 PH hub + 2 cylinders
#SBATCH --time=02:00:00
#SBATCH --partition=<your-partition>  # plus --account etc. for your site

module load mpi             # however your site provides MPI
source activate mpi-sppy    # the environment with mpi-sppy and a solver

# manifest["solve_command"], with srun as the launcher instead of `mpiexec -np 3`.
# The run needs exactly manifest["mpi_ranks"] ranks (one per cylinder, plus the hub):
srun -n "$SLURM_NTASKS" python -m mpi4py -m mpisppy.generic_cylinders \
    --mps-files-directory /shared/run42 \
    --solver-name gurobi_persistent --lagrangian --xhatshuffle \
    --default-rho 1.0 --max-iterations 200 --max-solver-threads 2 \
    --write-xhat-file /shared/run42/xhat.csv --coeff-rho
```

### Phase 3 — read (PyPSA, one core)

A final single-core PyPSA job rebuilds the same network and reads the incumbent first
stage back onto it (optionally re-solving the dispatch):

```python
# read_step.py
from my_model import build_network  # the same network as in write_step.py

n = build_network()
n.optimize.read_stochastic_solution_mpisppy("/shared/run42", dispatch="resolve")
# n now carries the optimized *_nom_opt capacities and, with dispatch="resolve",
# the per-scenario dispatch and marginal prices — inspect or persist as you like.
```

run it from the one-task batch script `read.sbatch`:

```bash
#!/bin/bash
# read.sbatch
#SBATCH --job-name=stoch-read
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --partition=<your-partition>

source activate pypsa
python read_step.py
```

!!! warning "Directory hygiene"

    mpi-sppy discovers scenarios by scanning the transfer directory for model files,
    so a previous run that wrote *more* scenarios would leave stale files that a
    smaller new run silently picks up as phantom scenarios.
    `write_stochastic_problem_mpisppy(clean=True)` (the default) clears them in Python; the
    `sbatch_template` also `rm`s them as a belt-and-braces step covering an aborted
    write.

## Inline solve

For a laptop or a single interactive node, `solve_stochastic_mpisppy` runs the whole
workflow in one blocking call: write the per-scenario files, run the mpi-sppy driver
as a subprocess, and read the optimized first-stage capacities back onto the network
as `*_nom_opt`.

```python
import pypsa

n = pypsa.Network()
# ... build the network ...
n.set_scenarios({"low": 0.3, "med": 0.4, "high": 0.3})
# ... set per-scenario data ...

n.optimize.solve_stochastic_mpisppy(method="ph", solver_name="gurobi_persistent")
```

`method="ph"` (the default) runs Progressive Hedging with bounding cylinders under
`mpiexec`. `method="ef"` instead builds and solves the extensive form through
mpi-sppy in a single process; it needs no MPI and is mainly useful as a small-scale
correctness oracle (it should match `n.optimize()`).

By default only the first-stage capacities are written back. To recover the second
stage, use `dispatch=` (see
[Recovering the second stage](#recovering-the-second-stage-dispatch-and-prices)).

## Passing mpi-sppy options

Both entry points — the inline `solve_stochastic_mpisppy` driver and the `solve_command`
baked into the decoupled manifest — ultimately build a single
`mpisppy.generic_cylinders` command line. You can shape it at three levels, from most to
least convenient.

**1. Named keyword arguments** for the common Progressive-Hedging controls, accepted
by both `solve_stochastic_mpisppy` and `write_stochastic_problem_mpisppy`:

- `solver_name=` — the persistent subproblem solver, e.g. `"gurobi_persistent"`
  (→ `--solver-name`); see [A persistent QP solver](#a-persistent-qp-solver).
- `cylinders=` — the bounding spokes, default `("lagrangian", "xhatshuffle")`
  (→ one `--<cylinder>` flag each).
- `default_rho=`, `max_iterations=` (→ `--default-rho`, `--max-iterations`), plus the
  per-variable `rho=` policy (default `"cost-proportional"`) written into the
  per-scenario `_rho.csv` files.

`solve_stochastic_mpisppy` additionally takes `nprocs=` to override the MPI rank count
(default `1 + len(cylinders)`: a Progressive-Hedging hub rank plus one per cylinder).

!!! note "PH caps solver threads by default"

    **If you specify nothing**, the PyPSA interface to mpi-sppy caps each rank's
    subproblem solver at `--max-solver-threads 2` so that ranks sharing a node do
    not oversubscribe cores (you'll see it in `manifest["solve_command"]` and the
    `solve.sbatch` above). This applies to PH only — the single-process EF is left
    **uncapped**
    (it uses the solver's own default). To use a different cap, pass your own value
    — `mpisppy_options={"max_solver_threads": 4}` or
    `mpisppy_args=["--max-solver-threads", "4"]` — which **replaces** the default
    (the flag is not emitted twice).

**2. An options dict**, `mpisppy_options=` (inline `solve_stochastic_mpisppy` only). Each
entry becomes a CLI flag: the key gets a `--` prefix with underscores turned to
dashes, `True` becomes a bare flag, and `False`/`None` are dropped. For example,

```python
n.optimize.solve_stochastic_mpisppy(
    solver_name="gurobi_persistent",
    mpisppy_options={"rel_gap": 0.01, "max_solver_threads": 4, "presolve": True},
)
# appends:  --rel-gap 0.01 --max-solver-threads 4 --presolve
```

**3. Escape hatches** for anything the helpers don't model. `mpisppy_args=` is a list
of raw tokens appended to the command verbatim — best for a handful of extra flags —
while `config_file=` (→ `--config-file`) points at a full mpi-sppy config file, better
for a large or reusable option set. Both are accepted by `solve_stochastic_mpisppy` and
`write_stochastic_problem_mpisppy`. Between them you can reach any `generic_cylinders` option;
for the full list and what each one does, see the mpi-sppy
[`generic_cylinders` documentation](https://mpi-sppy.readthedocs.io/en/latest/generic_cylinders.html)
or run `python -m mpisppy.generic_cylinders --help`.

!!! note "Adapting rho with a rho setter"

    `rho=` and `default_rho=` set the *initial* penalty. mpi-sppy can also **adapt**
    rho during PH with a *rho setter*, enabled by an `mpisppy_args` flag —
    `--coeff-rho` (coefficient-based), `--sep-rho`, `--sensi-rho` or `--grad-rho` (see
    the mpi-sppy
    [rho-setting documentation](https://mpi-sppy.readthedocs.io/en/latest/rho_setting.html)).
    At most one may be set, and it builds on the initial values from `rho=`. `--coeff-rho`
    derives rho from each variable's objective coefficient, so it needs no extra input
    and works directly with the file interface:

    ```python
    n.optimize.solve_stochastic_mpisppy(
        solver_name="gurobi_persistent", mpisppy_args=["--coeff-rho"]
    )
    ```

!!! note "Tuning the decoupled solve"

    `write_stochastic_problem_mpisppy` bakes the named arguments, `config_file=` and
    `mpisppy_args=` into `manifest["solve_command"]` — but **not** `mpisppy_options=`,
    which is inline-only. On a cluster, drive your tuning through the named arguments
    and `mpisppy_args=`/`config_file=` so it survives into the command `solve.sbatch`
    actually runs.

## Recovering the second stage (dispatch and prices)

The decomposed solve returns the optimized **first-stage** capacities. The
per-scenario **dispatch** (operational time series) and **marginal prices** are
recovered through the `dispatch=` argument, which both entry points accept — the
inline `solve_stochastic_mpisppy` and the decoupled `read_stochastic_solution_mpisppy`
(Phase 3 of the SLURM workflow). The recovery itself is identical either way:

```python
# Inline: solve and recover in one call (dispatch + scenario-conditional duals)
n.optimize.solve_stochastic_mpisppy(method="ph", dispatch="resolve")

# Decoupled: recover while reading the incumbent back, optionally only for the
# scenarios you care about (e.g. the stressed one)
n.optimize.read_stochastic_solution_mpisppy(
    "/shared/run42", dispatch="resolve", scenarios=["high"]
)
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
  `solve_stochastic_mpisppy`, `write_stochastic_problem_mpisppy` and `read_stochastic_solution_mpisppy`.
