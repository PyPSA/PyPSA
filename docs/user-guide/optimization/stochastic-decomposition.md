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

As an alternative, PyPSA can solve the same problem by **decomposition** —
Progressive Hedging (PH) with bounding cylinders — using
[mpi-sppy](https://github.com/Pyomo/mpi-sppy). Instead of one big LP, each scenario
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

The decomposition path is an optional extra:

```bash
pip install "pypsa[mpisppy]"
```

This installs `mpi-sppy`, `mpi4py` and `mip`. Progressive Hedging runs the
subproblem solvers in parallel under MPI, so you also need an MPI runtime providing
`mpiexec` (e.g. MPICH or Open MPI) and a solver such as Gurobi or HiGHS. The
`write`/`read` helpers below need neither mpi-sppy nor MPI — only the inline
`solve_stochastic` driver does.

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

n.optimize.solve_stochastic(method="ph", solver_name="gurobi")
```

`method="ph"` (the default) runs Progressive Hedging with bounding cylinders under
`mpiexec`. `method="ef"` instead builds and solves the extensive form through
mpi-sppy in a single process; it needs no MPI and is mainly useful as a small-scale
correctness oracle (it should match `n.optimize()`).

By default only the first-stage capacities are written back. To recover the second
stage, use `dispatch=` (next section).

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
keys help drive the remaining phases:

- `manifest["solve_command"]` — the exact mpi-sppy command to run for phase 2.
- `manifest["sbatch_template"]` — a copy-paste SLURM dependency chain
  (`write → solve → read`, chained with `afterok`), including the directory-hygiene
  `rm` (see the warning below).

Phase 2 runs that mpi-sppy command in the mpi-sppy environment, across many ranks.
Phase 3 then reads the solution back, again in the PyPSA environment:

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
