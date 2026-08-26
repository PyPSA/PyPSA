<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Optimization Methods

::: pypsa.optimization.OptimizationAccessor
    options:
        heading_level: 2
        show_bases: False
        inherited_members: true

::: pypsa.optimization.mga
    options:
        heading_level: 2
        filters:
          - "!^_[^_]"
          - "!logger"
          - "!OptimizationAbstractMGAMixin"

## Stochastic decomposition (mpi-sppy)

The functions behind the `n.optimize.*_mpisppy` accessor methods; see the
[Stochastic Optimization by Decomposition](../../user-guide/optimization/stochastic-decomposition.md)
guide.

::: pypsa.optimization.stochastic_mpisppy
    options:
        heading_level: 3
        members:
          - write_stochastic_problem_mpisppy
          - read_stochastic_solution_mpisppy
          - solve_stochastic_mpisppy
