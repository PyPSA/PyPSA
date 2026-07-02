<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# SMS++ Optimization

PyPSA can use [SMS++](https://gitlab.com/smspp/smspp-project) as an optional
optimization backend through the Python packages
[pypsa2smspp](https://github.com/SPSUnipi/pypsa2smspp) and
[pySMSpp](https://github.com/SPSUnipi/pySMSpp).

The integration follows the same high-level workflow as
[`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__]:

1. convert the PyPSA network to an SMS++ model with `pypsa2smspp`,
2. solve the SMS++ model through `pySMSpp`,
3. retrieve the solution and write it back to the PyPSA network.

The SMS++ backend is optional. Install the Python interface with PyPSA's
`smspp` extra and install the native SMS++ binaries, for example from
conda-forge:

```bash
pip install "pypsa[smspp]"
conda install -c conda-forge smspp-project
```

The PyPSA extra depends on `pypsa2smspp>=0.0.3` and `pysmspp`. The native
`smspp-project` package provides the SMS++ executables used by `pySMSpp`.

See also the
[:material-notebook: SMS++ optimization example](../../examples/smspp-optimization.ipynb).

## One-Step Usage

To solve a network with SMS++, pass `solver_name="smspp"` to
[`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__].
The call returns PyPSA's usual `(status, condition)` tuple and writes the
solution back to the network.

```python
import pypsa

n = pypsa.examples.ac_dc_meshed()

# Global constraints are not yet supported by the SMS++ conversion.
n.remove("GlobalConstraint", n.global_constraints.index)

status, condition = n.optimize(solver_name="smspp")
```

Keyword arguments for `pypsa2smspp.Transformation` can be passed through
`solver_options`:

```python
status, condition = n.optimize(
    solver_name="smspp",
    solver_options={
        "workdir": "output",
        "name": "test_case",
    },
)
```

As with other solver calls, keyword arguments can also be passed directly to
`n.optimize()`; direct keyword arguments take precedence over entries in
`solver_options`.

These options are interpreted by `pypsa2smspp`, not by Linopy. See the
[pypsa2smspp examples](https://github.com/SPSUnipi/pypsa2smspp/tree/main/docs/examples)
for backend-specific configuration patterns.

## Step-by-Step Usage

The SMS++ accessor is also available as `n.optimize.smspp` for users who want
to inspect or run the conversion, solve, and retrieval steps separately:

```python
smspp = n.optimize.smspp

sms_network = smspp.create_model(
    solver_options={
        "workdir": "output",
        "name": "test_case",
    }
)

status, condition = smspp.solve_model()
```

After `solve_model()`, optimized component values such as dispatch and
optimized capacities are available on the PyPSA network in the usual result
fields.

## Notes

The SMS++ path is independent of PyPSA's default Linopy model construction.
Objects and options exposed by `pypsa2smspp` and `pySMSpp` therefore follow the
SMS++ backend rather than Linopy's solver interface.

Supported components and formulations are determined by `pypsa2smspp`. For
advanced examples, supported options, and the latest conversion details, refer
to the upstream projects:

- [SPSUnipi/pypsa2smspp](https://github.com/SPSUnipi/pypsa2smspp)
- [SPSUnipi/pySMSpp](https://github.com/SPSUnipi/pySMSpp)
