# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Benchmark `n.optimize.create_model` against a reference installation.

    python test/benchmarks/build_model.py measure <result.json>
    python test/benchmarks/build_model.py compare <reference.json> <result.json>

`measure` times the installed PyPSA: each network runs in a fresh process, so
`cold` is the first build of a process and `warm` the fastest of the next ones.
`compare` prints a table and fails when a warm build exceeds its budget, the
largest allowed ratio to the reference. The budgets are a ratchet: lower one
to about 1.15 times its measured ratio when a change speeds the build up, until
it reaches 1.1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import linopy

import pypsa

REPEATS = 5
SPARSE = {
    "ac_dc_meshed": False,
    "scigrid_de": False,
    "model_energy": False,
    "carbon_management": True,
}
BUDGET = {
    "ac_dc_meshed": 2.05,
    "scigrid_de": 1.6,
    "model_energy": 2.15,
    "carbon_management": 1.6,
}


def build_times(network: str) -> dict[str, float]:
    """Return the cold and the warm build time of an example network."""
    model_kwargs: dict[str, bool] = {}
    if SPARSE[network]:
        model_kwargs["sparse"] = True
        if not hasattr(linopy.config, "SEMANTICS_DEPRECATION"):
            linopy.options.set_value(semantics="v1")
    n = getattr(pypsa.examples, network)()
    times = []
    for _ in range(REPEATS + 1):
        start = perf_counter()
        n.optimize.create_model(**model_kwargs)
        times.append(perf_counter() - start)
    return {"cold": times[0], "warm": min(times[1:])}


def measure(path: str) -> None:
    """Time every network in a process of its own and store the result."""
    results = {}
    for network in SPARSE:
        command = [sys.executable, __file__, "build_times", network]
        run = subprocess.run(command, check=True, capture_output=True, text=True)
        results[network] = json.loads(run.stdout.splitlines()[-1])
    Path(path).write_text(json.dumps(results, indent=2) + "\n")


def compare(reference_path: str, result_path: str) -> None:
    """Print the comparison and fail when a warm build exceeds its budget."""
    reference = json.loads(Path(reference_path).read_text())
    result = json.loads(Path(result_path).read_text())
    print("| network | backend | build | reference [s] | result [s] | ratio | budget |")
    print("|---|---|---|---:|---:|---:|---:|")
    exceeded = []
    for network, sparse in SPARSE.items():
        backend = "sparse" if sparse else "dense"
        for build in ("cold", "warm"):
            before, after = reference[network][build], result[network][build]
            ratio = after / before
            budget = f"{BUDGET[network]:.2f}" if build == "warm" else ""
            if build == "warm" and ratio > BUDGET[network]:
                exceeded.append(network)
            print(
                f"| `{network}` | {backend} | {build} | {before:.2f} | {after:.2f} "
                f"| {ratio:.2f} | {budget} |"
            )
    if exceeded:
        sys.exit(f"Warm build over budget: {', '.join(exceeded)}")


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    if command == "build_times":
        print(json.dumps(build_times(*arguments)))
    elif command == "measure":
        measure(*arguments)
    else:
        compare(*arguments)
