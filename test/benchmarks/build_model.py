# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Benchmark `n.optimize.create_model` against a reference installation.

    python test/benchmarks/build_model.py measure <result.json>
    python test/benchmarks/build_model.py compare <reference.json> <result.json>

`measure` times the PyPSA that is imported: each network runs in a fresh
process, so `cold` is the first build of a process and `warm` the fastest of
the next ones. The reference is `master`, measured in the same environment
with its checkout first on `PYTHONPATH`, so both sides build with the same
linopy. `compare` prints a table and fails when a warm build exceeds its
budget, the largest allowed ratio to the reference, or when both files
measure the same PyPSA. The budgets are a ratchet: lower one
to about 1.15 times its measured ratio when a change speeds the build up, until
it reaches 1.1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import pypsa

REPEATS = 5
BUDGET = {
    "ac_dc_meshed": 3.7,
    "scigrid_de": 2.45,
    "model_energy": 3.8,
    "carbon_management": 2.5,
}


def build_times(network: str) -> dict[str, float]:
    """Return the cold and the warm build time of an example network on its default backend."""
    n = getattr(pypsa.examples, network)()
    times = []
    for _ in range(REPEATS + 1):
        start = perf_counter()
        n.optimize.create_model()
        times.append(perf_counter() - start)
    return {"cold": times[0], "warm": min(times[1:])}


def measure(path: str) -> None:
    """Time every network in a process of its own and store the result."""
    results: dict[str, Any] = {"source": str(Path(pypsa.__file__).parent)}
    for network in BUDGET:
        command = [sys.executable, __file__, "build_times", network]
        run = subprocess.run(command, check=True, capture_output=True, text=True)
        results[network] = json.loads(run.stdout.splitlines()[-1])
    Path(path).write_text(json.dumps(results, indent=2) + "\n")


def compare(reference_path: str, result_path: str) -> None:
    """Print the comparison and fail when a warm build exceeds its budget."""
    reference = json.loads(Path(reference_path).read_text())
    result = json.loads(Path(result_path).read_text())
    if reference["source"] == result["source"]:
        sys.exit(f"Reference and result both measure {result['source']}")
    print("| network | build | reference [s] | result [s] | ratio | budget |")
    print("|---|---|---:|---:|---:|---:|")
    exceeded = []
    for network in BUDGET:
        for build in ("cold", "warm"):
            before, after = reference[network][build], result[network][build]
            ratio = after / before
            budget = f"{BUDGET[network]:.2f}" if build == "warm" else ""
            if build == "warm" and ratio > BUDGET[network]:
                exceeded.append(network)
            print(
                f"| `{network}` | {build} | {before:.2f} | {after:.2f} "
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
