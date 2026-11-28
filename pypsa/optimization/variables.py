# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The variable definitions, removed in PyPSA 1.5.0: `n.optimize()` builds the model from `pypsa/optimization/spec/`."""

from typing import TYPE_CHECKING

from pypsa.optimization._removed import removed_names

if not TYPE_CHECKING:
    __getattr__ = removed_names(__name__)
