# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Data prep of the spec: one-port units."""

from pypsa.optimization.spec.sources import relation, source

source("COMPONENT_bus")(relation)
