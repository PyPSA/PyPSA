# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pypsa
from pypsa.optimization.spec.build import SPEC


def test_every_declared_trait_has_one_file():
    declared = {t for c in pypsa.Network().components.values() for t in c._formulation}
    stated = {path.name.removesuffix(".yaml") for path in (SPEC / "traits").iterdir()}
    assert declared == stated
