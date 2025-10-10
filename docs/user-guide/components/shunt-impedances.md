<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Shunt Impedance

The [`ShuntImpedance`][pypsa.components.ShuntImpedances] components attach to a single bus and have a
voltage-dependent admittance. For shunt impedances the power consumption is
given by $s_i = |V_i|^2 y_i^*$ so that $p_i + j q_i = |V_i|^2 (g_i -jb_i)$.
However the `p` and `q` below are defined directly proportional to `g` and `b`
with $p = |V|^2g$ and $q = |V|^2b$, thus if $p>0$ the shunt impedance is
consuming active power from the bus and if $q>0$ it is supplying reactive power
(i.e. behaving like an capacitor).

!!! note

    Shunt impedances are only used in power flow calculations ([`n.pf()`][pypsa.Network.pf]), not in any of the optimisation problems ([`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__]).

{{ read_csv('../../../pypsa/data/component_attrs/shunt_impedances.csv', disable_numparse=True) }} 