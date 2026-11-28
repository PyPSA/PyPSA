<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Linearised Power Flow

## Kirchhoff's Current Law (KCL)

The Kirchhoff's Current Law (KCL) mandates that the sum of power flows into a
bus must equal the sum of power flows out of the bus at each time step. It is
covered by the more general [Energy Balance](energy-balance.md) constraints
which is also applied to non-electric buses. For electric buses, KCL is the
specific form of the energy balance constraint.

## Kirchhoff's Voltage Law (KVL)

For lines and transformers, whose power flows according the impedances, the
power flow $p_{l,s,t}$ in AC networks is governed by the cycle-based linearised formulation
of Kirchhoff's Voltage Law (KVL)

<!-- md:spec Kirchhoff_Voltage_Law Cycle_angle_sum Line_angle_sum -->

where the weight `Line_cycle_weight` is the entry of a [cycle basis
matrix](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.cycle_basis.html)
of the network graph times the series reactance $x_l$ of line $l$. The independent cycles $\xi$ are expressed as directed
linear combinations of lines and transformers. The constant factor $10^5$ only improves the numerical scaling.

While there are different formulations of KVL, the cycle-based formulation was
found to be substantially faster than other formulations due to its sparsity, as
shown in Hörsch et al. (2018)[^1]. This formulation defines the same feasible
space as other standard linearised formulations based on voltage angles that is
commonly found in textbooks (B-Theta) or the formulation based on Power Transfer
Distribution Factors (PTDFs).

By default, PyPSA constructs the cycle basis from breadth-first spanning trees by Deo et
al. (1982)[^2] and refines it through cycle exchanges to shorten the cycles
(`n.optimize(cycle_basis_method="bfs-refined")`).

These constraints carry the name `Kirchhoff-Voltage-Law`.

!!! tip "Tip: KVL with DC networks"

    For DC networks, replace the series reactance $x_l$ by the series resistance (`r_pu_eff`).

!!! note "Note: Retrieving the PTDF matrix"

    The PTDF matrix of sub-networks (i.e. synchronous zones) can be calculated with [`sn.calculate_PTDF()`][pypsa.networks.SubNetwork.calculate_PTDF] method.

!!! tip "Tip: Using the [`Link`][pypsa.components.Links] component for Net Transfer Capacities (NTCs)"

    For simplified transmission representation using Net Transfer Capacities (NTCs), use the [`Link`][pypsa.components.Links] component with controllable power flow like a transport model. The [`Link`][pypsa.components.Links] component can also be used to represent a point-to-point HVDC link.

!!! note "Note: Handling impedance changes with transmission expansion"

    If $P_l$ is also subject to optimisation (`s_nom_extendable=True`), the impedance $x$ of the line is **not** automatically changed with the capacity (e.g. to represent added parallel lines).
    However, the extension [`n.optimize.optimize_transmission_expansion_iteratively()`][pypsa.optimization.OptimizationAccessor.optimize_transmission_expansion_iteratively] covers this through an
    iterative process as done Hagspiel et al. (2014)[^3] .

## Phase-Shifting Transformers (PSTs)

A phase-shifting transformer (PST) inserts a controllable voltage phase-angle
shift between its terminals to redistribute power flows in a meshed network,
without adding or removing active power. Following the linearised power-flow
treatment of PSTs in Verboomen et al. (2008)[^4] (Section III-A "Linearized
Power Flow", eq. (3)), the phase shift enters the cycle-based KVL constraint
as an additive angle term in the transformer part of the cycle sum:

<!-- md:spec Transformer_angle_sum -->

where $\phi_{s,t,q}$ is the voltage phase-angle shift of transformer $q$ in radians. The term `Transformer_phase_shift_weight` holds a fixed phase shift.

For a `Transformer` with `phase_shift_min < phase_shift_max`, the phase shift
becomes a per-snapshot decision variable with bounds

<!-- md:spec Transformer_phase_shift -->

given by the input attributes `phase_shift_min` and `phase_shift_max` (in
degrees, converted to radians internally). Otherwise the phase shift is fixed
to the static or series input `phase_shift`. The realised per-snapshot
angle is written to the dynamic output `n.transformers_t["phase_shift_opt"]`.

The variables carry the name `Transformer-phase_shift`; the angle term enters
the `Kirchhoff-Voltage-Law` constraints.

!!! note "Note: Sign convention"

    PyPSA's power-flow sign convention is $p_{s,t,q} = (\theta_0 - \theta_1 - \phi_{s,t,q}) / x_q$,
    consistent with `n.pf()` and `n.lpf()`. This is the negative of the angle
    convention used in Verboomen et al. (2008)[^4], i.e.
    $\phi^{\mathrm{PyPSA}} = -\phi^{\mathrm{Verboomen}}$.

## Linear Loss Approximation

!!! note "Background"

    The AC transmission losses $\psi_{l,t}$ for line $l$ at timestep $t$ are given by the loss parabola

    $$\psi_{l,t} = r_{l}\, p_{l,t}^2,$$

    where $r_l$ is the resistance `r_pu_eff` of line $l$, following Neumann et al. (2022)[^5].

In PyPSA this non-linear function is linearized with piecewise linear constraints. The loss $\psi_{l,s,t}$ is bounded by the loss at full rating (`r_pu_eff * (s_max_pu * s_nom_max)**2`, with `s_nom` for non-extendable branches) and from below by a set of cuts $\epsilon$, one for each direction of flow:

<!-- md:spec Line_loss Line_loss_upper Line_loss_tangents_pos Line_loss_tangents_neg -->

Each cut $\epsilon$ is a line of the linear approximation with slope `Line_loss_slope` and offset `Line_loss_offset`. Transformers have the same constraints (`Transformer_loss_*`).

The losses also modify the [power balance](energy-balance.md), splitting losses equally between both connection points:

<!-- md:spec Line_injection -->

The [dispatch limits](dispatch-limits.md) of $p_{l,s,t}$ are now subtracted by $\psi_{l,s,t}$.


Two linearization methods are available in PyPSA: the tangent-based approximation and the secant-based approximation with error control.

The transmission loss approximation is not activated by default, but must be enabled in [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__].

```py
# Secant-based approximation with error control
n.optimize(transmission_losses=True)
# Or with custom tolerances
n.optimize(transmission_losses={"mode": "secants", "atol": 1, "rtol": 0.1})
# Tangent-based approximation
n.optimize(transmission_losses={"mode": "tangents", "segments": 3})
```

When passing a dict, the `mode` key selects the method. Additional keys set the options of that method:

- **Secant mode** (`"secants"`): `atol`, `rtol`, `max_segments`.
- **Tangent mode** (`"tangents"`): `segments`.

!!! hint "Hint: Calculating transmission losses"

    The losses can be calculated with `n.lines_t.p0 + n.lines_t.p1`.

??? note "Symbols"

    <!-- md:spec-legend Kirchhoff_Voltage_Law Line_angle_sum Transformer_angle_sum Transformer_phase_shift Line_loss_upper Line_loss_tangents_pos Line_injection -->

### Secant-Based Linearization with Error Control

The secant-based linearization is designed to overestimate the AC line losses and is guaranteed to stay within the error bounds defined by the specified absolute (`atol`) and relative (`rtol`) error tolerances. The slope and offset of each cut $\epsilon$ are derived as

!!! note "Background"

    $$\mathrm{Line\_loss\_slope}_{l,s,\epsilon,t} = r_l\, (\rho_{l,\epsilon-1} + \rho_{l,\epsilon})$$

    $$\mathrm{Line\_loss\_offset}_{l,s,\epsilon,t} = - r_l\, \rho_{l,\epsilon-1} \rho_{l,\epsilon}$$

where $\rho_{l,\epsilon}$ are suitably chosen points on the loss parabola. For details on how these points are derived see the [original PR](https://github.com/PyPSA/PyPSA/pull/1495).

### Tangent-Based Linearization

!!! hint "Warning: Depending on parameters, some lines can be lossless when the tangent-based linearization is used"
    - The tangent-based approximation is designed to underestimate losses
    - Its segments depend on `s_nom_max`
    - The first segment models zero losses
    - If a line has `s_nom_max > 2 * n_segments * s_nom_opt` then that line is lossless, since all flows stay in the first segment of the approximation.

For the tangent-based linearization the slope and offset of cut $\epsilon$ out of $n_{\mathrm{seg}}$ cuts are derived as:

!!! note "Background"

    $$\psi_{l,s,t}(\epsilon) = r_{l} \left(\frac{\epsilon}{n_{\mathrm{seg}}} \cdot \bar{p}_{l,s,t} \bar{P}_{l} \right)^2$$

    $$\mathrm{Line\_loss\_slope}_{l,s,\epsilon,t} = 2\, r_{l} \left(\frac{\epsilon}{n_{\mathrm{seg}}} \cdot \bar{p}_{l,s,t} \bar{P}_{l} \right)$$

    $$\mathrm{Line\_loss\_offset}_{l,s,\epsilon,t} = \psi_{l,s,t}(\epsilon) - \mathrm{Line\_loss\_slope}_{l,s,\epsilon,t} \left(\frac{\epsilon}{n_{\mathrm{seg}}} \cdot \bar{p}_{l,s,t} \bar{P}_{l} \right)$$

    where $\bar{P}_l$ is `s_nom_max` for extendable and `s_nom` for non-extendable branches.

The higher the number of tangents, the more accurate the approximation, but also the more constraints are added to the optimisation problem. Typically, 2-4 tangents are sufficient for a reasonably accurate approximation.


More details on this implementation can be found in Neumann et al. (2022)[^5].

## Examples


<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Meshed AC-DC Networks**

    ---

    Builds a stylized 3-node AC network coupled via AC-DC converters to a 3-node DC network.

    [:octicons-arrow-right-24: Go to example](../../examples/ac-dc-lopf.ipynb)


-   :material-notebook:{ .lg .middle } **Negative LMPs from Line Congestion**

    ---

    Explores how Kirchhoff's Voltage Law can lead to negative locational marginal prices when lines are congested.

    [:octicons-arrow-right-24: Go to example](../../examples/negative-prices-kvl-baker.ipynb)

</div>

[^1]: J. Hörsch, H. Ronellenfitsch, D. Witthaut, T. Brown (2018), [Linear optimal power flow using cycle flows](https://www.sciencedirect.com/science/article/abs/pii/S0378779617305138), Electric Power Systems Research, 158, 126-135, doi:10.1016/j.epsr.2017.12.034.

[^2]: N. Deo, G. M. Prabhu, M. S. Krishnamoorthy (1982), [Algorithms for Generating Fundamental Cycles in a Graph](https://doi.org/10.1145/355984.355988), ACM Transactions on Mathematical Software, 8 (1), 26-42, doi:10.1145/355984.355988.

[^3]: S. Hagspiel, C. Jägemann, D. Lindenberger, T. Brown, S. Cherevatskiy, E. Tröster (2014), [Cost-optimal power system extension under flow-based market coupling](https://doi.org/10.1016/j.energy.2014.01.025), Energy, 66, 654-666, doi:10.1016/j.energy.2014.01.025.

[^4]: J. Verboomen, D. Van Hertem, P. H. Schavemaker, W. L. Kling, R. Belmans (2008), [Analytical Approach to Grid Operation With Phase Shifting Transformers](https://doi.org/10.1109/TPWRS.2007.913197), IEEE Transactions on Power Systems, 23 (1), 41-46, doi:10.1109/TPWRS.2007.913197. See Section III-A "Linearized Power Flow", eq. (3).

[^5]: F. Neumann, V. Hagenmeyer, T. Brown (2022), [Assessments of linear power flow and transmission loss approximations in coordinated capacity expansion problems](https://doi.org/10.1016/j.apenergy.2022.118859), Applied Energy, 314, 118859, doi:10.1016/j.apenergy.2022.118859.
