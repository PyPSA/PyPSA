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
power flow $p_{l,t}$ in AC networks is governed by the cycle-based linearised formulation
of Kirchhoff's Voltage Law (KVL)

$$\sum_l C_{l,c} x_l p_{l,t} = 0  \quad \forall\, c,t$$

where  $C$ is a [cycle basis
matrix](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.cycle_basis.html)
of the network graph, where the independent cycles $c$ are expressed as directed
linear combinations of lines $l$, and $x_l$ is the series reactance.

While there are different formulations of KVL, the cycle-based formulation was
found to be substantially faster than other formulations due to its sparsity, as
shown in Hörsch et al. (2018)[^1]. This formulation defines the same feasible
space as other standard linearised formulations based on voltage angles that is
commonly found in textbooks (B-Theta) or the formulation based on Power Transfer
Distribution Factors (PTDFs).

These constraints are set in the function `define_kirchhoff_voltage_constraints()` and carry the name `Kirchhoff-Voltage-Law`.

!!! tip "Tip: KVL with DC networks"

    For DC networks, replace the series reactance $x_l$ by the series resistance $r_l$.

!!! note "Note: Retrieving the PTDF matrix"

    The PTDF matrix of sub-networks (i.e. synchronous zones) can be calculated with [`sn.calculate_PTDF()`][pypsa.networks.SubNetwork.calculate_PTDF] method.

!!! tip "Tip: Using the [`Link`][pypsa.components.Links] component for Net Transfer Capacities (NTCs)"

    For simplified transmission representation using Net Transfer Capacities (NTCs), use the [`Link`][pypsa.components.Links] component with controllable power flow like a transport model. The [`Link`][pypsa.components.Links] component can also be used to represent a point-to-point HVDC link.

!!! note "Note: Handling impedance changes with transmission expansion"

    If $F_l$ is also subject to optimisation (`s_nom_extendable=True`), the impedance $x$ of the line is **not** automatically changed with the capacity (e.g. to represent added parallel lines).
    However, the extension [`n.optimize.optimize_transmission_expansion_iteratively()`][pypsa.optimization.OptimizationAccessor.optimize_transmission_expansion_iteratively] covers this through an
    iterative process as done Hagspiel et al. (2014)[^2] .

## Linear Loss Approximation

The AC transmission losses $\psi_{l,t}$ for line $l$ at timestep $t$ are given by the loss parabola

$$
\psi_{l,t} = r_{l} p_{l,t}^2,
$$

where $r_l$ is the resistance of line $l$, following Neumann et al. (2022)[^3].

In PyPSA this non-linear function is linearized with piecewise linear constraints:

$$
0 \leq \psi_{l,t} \leq r_{l} (\bar{p}_{l,t} \overline{P}_{l})^2 \quad \forall l,t
$$

$$
\psi_{l,t} \geq m_k \cdot p_{l,t} + a_k \quad \forall l,t,\ k = 1, \dots, n
$$

$$
\psi_{l,t} \geq -m_k \cdot p_{l,t} + a_k \quad \forall l,t,\ k = 1, \dots, n
$$

where $\bar{p}_{l,t}$ is the maximum per-unit powerflow at time $t$ and $\overline{P}_{l}^2$ is the (maximum) line capacity. The index $k$ denotes the segments of the linear approximation with a total of $n$ segments, with offsets $a_k$ and with slopes $m_k$.

The losses also modify the [power balance](energy-balance.md) by adding the following term to its left-hand side:

$$
-0.5 \cdot \sum_{l} |K_{n,l}| \cdot \psi_{l,t} \quad \forall n,t
$$

splitting losses equally between both connection points.

The dispatch limits of $p_{\ell,t}$ are now subtracted by $\psi_{l,t}$.


Two linearization methods are available in PyPSA: the tangent-based approximation ([`define_tangent_loss_constraints()`][pypsa.optimization.constraints.define_tangent_loss_constraints]) and the secant-based approximation with error control ([`define_secant_loss_constraints()`][pypsa.optimization.constraints.define_secant_loss_constraints]).

The transmission loss approximation is not activated by default, but must be enabled in [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__].

```py
# Secant-based approximation with error control
n.optimize(transmission_losses=True)
# Or with custom tolerances
n.optimize(transmission_losses={"mode": "secants", "atol": 1, "rtol": 0.1})
# Tangent-based approximation
n.optimize(transmission_losses={"mode": "tangents", "segments": 3})
```

When passing a dict, the `mode` key selects the method. Additional keys are forwarded to the corresponding constraint function:

- **Secant mode** (`"secants"`, see [`define_secant_loss_constraints()`][pypsa.optimization.constraints.define_secant_loss_constraints]): `atol`, `rtol`, `max_segments`.
- **Tangent mode** (`"tangents"`, see [`define_tangent_loss_constraints()`][pypsa.optimization.constraints.define_tangent_loss_constraints]): `segments`.

!!! hint "Hint: Calculating transmission losses"

    The losses can be calculated with `n.lines_t.p0 + n.lines_t.p1`.

??? note "Mapping of symbols to component attributes"

| Symbol            | Attribute                                                                                                                                   | Type                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| $p_{l,t}$       | `n.lines_t.p0` or `n.transformers_t.p0`                                                                                                 | Decision variable             |
| $\psi_{l,t}$    | `n.lines_t.losses` or `n.transformers_t.losses`                                                                                         | Decision variable             |
| $\bar{p}_{l,t}$ | `n.lines_t.s_max_pu` or `n.transformers_t.s_max_pu`                                                                                     | Parameter                     |
| $\bar{P}_{l}$   | `n.lines.s_nom_opt` and `n.transformers.s_nom_opt` (if extendable) or `n.lines.s_nom` or `n.transformers.s_nom` (if non-extendable) | Decision variable / Parameter |
| $x_l$           | `n.lines.x_pu_eff` or `n.transformers.x_pu_eff`                                                                                         | Parameter                     |
| $r_l$           | `n.lines.r_pu_eff` or `n.transformers.r_pu_eff`                                                                                         | Parameter                     |
| $C_{l,c}$       | Cycle matrix calculated by `find_cycles()`                                                                                                | Parameter                     |
| $K_{n,l}$       | Incidence matrix calculated by [`n.incidence_matrix()`][pypsa.networks.SubNetwork.incidence_matrix]                                       | Parameter                     |

### Secant-Based Linearization with Error Control

The secant-based linearization is designed to overestimate the AC line losses and is guaranteed to stay within the error bounds defined by the specified absolute (`atol`) and relative (`rtol`) error tolerances. The slope $m_k$ and offset $a_k$ are derived as

$$
m_{k, l} = r_l (\rho_{k-1} + \rho_{k})
$$

$$
a_{k, l} = - r_l \rho_{k-1} \rho_{k}
$$

where $\rho_k$ are suitably chosen points on the loss parabola. For details on how these points are derived see the [original PR](https://github.com/PyPSA/PyPSA/pull/1495).

### Tangent-Based Linearization

!!! hint "Warning: Depending on parameters, some lines can be lossless when the tangent-based linearization is used"
    - The tangent-based approximation is designed to underestimate losses
    - Its segments depend on `s_nom_max`
    - The first segment models zero losses
    - If a line has `s_nom_max > 2 * n_segments * s_nom_opt` then that line is lossless, since all flows stay in the first segment of the approximation.

For the tangent-based linearization the slope $m_k$ and offset $a_k$ are derived as:

$$
\psi_{l,t}(k) = r_{l} \left(\frac{k}{n} \cdot \bar{p}_{l,t} \overline{P}_{l} \right)^2
$$

$$
m_k = \frac{d \psi_{l,t}(k)}{dk} = 2 r_{l} \left(\frac{k}{n} \cdot \bar{p}_{l,t} \overline{P}_{l} \right)
$$

$$
a_k = \psi_{l,t}(k) - m_k \left(\frac{k}{n} \cdot \bar{p}_{l,t} \overline{P}_{l} \right)
$$

The higher the number of tangents, the more accurate the approximation, but also the more constraints are added to the optimisation problem. Typically, 2-4 tangents are sufficient for a reasonably accurate approximation.


More details on this implementation can be found in Neumann et al. (2022)[^3].

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

[^2]: S. Hagspiel, C. Jägemann, D. Lindenberger, T. Brown, S. Cherevatskiy, E. Tröster (2014), [Cost-optimal power system extension under flow-based market coupling](https://doi.org/10.1016/j.energy.2014.01.025), Energy, 66, 654-666, doi:10.1016/j.energy.2014.01.025.

[^3]: F. Neumann, V. Hagenmeyer, T. Brown (2022), [Assessments of linear power flow and transmission loss approximations in coordinated capacity expansion problems](https://doi.org/10.1016/j.apenergy.2022.118859), Applied Energy, 314, 118859, doi:10.1016/j.apenergy.2022.118859.
