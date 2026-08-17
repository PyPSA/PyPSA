<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Linear Power Flow

The linear power flow [`n.lpf()`][pypsa.Network.lpf] is the linearised equivalent of [`n.pf()`][pypsa.Network.pf].

## AC networks

For AC networks, it is assumed for the linear power flow that reactive power decouples, there are no voltage magnitude variations, voltage angles differences across branches are small and branch resistances are much smaller than branch reactances.

For AC networks, the linear power flow is calculated using small voltage angle differences and the series reactances alone.

It is assumed that the active powers $P_i$ are given for all buses except the slack bus and the task is to find the voltage angles $\theta_i$ at all buses except the slack bus, where it is assumed $\theta_0 = 0$.

To find the voltage angles, the following linear set of equations are solved

$$P_i = \sum_j (KBK^T)_{ij} \theta_j - \sum_l K_{il} b_l \theta_l^{\textrm{shift}}$$

where $K$ is the incidence matrix of the network, $B$ is the diagonal matrix of inverse branch series reactances $x_l$ multiplied by the tap ratio $\tau_l$, i.e. $B_{ll} = b_l = \frac{1}{x_l\tau_l}$ and $\theta_l^{\textrm{shift}}$ is the phase shift for a transformer. The matrix $KBK^T$ is singular with a single zero eigenvalue for a connected network, therefore the row and column corresponding to the slack bus is deleted before inverting.

The flows `p0` in the network branches at `bus0` can then be found by multiplying by the transpose incidence matrix and inverse series reactances:

$$F_l = \sum_i (BK^T)_{li} \theta_i - b_l \theta_l^{\textrm{shift}}$$

## DC networks

For DC networks, it is assumed for the linear power flow that voltage magnitude differences across branches are all small.

For DC networks, the linear load flow is calculated using small voltage magnitude differences and series resistances alone.

The linear load flow for DC networks follows the same calculation as for AC networks, but replacing the voltage angles by the difference in voltage magnitude $\delta V_{n,t}$ and the series reactance by the series resistance $r_l$.


## Linear Power Flow Contingency Analysis

The function [n.lpf_contingency()][pypsa.Network.lpf_contingency] computes a
base case linear power flow (LPF) with no outages, and then cycles through the
list of branches in `branch_outages` and computes the line flows after the
outage of that branch using the branch outage distribution factor (BODF, see [Contingencies](./optimization/contingencies.md#calculating-branch-outage-distribution-factors-bodf)).


## Inputs

For the linear power flow, the following data for each component are used. For the defaults and units, see <!-- md:guide components.md -->.

- `n.buses.{v_nom}`
- `n.loads.{p_set}`
- `n.generators.{p_set}`
- `n.storage_units.{p_set}`
- `n.stores.{p_set}`
- `n.shunt_impedances.{g}`
- `n.lines.{x}`
- `n.transformers.{x}`
- `n.links.{p_set}`

!!! note

    Note that for lines and transformers you must make sure that $x$ is non-zero, otherwise the bus admittance matrix will be singular.

## Outputs

- `n.buses.{v_mag_pu, v_ang, p}`
- `n.loads.{p}`
- `n.generators.{p}`
- `n.storage_units.{p}`
- `n.stores.{p}`
- `n.shunt_impedances.{p}`
- `n.lines.{p0, p1}`
- `n.transformers.{p0, p1}`
- `n.links.{p0, p1}`


## Examples

<div class="grid cards" markdown>


-   :material-notebook:{ .lg .middle } **Newton-Raphson Power Flow**

    ---

    Solves non-linear AC power flow equations using the Newton-Raphson method to inspect voltage magnitudes and angles.

    [:octicons-arrow-right-24: Go to example](../examples/minimal-example-pf.ipynb)

-   :material-notebook:{ .lg .middle } **SciGRID Network**

    ---

    Performs linear optimal power flow on a high-resolution German grid model to analyze power flows and nodal prices.

    [:octicons-arrow-right-24: Go to example](../examples/scigrid-lopf-then-pf.ipynb)

</div>
