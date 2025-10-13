<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Contingencies

Contingency analysis is concerned with the behaviour of the power system after
contingencies, such as the outage of particular branches (i.e. transformers and
lines). Only branch outages are considered here.

!!! tip "Tip: Modelling of generator outages"

    Generator outages can be modelled through stochastic optimisation, by setting the attribute `p_max_pu` to zero for a scenario-dependent outage period. See <!-- md:guide optimization/stochastic.md --> for more details.

## N-1 Security Constraints for SCLOPF

The Security-Constrained Linear Optimal Power Flow (SCLOPF) builds on the Linear
Optimal Power Flow (LOPF) (part of [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__]) by including
additional constraints that branches may not become overloaded after the outage
of a selection of branches. The implementation follows Ronellenfitsch et al.
(2017)[^1].

An optimisation with such $N-1$ security-constrained power flow constraints is
executed with
[`n.optimize.optimize_security_constrained()`][pypsa.optimization.OptimizationAccessor.optimize_security_constrained],
which ensures that the optimised system is robust against the outage of a
selection of branches. A selection of branches for which the outage is
considered can be made by passing the keyword argument `branch_outages` (e.g.
`branch_outages=n.lines[:10]` for considering outages of the first 10 lines). If
no selection is made, all branches are considered.

The effect of an outage of a branch $c$ on the flows of all other branches $b$
in a sub-network is captured by the Branch Outage Distribution Factor (BODF)
matrix, which is built from linear power flow assumptions. For the outage of
branch $c$, let $p_b$ be the flows before the outage and $p_b^{(c)}$ be the
flows after the outage. Then, the BODF is defined by:

$$p_b^{(c)} = p_b + BODF_{bc}p_{c}$$

To ensure that the flows of all other branches $b$ do not exceed their
capacity $P_b$ after the outage of branch $c$, the following constraints are
added to the optimisation problem:

$$|p_{b,t} + BODF_{bc}p_{c,t}| \leq |P_b| \quad \forall b,c,t$$

These constraints are added in the function [`n.optimize_security_constrained()`][pypsa.optimization.OptimizationAccessor.optimize_security_constrained].

!!! tip "Tip: Avoiding the computational burden of $N-1$ security constraints"

    Running security-constrained optimisation problems can be computationally expensive due to the product of lines, outages and snapshots.
    Several approaches have been proposed to approximate the $N-1$ security constraints,
    the simplest of which consists of reserving a security margin
    from the branch capacity for contingencies (e.g. by setting `s_max_pu=0.7` to prevent line loading above 70% of the branches' rated capacity).
    See, for example, the discussion in Gazafroudi et al. (2022)[^4].

??? note "Mapping of symbols to component attributes"

    The following table maps the symbols used in this section to the component attributes in PyPSA:

    | Symbol | Attribute | Type |
    |--------|---------------------|------|
    | $p_{\{b,c\},t}$ | `n.{lines,transformers}_t.p0` | Decision Variable |
    | $P_b$ | For extendable components `n.{lines,transformers}.eval("s_max_pu * s_nom_opt")` | Decision Variable |
    | $P_b$ | For non-extendable components`n.{lines,transformers}.eval("s_max_pu * s_nom")` | Parameter |
    | $BODF_{bc}$ | Calculated by [`sn.calculate_BODF()`][pypsa.networks.SubNetwork.calculate_BODF] | Parameter |

## Calculating Branch Outage Distribution Factors (BODF)

The BODF is calculated by the function
[`sn.calculate_BODF()`][pypsa.SubNetwork.calculate_BODF].
It is determined from the Power Transfer Distribution Factors (PTDF) matrix
and the incidence matrix $K$ of the network.[^2] [^3]

The first step consists of building the branch $BPTDF$
from the $PTDF$ and incidence matrix $K$

$$BPTDF_{bc} = \sum_{i} PTDF_{bi} K_{ic}$$

$BPTDF_{bc}$ gives the change in flow on branch $b$ if a
unit of power is injected at `bus0` of branch $c$ and
withdrawn from the `bus1` of branch $c$. If branch $b$ is
the only branch connecting two regions, then $BPTDF_{bb} = 1$,
since the power can only flow between the two ends of the branch
through the branch itself.

The off-diagonal entries of the $BODF$ for $b \neq c$ are given by:

$$BODF_{bc} = \frac{BPTDF_{bc}}{1-BPTDF_{cc}}$$

The diagonal entries of the $BODF$ are simply:

$$BODF_{bb} = -1$$

!!! warning "Warning: Singular $BODF$ matrix"

    If $c$ is the only branch connecting two regions, so that the
    regions become disconnected after the outage of $c$, then
    $BPTDF_{cc} = 1$ and $BODF_{bc}$ becomes singular; this
    case must be treated separately since, for example, each region will
    need its own slack bus.

!!! note "Note: BODF versus LODF matrix"

    The *Branch Outage Distribution Factor (BODF)* matrix is also called *Line Outage Distribution
    Factor (LODF)* matrix in the literature, but in PyPSA both lines and transformers are included.
    This is where the more general name stems from.

## Examples


<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Security-Constrained LOPF**

    ---

    Implements N-1 security constraints in linear optimal power flow models to ensure grid reliability under line outage events.

    [:octicons-arrow-right-24: Go to example](../../examples/scigrid-sclopf.ipynb)

</div>

[^1]: H. Ronellenfitsch, D. Manik, J. Hörsch, T. Brown and D. Witthaut, [Dual Theory of Transmission Line Outages](https://doi.org/10.1109/TPWRS.2017.2658022), in IEEE Transactions on Power Systems, vol. 32, no. 5, pp. 4060-4068, doi:10.1109/TPWRS.2017.2658022.

[^2]: T. Guler, G. Gross and M. Liu (2007), [Generalized Line Outage Distribution Factors](https://doi.org/10.1109/TPWRS.2006.888950), in IEEE Transactions on Power Systems, vol. 22, no. 2, pp. 879-881, doi:10.1109/TPWRS.2006.888950.

[^3]: J. Guo, Y. Fu, Z. Li and M. Shahidehpour (2009), [Direct Calculation of Line Outage Distribution Factors](https://doi.org/10.1109/TPWRS.2009.2023273), in IEEE Transactions on Power Systems, vol. 24, no. 3, pp. 1633-1634, doi:10.1109/TPWRS.2009.2023273.

[^4]: A. S. Gazafroudi, F. Neumann, T. Brown, [Topology-based approximations for N−1 contingency constraints in power transmission networks](https://doi.org/10.1016/j.ijepes.2021.107702), International Journal of Electrical Power & Energy Systems, 137, 107702, doi:10.1016/j.ijepes.2021.107702.