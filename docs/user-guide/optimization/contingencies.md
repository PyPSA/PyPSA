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

The effect of an outage $\zeta$ of a branch on the flows of all other branches $l$
in a sub-network is captured by the Branch Outage Distribution Factor (BODF)
matrix, which is built from linear power flow assumptions. Let $p_l$ be the
flows before the outage and $p^{(\zeta)}_l$ be the flows after the outage. Then,
the BODF is defined by:

!!! note "Background"

    $$p^{(\zeta)}_{l} = p_{l} + \mathrm{BODF}_{l,\zeta}\, p_{\zeta}$$

To ensure that the flows of all other branches $l$ do not exceed their
capacity $\bar{p}_{l,t} \hat{p}_l$ (or $\bar{p}_{l,t} P_l$ if extendable) after the outage $\zeta$, the following constraints are
added to the optimisation problem. The flow $\mathit{Outage\_s}_{\zeta,s,t}$ is the flow on the branch that the outage $\zeta$ takes out. The [trait `passive_branch`](formulation/traits/passive_branch.md) states them for both lines and transformers, for fixed and extendable capacities:

<!-- md:spec Line_fix_s_upper_security Line_fix_s_lower_security Line_ext_s_upper_security Line_ext_s_lower_security Transformer_fix_s_upper_security Transformer_fix_s_lower_security Transformer_ext_s_upper_security Transformer_ext_s_lower_security -->

These constraints are added in the function [`n.optimize_security_constrained()`][pypsa.optimization.OptimizationAccessor.optimize_security_constrained].

With `multi_investment_periods=True`, the BODF matrix is computed from the network topology of each investment period. The constraints are added per investment period and sub-network, with the period as name suffix (e.g. `Line-fix-s-upper-security-for-Line-outage-in-sub-network-0-period-2030`). They only include the snapshots of that period and the branches active in it.

Explicitly requested outages of branches that are not active in any optimized period raise an error. This includes an outage of a branch that is only active in periods outside the passed `snapshots`, which matters when you write your own rolling-horizon loop. With the default `branch_outages=None`, inactive branches are skipped.

!!! tip "Tip: Avoiding the computational burden of $N-1$ security constraints"

    Running security-constrained optimisation problems can be computationally expensive due to the product of lines, outages and snapshots.
    Several approaches have been proposed to approximate the $N-1$ security constraints,
    the simplest of which consists of reserving a security margin
    from the branch capacity for contingencies (e.g. by setting `s_max_pu=0.7` to prevent line loading above 70% of the branches' rated capacity).
    See, for example, the discussion in Gazafroudi et al. (2022)[^4].

??? note "Symbols"

    <!-- md:spec-legend Line_fix_s_upper_security Line_ext_s_upper_security Transformer_fix_s_upper_security Transformer_ext_s_upper_security Outage_s -->

    The BODF matrix is calculated by [`sn.calculate_BODF()`][pypsa.networks.SubNetwork.calculate_BODF].

## Calculating Branch Outage Distribution Factors (BODF)

The BODF is calculated by the function
[`sn.calculate_BODF()`][pypsa.SubNetwork.calculate_BODF].
It is determined from the Power Transfer Distribution Factors (PTDF) matrix
and the incidence matrix $K$ of the network.[^2] [^3]

!!! note "Background"

    The first step consists of building the branch $\mathrm{BPTDF}$
    from the $\mathrm{PTDF}$ and incidence matrix $K$:

    $$\mathrm{BPTDF}_{l,\zeta} = \sum_{n \in \mathcal{N}} \mathrm{PTDF}_{l,n} K_{n,\zeta}$$

    $\mathrm{BPTDF}_{l,\zeta}$ gives the change in flow on branch $l$ if a
    unit of power is injected at `bus0` of the branch of outage $\zeta$ and
    withdrawn from its `bus1`. If branch $l$ is
    the only branch connecting two regions, then $\mathrm{BPTDF}_{l,l} = 1$,
    since the power can only flow between the two ends of the branch
    through the branch itself.

    The off-diagonal entries of the $\mathrm{BODF}$ for $l \neq \zeta$ are given by:

    $$\mathrm{BODF}_{l,\zeta} = \frac{\mathrm{BPTDF}_{l,\zeta}}{1-\mathrm{BPTDF}_{\zeta,\zeta}}$$

    The diagonal entries of the $\mathrm{BODF}$ are simply:

    $$\mathrm{BODF}_{\zeta,\zeta} = -1$$

!!! warning "Warning: Singular BODF matrix"

    If the branch of outage $\zeta$ is the only branch connecting two regions, so that the
    regions become disconnected after the outage, then
    $\mathrm{BPTDF}_{\zeta,\zeta} = 1$ and $\mathrm{BODF}_{l,\zeta}$ becomes singular; this
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
