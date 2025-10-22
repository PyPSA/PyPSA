<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Unit Commitment

Unit commitment constraints are implemented for the [`Generator`][pypsa.components.Generators] and [`Link`][pypsa.components.Links]
components. They are used to model the start-up and shut-down constraints, as
well as ramping constraints. The implementation is based on Taylor (2015)[^1],
and is supplemented with work by Hua et al. (2017)[^2] for a tightened linear
relaxation.

## Start-Up and Shut-down

For components marked as `committable=True`, new binary status variables
$u_{*,t} \in \{0,1\}$ are introduced, which indicate whether the component is
running (1) or not (0) in period $t$. This turns the model into a mixed-integer
linear programme (MILP). The restrictions on the dispatch now enforce that the
dispatch $g_{n,s,t}$ or $f_{l,t}$ is only non-zero if the component is running,
i.e. $u_{*,t} = 1$:

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $g_{n,s,t} \geq u_{n,s,t} \cdot \underline{g}_{n,s,t} \cdot \hat{g}_{n,s}$ | `Generator-com-p-lower` |
    | $g_{n,s,t} \leq u_{n,s,t} \cdot \bar{g}_{n,s,t} \cdot \hat{g}_{n,s}$ | `Generator-com-p-upper` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $f_{l,t} \geq u_{l,t} \cdot \underline{f}_{l,t} \cdot \hat{f}_{l}$ | `Link-com-p-lower` |
    | $f_{l,t} \leq u_{l,t} \cdot \bar{f}_{l,t} \cdot \hat{f}_{l}$ | `Link-com-p-upper` |

!!! warning

    Note that components cannot be both extendable (`p_nom_extendable=True`) and committable (`committable=True`) because of the non-linear coupling of status and capacity.

If the **minimum up time** $T_{\textrm{min_up}}$ is set, status switches are constrained to ensure
that the component is running for at least $T_{\textrm{min_up}}$ snapshots after it has been started up:

=== "Generator"

    Constraint `Generator-com-up-time`:

    $$\sum_{t'=t}^{t+T_\textrm{min_up}} u_{n,s,t'}\geq T_\textrm{min_up} (u_{n,s,t} - u_{n,s,t-1})$$

=== "Link"

    Constraint `Link-com-up-time`:

    $$\sum_{t'=t}^{t+T_\textrm{min_up}} u_{l,t'}\geq T_\textrm{min_up} (u_{l,t} - u_{l,t-1})$$

The component may have been up for some periods before the optimisation period (`n.optimize(snapshots=snapshots)`). If the up-time before `snapshots` starts is less than the minimum up-time, the component is forced remain up for the difference at the start of `snapshots`. If the start of `snapshots` is the start of `n.snapshots`, the up-time before the simulation is read from the input attribute `up_time_before`. If `snapshots` falls in the middle of `n.snapshots`, then the statuses before `snapshots` are assumed to be set by previous runs. If the start of `snapshots` is very close to the start of `n.snapshots`, it will also take account of `up_time_before` as well as the statuses in between.

At the end of `snapshots` the minimum up-time in the constraint is only enforced for the remaining snapshots, if the number of remaining snapshots is less than $T_{\textrm{min_up}}$.

If the **minimum down time** $T_{\textrm{min_down}}$ is set, status switches are constrained to ensure that the component is not running for at least $T_{\textrm{min_down}}$ snapshots after it has been shut down:

=== "Generator"

    Constraint `Generator-com-down-time`:

    $$\sum_{t'=t}^{t+T_\textrm{min_down}} (1-u_{n,s,t'})\geq T_\textrm{min_down} (u_{n,s,t-1} - u_{n,s,t})$$

=== "Link"

    Constraint `Link-com-down-time`:

    $$\sum_{t'=t}^{t+T_\textrm{min_down}} (1-u_{l,t'})\geq T_\textrm{min_down} (u_{l,t-1} - u_{l,t})$$

The component may have been down for some periods before the optimisation period (`n.optimize(snapshots=snapshots)`). If the down-time before `snapshots` starts is less than the minimum down-time, the component is forced to remain down for the difference at the start of `snapshots`. If the start of `snapshots` is the start of `n.snapshots`, the down-time before the simulation is read from the input attribute `down_time_before`. If `snapshots` falls in the middle of `n.snapshots`, then the statuses before `snapshots` are assumed to be set by previous runs. If the start of `snapshots` is very close to the start of `n.snapshots`, it will also take account of `down_time_before` as well as the statuses in between.

Furthermore, two **state transition variables** for start-up ($su_{*,t} \in \{0,1\}$) and shut-down ($sd_{*,t} \in$ \{0,1\}) are introduced to associate them with start-up and shut-down cost terms in the objective function. The constraints are set so that the start-up variable is only non-zero if the component has just started up, i.e. $u_{n,s,t} - u_{n,s,t-1} = 1$, and the shut-down variable is only non-zero if the component has just shut down, i.e. $u_{n,s,t-1} - u_{n,s,t} = 1$:

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $su_{n,s,t} \geq u_{n,s,t} - u_{n,s,t-1}$ | `Generator-com-transition-start-up` |
    | $sd_{n,s,t} \geq u_{n,s,t-1} - u_{n,s,t}$ | `Generator-com-transition-shut-down` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $su_{l,t} \geq u_{l,t} - u_{l,t-1}$ | `Link-com-transition-start-up` |
    | $sd_{l,t} \geq u_{l,t-1} - u_{l,t}$ | `Link-com-transition-shut-down` |



These constraints are defined in the function `define_operational_constraints_for_committables()`.

??? note "Mapping of symbols to component attributes"

    === "Generator"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $g_{n,s,t}$       | `n.generators_t.p` | Decision variable |
        | $u_{n,s,t}$       | `n.generators_t.status` | Decision variable |
        | $su_{n,s,t}$      | `n.generators_t.start_up` | Decision variable |
        | $sd_{n,s,t}$      | `n.generators_t.shut_down` | Decision variable |
        | $\hat{g}_{n,s}$   | `n.generators.p_nom` | Parameter |
        | $\underline{g}_{n,s,t}$ | `n.generators_t.p_min_pu` | Parameter |
        | $\bar{g}_{n,s,t}$ | `n.generators_t.p_max_pu` | Parameter |
        | $T_{\textrm{min_up}}$  | `n.generators.min_up_time` | Parameter |
        | $T_{\textrm{min_down}}$| `n.generators.min_down_time`| Parameter |
        | $T_{\textrm{up_time_before}}$ | `n.generators.up_time_before` | Parameter |
        | $T_{\textrm{down_time_before}}$ | `n.generators.down_time_before` | Parameter |

    === "Link"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $f_{l,t}$         | `n.links_t.p`  | Decision variable |
        | $u_{l,t}$         | `n.links_t.status`  | Decision variable |
        | $su_{l,t}$        | `n.links_t.start_up`  | Decision variable |
        | $sd_{l,t}$        | `n.links_t.shut_down`  | Decision variable |
        | $\hat{f}_{l}$     | `n.links.p_nom`  | Parameter |
        | $\underline{f}_{l,t}$| `n.links_t.p_min_pu`  | Parameter |
        | $\bar{f}_{l,t}$   | `n.links_t.p_max_pu`  | Parameter |
        | $T_{\textrm{min_up}}$  | `n.links.min_up_time`  | Parameter |
        | $T_{\textrm{min_down}}$| `n.links.min_down_time` | Parameter |
        | $T_{\textrm{up_time_before}}$ | `n.links.up_time_before` | Parameter |
        | $T_{\textrm{down_time_before}}$ | `n.links.down_time_before` | Parameter |

## Ramping

Ramp rate limits can be defined for increasing output $ru_{n,s,t}$ and decreasing output $rd_{n,s,t}$. By default these are null and ignored. They must be provided per unit of the nominal capacity.

!!! note

    When provided, ramping limits are also considered if they are not
    committable (`committable=False`), i.e. the component is not subject to start-up and shut-down constraints.

For **non-extendable** but **non-committable** components, the dispatch obeys for $t \in \{1,\dots |T|-1\}$:

=== "Generator"

    | Constraint | Dual Variable | Name |
    |-------------------|------------------|------------------|
    | $(g_{n,s,t} - g_{n,s,t-1}) \geq -rd_{n,s,t} \cdot \hat{g}_{n,s}$ | `n.generators_t.mu_ramp_limit_down` | `Generator-fix-p-ramp_limit_down` |
    | $(g_{n,s,t} - g_{n,s,t-1}) \leq ru_{n,s,t} \cdot \hat{g}_{n,s}$ | `n.generators_t.mu_ramp_limit_up` | `Generator-fix-p-ramp_limit_up` |

=== "Link"

    | Constraint | Dual Variable | Name |
    |-------------------|------------------|------------------|
    | $(f_{l,t} - f_{l,t-1}) \geq -rd_{l,t} \cdot \hat{f}_{l}$ | `n.links_t.mu_ramp_limit_down` | `Link-fix-p-ramp_limit_down` |
    | $(f_{l,t} - f_{l,t-1}) \leq ru_{l,t} \cdot \hat{f}_{l}$ | `n.links_t.mu_ramp_limit_up` | `Link-fix-p-ramp_limit_up` |



For **extendable** and **non-committable** components, the dispatch obeys for $t \in \{1,\dots |T|-1\}$:

=== "Generator"

    | Constraint | Dual Variable | Name |
    |-------------------|------------------|------------------|
    | $(g_{n,s,t} - g_{n,s,t-1}) \geq -rd_{n,s,t} \cdot G_{n,s}$ | `n.generators_t.mu_ramp_limit_down` | `Generator-ext-p-ramp_limit_down` |
    | $(g_{n,s,t} - g_{n,s,t-1}) \leq ru_{n,s,t} \cdot G_{n,s}$ | `n.generators_t.mu_ramp_limit_up` | `Generator-ext-p-ramp_limit_up` |

=== "Link"

    | Constraint | Dual Variable | Name |
    |-------------------|------------------|------------------|
    | $(f_{l,t} - f_{l,t-1}) \geq -rd_{l,t} \cdot F_{l}$ | `n.links_t.mu_ramp_limit_down` | `Link-ext-p-ramp_limit_down` |
    | $(f_{l,t} - f_{l,t-1}) \leq ru_{l,t} \cdot F_{l}$ | `n.links_t.mu_ramp_limit_up` | `Link-ext-p-ramp_limit_up` |


For **committable** and **non-extendable** components, additional ramp limits at start-up $rusu_{n,s}$ and shut-down $rdsd_{n,s}$ can be specified for $t \in \{1,\dots |T|-1\}$:

=== "Generator"

    | Constraint | Name |
    |-------------------|------------------|
    | $(g_{n,s,t} - g_{n,s,t-1}) \geq \left[ -rd_{n,s,t} \cdot u_{n,s,t} -rdsd_{n,s}(u_{n,s,t-1} - u_{n,s,t})\right] \hat{g}_{n,s}$ | `Generator-com-p-ramp_limit_down` |
    | $(g_{n,s,t} - g_{n,s,t-1}) \leq \left[ru_{n,s,t} \cdot u_{n,s,t-1} + rusu_{n,s} (u_{n,s,t} - u_{n,s,t-1})\right] \hat{g}_{n,s}$ | `Generator-com-p-ramp_limit_up` |

=== "Link"

    | Constraint | Name |
    |-------------------|------------------|
    | $(f_{l,t} - f_{l,t-1}) \geq \left[ -rd_{l,t} \cdot u_{l,t} -rdsd_{l,t}(u_{l,t-1} - u_{l,t})\right] \hat{f}_{l}$ | `Link-com-p-ramp_limit_down` |
    | $(f_{l,t} - f_{l,t-1}) \leq \left[ru_{l,t} \cdot u_{l,t-1} + rusu_{l} (u_{l,t} - u_{l,t-1})\right] \hat{f}_{l}$ | `Link-com-p-ramp_limit_up` |

These constraints are defined in the function `define_ramp_limit_constraints()`.

??? note "Mapping of symbols to component attributes"

    === "Generator"

        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $g_{n,s,t}$       | `n.generators_t.p` | Decision variable |
        | $G_{n,s}$         | `n.generators.p_nom_opt` | Decision variable |
        | $u_{n,s,t}$       | `n.generators_t.status` | Decision variable |
        | $\hat{g}_{n,s}$   | `n.generators.p_nom` | Parameter |
        | $\underline{g}_{n,s,t}$ | `n.generators_t.p_min_pu` | Parameter |
        | $\bar{g}_{n,s,t}$ | `n.generators_t.p_max_pu` | Parameter |
        | $ru_{n,s,t}$ | `n.generators_t.ramp_limit_up` | Parameter |
        | $rd_{n,s,t}$ | `n.generators_t.ramp_limit_down` | Parameter |
        | $rusu_{n,s}$     | `n.generators.ramp_limit_start_up` | Parameter |
        | $rdsd_{n,s}$     | `n.generators.ramp_limit_shut_down` | Parameter |

    === "Link"


        | Symbol | Attribute | Type |
        |-------------------|-----------|-------------|
        | $f_{l,t}$         | `n.links_t.p` | Decision variable |
        | $F_{l}$           | `n.links.p_nom_opt` | Decision variable |
        | $u_{l,t}$ | `n.links_t.status` | Decision variable |
        | $\hat{f}_{l}$     | `n.links.p_nom` | Parameter |
        | $\underline{f}_{l,t}$ | `n.links_t.p_min_pu` | Parameter |
        | $\bar{f}_{l,t}$   | `n.links_t.p_max_pu` | Parameter |
        | $ru_{l,t}$ | `n.links_t.ramp_limit_up` | Parameter |
        | $rd_{l,t}$ | `n.links_t.ramp_limit_down` | Parameter |
        | $rusu_{l}$     | `n.links.ramp_limit_start_up` | Parameter |
        | $rdsd_{l}$     | `n.links.ramp_limit_shut_down` | Parameter |

## Linearization

The implementation is based on Hua et al. (2017)[^2] and relaxes the binary unit commitment variables to continuous variables:

$$u_{*,t},\quad su_{*,t},\quad sd_{*,t}\quad  \in [0,1]$$

This allows for partial commitment states (generators can be partially on/off), making the problem more computationally tractable while loosing some accuracy. To enable this, use:

``` py
n.optimize(linearized_unit_commitment=True)
```

To tighten the relaxation, additional constraints are introduced that improve capturing the relationship between commitment status, ramping, and dispatch. This requires start up and shut down costs need to be equal. Otherwise the unit commitment variables are purely relaxed. The added constraints limit the dispatch during partial start-up and shut-down, as well as ramping during partial commitment:

=== "Generator"

    Constraint `Generator-com-p-before`:

    $$\begin{gather*}
    g_{n,s,t-1} \leq rdsd_{n,s} \hat{g}_{n,s} \cdot u_{n,s,t-1} + (\bar{g}_{n,s,t} \hat{g}_{n,s} - rdsd_{n,s}  \hat{g}_{n,s}) \cdot (u_{n,s,t} - su_{n,s,t})
    \end{gather*}$$

    Constraint `Generator-com-p-current`:

    $$\begin{gather*}
    g_{n,s,t} \leq \bar{g}_{n,s,t} \hat{g}_{n,s} \cdot u_{n,s,t} - (\bar{g}_{n,s,t} \hat{g}_{n,s} - rusu_{n,s} \hat{g}_{n,s}) \cdot su_{n,s,t}
    \end{gather*}$$

    Constraint `Generator-com-partly-start-up`:

    $$\begin{gather*}
    g_{n,s,t} - g_{n,s,t-1} \leq (\underline{g}_{n,s,t} \hat{g}_{n,s} + ru_{n,s,t} \hat{g}_{n,s}) \cdot u_{n,s,t} - \underline{g}_{n,s,t} \hat{g}_{n,s} \cdot u_{n,s,t-1} \\
    - (\underline{g}_{n,s,t} \hat{g}_{n,s} + ru_{n,s,t} \hat{g}_{n,s} - rusu_{n,s} \hat{g}_{n,s}) \cdot su_{n,s,t}
    \end{gather*}$$

    Constraint `Generator-com-partly-shut-down`:

    $$\begin{gather*}
    g_{n,s,t-1} - g_{n,s,t} \leq rdsd_{n,s} \hat{g}_{n,s} \cdot u_{n,s,t-1} - (rdsd_{n,s} \hat{g}_{n,s} - rd_{n,s,t} \hat{g}_{n,s}) \cdot u_{n,s,t} \\
    - (\underline{g}_{n,s,t} \hat{g}_{n,s} + rd_{n,s,t} \hat{g}_{n,s} - rdsd_{n,s} \hat{g}_{n,s}) \cdot su_{n,s,t}
    \end{gather*}$$

=== "Link"

    Constraint `Link-com-p-before`:

    $$\begin{gather*}
    f_{l,t-1} \leq rdsd_{l} \hat{f}_{l} \cdot u_{l,t-1} + (\bar{f}_{l,t} \hat{f}_{l} - rdsd_{l}  \hat{f}_{l}) \cdot (u_{l,t} - su_{l,t})
    \end{gather*}$$

    Constraint `Link-com-p-current`:

    $$\begin{gather*}
    f_{l,t} \leq \bar{f}_{l,t} \hat{f}_{l} \cdot u_{l,t} - (\bar{f}_{l,t} \hat{f}_{l} - rusu_{l} \hat{f}_{l}) \cdot su_{l,t}
    \end{gather*}$$

    Constraint `Link-com-partly-start-up`:

    $$\begin{gather*}
    f_{l,t} - f_{l,t-1} \leq (\underline{f}_{l,t} \hat{f}_{l} + ru_{l,t} \hat{f}_{l}) \cdot u_{l,t} - \underline{f}_{l,t} \hat{f}_{l} \cdot u_{l,t-1} \\
    - (\underline{f}_{l,t} \hat{f}_{l} + ru_{l,t} \hat{f}_{l} - rusu_{l} \hat{f}_{l}) \cdot su_{l,t}
    \end{gather*}$$

    Constraint `Link-com-partly-shut-down`:

    $$\begin{gather*}
    f_{l,t-1} - f_{l,t} \leq rdsd_{l} \hat{f}_{l} \cdot u_{l,t-1} - (rdsd_{l} \hat{f}_{l} - rd_{l,t} \hat{f}_{l}) \cdot u_{l,t} \\
    - (\underline{f}_{l,t} \hat{f}_{l} + rd_{l,t} \hat{f}_{l} - rdsd_{l} \hat{f}_{l}) \cdot su_{l,t}
    \end{gather*}$$


These constraints are defined in the function `define_operational_constraints_for_committables()`.

## Examples


<div class="grid cards" markdown>

-   :material-notebook:{ .lg .middle } **Unit Commitment**

    ---

    Models generator unit commitment with start-up and shut-down costs, ramping limits, minimum part loads, up and down times using binary variables.

    [:octicons-arrow-right-24: Go to example](../../examples/unit-commitment.ipynb)

</div>

[^1]: J.A. Taylor (2015), [Convex Optimization of Power Systems](http://www.cambridge.org/de/academic/subjects/engineering/control-systems-and-optimization/convex-optimization-power-systems), Cambridge University Press, Chapter 4.3.

[^2]: B. Hua, R. Baldick and J. Wang (2018), [Representing Operational Flexibility in Generation Expansion Planning Through Convex Relaxation of Unit Commitment](https://doi.org/10.1109/TPWRS.2017.2735026), IEEE Transactions on Power Systems, 33, 2, 2272-2281, doi:10.1109/TPWRS.2017.2735026, equations (21-24).
