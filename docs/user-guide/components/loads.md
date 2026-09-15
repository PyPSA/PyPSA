<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Load

The [`Load`][pypsa.components.Loads] components attach to a single bus and represent a demand for the
[`Bus`][pypsa.components.Buses] carrier they are connected to. With inverted sign, they can also be used
to model an exogenous supply. For "AC" buses, they act as a PQ load. If $p>0$
the load is consuming active power from the bus and if $q>0$ it is consuming
reactive power (i.e. behaving like an inductor).

## Passive and dispatchable loads

A load is **passive** wherever its `p_set` is set: the consumption enters the
nodal balance as a fixed value, exactly as before. This is the common case for
inelastic demand and keeps the fast constant right-hand-side formulation.

A load is **dispatchable** wherever its `p_set` is `NaN`.
The served power is then optimised within `[p_min_pu, p_max_pu] * p_nom` and
enters the optimisation like a generator with negative dispatch. This is the way
to model price-responsive or flexible demand:

- Set `p_nom` to the maximum demand and leave `p_set` unset (`NaN`) to make the
  load fully dispatchable, or set `p_set` on some snapshots and `NaN` on others
  to mix fixed and flexible demand within a single load. The set snapshots are
  pinned to their `p_set` value.
- Use a negative `marginal_cost` to express the willingness-to-pay for served
  demand (a demand curve; see the [demand elasticity example](../../examples/demand-elasticity.ipynb)
  and the [demand and supply bids example](../../examples/demand-supply-bids.ipynb)).
- Dispatchable loads share the generator operational constraints: `p_min_pu`/`p_max_pu`
  bounds, unit commitment (`committable`) and ramp limits (`ramp_limit_up`/`down`).

Loads have no capacity-expansion dimension: `p_nom` is always a fixed input and
loads are never extendable.

!!! warning "Change of `p_set` default"

    The default of `p_set` changed from `0` to `NaN`. A load added with a `p_nom`
    but no `p_set` is therefore now dispatched actively instead of being a
    zero-consumption load.

{{ read_csv('../../../pypsa/data/component_attrs/loads.csv', disable_numparse=True) }}
