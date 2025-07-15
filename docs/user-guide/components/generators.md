# Generator

Generators attach to a single bus and can feed in power. They convert energy from their carrier to the carrier of the bus to which they attach.

In the linear optimal power flow (LOPF) and capacity expansion (CE) the limits which a generator can output are set by `p_nom*p_max_pu` and `p_nom*p_min_pu`, i.e. by limits defined per unit of the nominal power `p_nom`.

Generators can either have static or time-varying `p_max_pu` and `p_min_pu`.

Generators with static limits are like controllable conventional generators which can dispatch anywhere between `p_nom*p_min_pu` and `p_nom*p_max_pu` at all times. The static factor `p_max_pu`, stored at `n.generator.loc[gen_name, "p_max_pu"]` essentially acts like a de-rating factor.

Generators with time-varying limits are like variable weather-dependent renewable generators. The time series `p_max_pu`, stored as a series in `n.generators_t.p_max_pu[gen_name]`, dictates the active power availability for each snapshot per unit of the nominal power `p_nom` and another time series `p_min_pu` which dictates the minimum dispatch.

This time series is then multiplied by `p_nom` to get the available power dispatch, which is the maximum that may be dispatched. The actual dispatch `p`, stored in `n.generators_t.p[gen_name]`, may be below this value.

For the implementation of unit commitment, see [unit-commitment](#unit-commitment).

For generators, if $p>0$ the generator is supplying active power to the bus and if $q>0$ it is supplying reactive power (i.e. behaving like a capacitor).

{{ read_csv('../../../pypsa/data/component_attrs/generators.csv') }} 
