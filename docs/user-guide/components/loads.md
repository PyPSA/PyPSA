# Load

The `Load` components attach to a single bus and represent a demand for the
`Bus` carrier they are connected to. With inverted sign, they can also be used
to model an exogenous supply. For "AC" buses, they act as a PQ load. If $p>0$
the load is consuming active power from the bus and if $q>0$ it is consuming
reactive power (i.e. behaving like an inductor).

!!! note "When to use `Generator` instead?"

    Use the `Generator` component with a negative `sign` to model elastic demands following a linear demand curve or to represent a comnsumption at a given price.



#TODO Table
