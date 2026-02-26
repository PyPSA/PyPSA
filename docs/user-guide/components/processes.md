<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Process

The [`Process`][pypsa.components.Processes] component is used for controllable
energy conversion processes between two or more buses with arbitrary energy
carriers (`bus0`, `bus1`, `bus2`, etc.). It serves as an alternative to
[`Link`][pypsa.components.Links] with a different parameterisation for
multi-carrier processes like electrolysis, heat pumps, or combined heat and power
(CHP) plants.

- The [`Process`][pypsa.components.Processes] component has one internal
  dispatch variable `p` and one or more power outputs `p0`, `p1`, `p2`, etc.
  associated with the buses `bus0`, `bus1`, `bus2`, etc. The power at each bus is
  determined by `pX = rateX * p`, where `rateX` is the corresponding rate
  parameter. Positive rates produce energy at the bus, negative rates consume
  energy from it.

- `p_nom` constrains the internal dispatch `p` and also determines how `capital_cost` and
  `marginal_cost` are accounted.

- For consistency, the internal dispatch carries the same unit as the output at an optional *reference bus*, chosen by setting its corresponding `rateX` to `1.` or `-1.`.

- The columns `bus2`, `rate2`, `bus3`, `rate3`, etc. in `n.processes` are
  automatically added to the component attributes.

- For delayed energy transport, use the attribute `delay0`, `delay1`, ... which postpones
  the output at `bus0`, `bus1`, ... in terms of elapsed time, taking snapshot weightings into account.
  See [this example](../../examples/transport-delay.ipynb).


!!! note "Comparison with [`Link`][pypsa.components.Links]"

    The key difference to [`Link`][pypsa.components.Links] is that `efficiency`,
    `efficiency2`, ... are replaced by `rate0`, `rate1`, `rate2`, ... with a
    single sign convention: the power output at `busX` is always `rateX * p`.
    This avoids the implicit sign convention of Links where `bus0` is always the
    input and efficiencies for additional inputs must be negative.

!!! example "[`Process`][pypsa.components.Processes] for an electrolyser"

    An electrolyser consuming electricity at `bus0` and producing hydrogen at
    `bus1` with 70% efficiency:

    ```python
    n.add(
        "Process",
        "electrolyser",
        bus0="electricity",
        bus1="hydrogen",
        rate0=-1,          # consumes 1 unit of electricity (reference bus)
        rate1=0.7,         # produces 0.7 units of hydrogen
        p_nom_extendable=True,
        capital_cost=100,  # cost per MW of electricity input
    )
    ```

!!! example "[`Process`][pypsa.components.Processes] for a CHP plant"

    A combined heat and power plant taking gas as input and producing electricity
    and heat. [This example](../../examples/sector-coupling-single-node-process.ipynb)
    illustrates sector coupling with processes including a CHP plant.

    ```python
    n.add(
        "Process",
        "CHP",
        bus0="gas",
        bus1="electricity",
        bus2="heat",
        rate0=-1,          # consumes 1 unit of gas
        rate1=0.3,         # produces 0.3 units of electricity
        rate2=0.5,         # produces 0.5 units of heat
        p_nom_extendable=True,
        capital_cost=50,
    )
    ```

!!! example "[`Process`][pypsa.components.Processes] for the Haber-Bosch process"

    The Haber-Bosch process consumes electricity and hydrogen and produces
    ammonia. Using the ammonia output as reference bus (`rate1=1`):

    ```python
    n.add(
        "Process",
        "Haber-Bosch",
        bus0="electricity",
        bus1="ammonia",
        bus2="hydrogen",
        rate0=-costs.at["Haber-Bosch", "electricity-input"],
        rate1=1,           # reference bus
        rate2=-costs.at["Haber-Bosch", "hydrogen-input"], # in tH2/tNH3
        p_nom_extendable=True,
        capital_cost=costs.at["Haber-Bosch", "capital_cost"], # in Eur/(tNH3/h)
        marginal_cost=costs.at["Haber-Bosch", "VOM"], # in Eur/tNH3
    )
    ```

{{ read_csv('../../../pypsa/data/component_attrs/processes.csv', disable_numparse=True) }}
