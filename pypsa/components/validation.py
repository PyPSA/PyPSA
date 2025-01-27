from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pypsa.components.abstract import Components


def validate(c: "Components", output_attrs: bool = False) -> None:
    c.static = c.ctype.schema_output_static(c.static)

    # TODO: This can be removed, once the dynamic schema checks both axes as well
    for attr in c.defaults[(c.defaults.status == "output") & c.defaults.dynamic].index:
        if attr not in c.dynamic:
            c.dynamic[attr] = pd.DataFrame(
                index=c.n_save.snapshots, columns=c.static.index
            )

    for k, v in c.dynamic.items():
        if v.empty:
            continue
        if k in c.ctype.schemas_output_dynamic:
            c.dynamic[k] = c.ctype.schemas_output_dynamic[k](v)
        elif k in c.ctype.schemas_input_dynamic:
            c.dynamic[k] = c.ctype.schemas_input_dynamic[k](v)
        else:
            raise ValueError(f"Schema for {k} not found")
