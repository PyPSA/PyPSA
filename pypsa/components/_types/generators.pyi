from collections.abc import Sequence
from typing import TypeAlias, TypeVar

import pandas as pd
from numpy.typing import ArrayLike

_T = TypeVar("_T")

SeriesLike: TypeAlias = Sequence[_T] | pd.Series

class Generators:
    def add(
        self,
        name: str | int | Sequence[int | str],
        bus: str | SeriesLike[str] = "",
        control: str | SeriesLike[str] = "PQ",
        type: str | SeriesLike[str] = "n/a",
        p_nom: float | SeriesLike[float] = 0,
        p_nom_mod: float | SeriesLike[float] = 0,
        p_nom_extendable: bool | SeriesLike[bool] = False,
        p_nom_min: float | SeriesLike[float] = 0,
        p_nom_max: float | SeriesLike[float] = ...,
        p_min_pu: float | SeriesLike[float] | ArrayLike = 0.0,
        p_max_pu: float | SeriesLike[float] | ArrayLike = 1.0,
        p_set: float | SeriesLike[float] | ArrayLike = 0.0,
        e_sum_min: float | SeriesLike[float] = ...,
        e_sum_max: float | SeriesLike[float] = ...,
        q_set: float | SeriesLike[float] | ArrayLike = 0.0,
        sign: float | SeriesLike[float] = 1,
        carrier: str | None | SeriesLike[str | None] = None,
        marginal_cost: float | SeriesLike[float] | ArrayLike = 0.0,
        marginal_cost_quadratic: float | SeriesLike[float] | ArrayLike = 0.0,
        active: bool | SeriesLike[bool] = True,
        build_year: int | SeriesLike[int] = 0,
        lifetime: float | SeriesLike[float] = ...,
        capital_cost: float | SeriesLike[float] = 0,
        efficiency: float | SeriesLike[float] | ArrayLike = 1.0,
        committable: bool | SeriesLike[bool] = False,
        start_up_cost: float | SeriesLike[float] = 0,
        shut_down_cost: float | SeriesLike[float] = 0,
        stand_by_cost: float | SeriesLike[float] | ArrayLike = 0.0,
        min_up_time: int | SeriesLike[int] = 0,
        min_down_time: int | SeriesLike[int] = 0,
        up_time_before: int | SeriesLike[int] = 1,
        down_time_before: int | SeriesLike[int] = 0,
        ramp_limit_up: float | SeriesLike[float] | ArrayLike = ...,
        ramp_limit_down: float | SeriesLike[float] | ArrayLike = ...,
        ramp_limit_start_up: float | SeriesLike[float] = 1.0,
        ramp_limit_shut_down: float | SeriesLike[float] = 1.0,
        weight: float | SeriesLike[float] = 1.0,
    ) -> None: ...
