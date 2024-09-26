"""
Power flow functionality.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
from numpy import ones, r_
from numpy.linalg import norm
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, issparse
from scipy.sparse import hstack as shstack
from scipy.sparse import vstack as svstack
from scipy.sparse.linalg import spsolve

from pypsa.definitions.structures import Dict
from pypsa.descriptors import (
    additional_linkports,
    allocate_series_dataframes,
    update_linkports_component_attrs,
    zsum,
)
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.utils import as_index, deprecated_common_kwargs

if TYPE_CHECKING:
    from components import Network, SubNetwork

pd.Series.zsum = zsum
logger = logging.getLogger(__name__)


def normed(s: pd.Series) -> pd.Series:
    return s / s.sum()


def real(X: pd.Series) -> pd.Series:
    return np.real(X.to_numpy())


def imag(X: pd.Series) -> pd.Series:
    return np.imag(X.to_numpy())


@deprecated_common_kwargs
def _allocate_pf_outputs(n: Network, linear: bool = False) -> None:
    to_allocate = {
        "Generator": ["p"],
        "Load": ["p"],
        "StorageUnit": ["p"],
        "Store": ["p"],
        "ShuntImpedance": ["p"],
        "Bus": ["p", "v_ang", "v_mag_pu"],
        "Line": ["p0", "p1"],
        "Transformer": ["p0", "p1"],
        "Link": ["p" + col[3:] for col in n.links.columns if col[:3] == "bus"],
    }

    if not linear:
        for component, attrs in to_allocate.items():
            if "p" in attrs:
                attrs.append("q")
            if "p0" in attrs and component != "Link":
                attrs.extend(["q0", "q1"])

    allocate_series_dataframes(n, to_allocate)


def _calculate_controllable_nodal_power_balance(
    sub_network: SubNetwork,
    network: Network,
    snapshots: Sequence,
    buses_o: Sequence,
) -> None:
    for power in ("q", "p"):
        # allow all one ports to dispatch as set
        for c in sub_network.iterate_components(
            network.controllable_one_port_components
        ):
            c_n_set = get_as_dense(
                network,
                c.name,
                power + "_set",
                snapshots,
                c.static.query("active").index,
            )
            network.dynamic(c.name)[power].loc[
                snapshots, c.static.query("active").index
            ] = c_n_set

        # set the power injection at each node from controllable components
        network.buses_t[power].loc[snapshots, buses_o] = sum(
            (
                (
                    c.dynamic[power].loc[snapshots, c.static.query("active").index]
                    * c.static.loc[c.static.query("active").index, "sign"]
                )
                .T.groupby(c.static.loc[c.static.query("active").index, "bus"])
                .sum()
                .T.reindex(columns=buses_o, fill_value=0.0)
            )
            for c in sub_network.iterate_components(
                network.controllable_one_port_components
            )
        )

        if power == "p":
            network.buses_t[power].loc[snapshots, buses_o] += sum(
                -c.dynamic[power + str(i)]
                .loc[snapshots]
                .T.groupby(c.static[f"bus{str(i)}"])
                .sum()
                .T.reindex(columns=buses_o, fill_value=0)
                for c in network.iterate_components(
                    network.controllable_branch_components
                )
                for i in [int(col[3:]) for col in c.static.columns if col[:3] == "bus"]
            )


@deprecated_common_kwargs
def _network_prepare_and_run_pf(
    n: Network,
    snapshots: Sequence | None,
    skip_pre: bool,
    linear: bool = False,
    distribute_slack: bool = False,
    slack_weights: str = "p_set",
    **kwargs: Any,
) -> Dict | None:
    if linear:
        sub_network_pf_fun: Callable = sub_network_lpf
        sub_network_prepare_fun: Callable = calculate_B_H
    else:
        sub_network_pf_fun: Callable = sub_network_pf  # type: ignore
        sub_network_prepare_fun: Callable = calculate_Y  # type: ignore

    if not skip_pre:
        n.determine_network_topology()
        calculate_dependent_values(n)
        _allocate_pf_outputs(n, linear)

    sns = as_index(n, snapshots, "snapshots", "snapshot")

    # deal with links
    if not n.links.empty:
        p_set = get_as_dense(n, "Link", "p_set", sns)
        n.links_t.p0.loc[sns] = p_set.loc[sns]
        for i in ["1"] + additional_linkports(n):
            eff_name = "efficiency" if i == "1" else f"efficiency{i}"
            efficiency = get_as_dense(n, "Link", eff_name, sns)
            links = n.links.index[n.links[f"bus{i}"] != ""]
            n.links_t[f"p{i}"].loc[sns, links] = (
                -n.links_t.p0.loc[sns, links] * efficiency.loc[sns, links]
            )

    itdf = pd.DataFrame(index=sns, columns=n.sub_networks.index, dtype=int)
    difdf = pd.DataFrame(index=sns, columns=n.sub_networks.index)
    cnvdf = pd.DataFrame(index=sns, columns=n.sub_networks.index, dtype=bool)
    for sub_network in n.sub_networks.obj:
        if not skip_pre:
            find_bus_controls(sub_network)

            branches_i = sub_network.branches_i(active_only=True)
            if len(branches_i) > 0:
                sub_network_prepare_fun(sub_network, skip_pre=True)

        if isinstance(slack_weights, dict):
            sn_slack_weights = slack_weights[sub_network.name]
        else:
            sn_slack_weights = slack_weights

        if isinstance(sn_slack_weights, dict):
            sn_slack_weights = pd.Series(sn_slack_weights)

        if linear:
            sub_network_pf_fun(sub_network, snapshots=sns, skip_pre=True, **kwargs)

        elif len(sub_network.buses()) <= 1:
            (
                itdf[sub_network.name],
                difdf[sub_network.name],
                cnvdf[sub_network.name],
            ) = sub_network_pf_singlebus(
                sub_network,
                snapshots=sns,
                skip_pre=True,
                distribute_slack=distribute_slack,
                slack_weights=sn_slack_weights,
            )
        else:
            (
                itdf[sub_network.name],
                difdf[sub_network.name],
                cnvdf[sub_network.name],
            ) = sub_network_pf_fun(
                sub_network,
                snapshots=sns,
                skip_pre=True,
                distribute_slack=distribute_slack,
                slack_weights=sn_slack_weights,
                **kwargs,
            )
    if not linear:
        return Dict({"n_iter": itdf, "error": difdf, "converged": cnvdf})
    else:
        return None


@deprecated_common_kwargs
def network_pf(
    n: Network,
    snapshots: Sequence | None = None,
    skip_pre: bool = False,
    x_tol: float = 1e-6,
    use_seed: bool = False,
    distribute_slack: bool = False,
    slack_weights: str = "p_set",
) -> Dict:
    """
    Full non-linear power flow for generic network.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of n.snapshots on which to run
        the power flow, defaults to n.snapshots
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating dependent values and finding bus controls.
    x_tol: float
        Tolerance for Newton-Raphson power flow.
    use_seed : bool, default False
        Use a seed for the initial guess for the Newton-Raphson algorithm.
    distribute_slack : bool, default False
        If ``True``, distribute the slack power across generators proportional to generator dispatch by default
        or according to the distribution scheme provided in ``slack_weights``.
        If ``False`` only the slack generator takes up the slack.
    slack_weights : dict|str, default 'p_set'
        Distribution scheme describing how to determine the fraction of the total slack power
        (of each sub network individually) a bus of the sub-network takes up.
        Default is to distribute proportional to generator dispatch ('p_set').
        Another option is to distribute proportional to (optimised) nominal capacity ('p_nom' or 'p_nom_opt').
        Custom weights can be specified via a dictionary that has a key for each
        sub-network index (``n.sub_networks.index``) and a
        pandas.Series/dict with buses or generators of the
        corresponding sub-network as index/keys.
        When specifying custom weights with buses as index/keys the slack power of a bus is distributed
        among its generators in proportion to their nominal capacity (``p_nom``) if given, otherwise evenly.

    Returns
    -------
    dict
        Dictionary with keys 'n_iter', 'converged', 'error' and dataframe
        values indicating number of iterations, convergence status, and
        iteration error for each snapshot (rows) and sub_network (columns)
    """

    return _network_prepare_and_run_pf(
        n,
        snapshots,
        skip_pre,
        linear=False,
        x_tol=x_tol,
        use_seed=use_seed,
        distribute_slack=distribute_slack,
        slack_weights=slack_weights,
    )


def newton_raphson_sparse(
    f: Callable,
    guess: np.ndarray,
    dfdx: Callable,
    x_tol: float = 1e-10,
    lim_iter: int = 100,
    distribute_slack: bool = False,
    slack_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, int, Any, bool]:
    """
    Solve f(x) = 0 with initial guess for x and dfdx(x).

    dfdx(x) should return a sparse Jacobian.  Terminate if error on norm
    of f(x) is < x_tol or there were more than lim_iter iterations.
    """

    slack_args = {"distribute_slack": distribute_slack, "slack_weights": slack_weights}
    converged = False
    n_iter = 0
    F = f(guess, **slack_args)
    diff = norm(F, np.inf)

    logger.debug("Error at iteration %d: %f", n_iter, diff)

    while diff > x_tol and n_iter < lim_iter:
        n_iter += 1

        guess = guess - spsolve(dfdx(guess, **slack_args), F)

        F = f(guess, **slack_args)
        diff = norm(F, np.inf)

        logger.debug("Error at iteration %d: %f", n_iter, diff)

    if diff > x_tol:
        logger.warning(
            'Warning, we didn\'t reach the required tolerance within %d iterations, error is at %f. See the section "Troubleshooting" in the documentation for tips to fix this. ',
            n_iter,
            diff,
        )
    elif not np.isnan(diff):
        converged = True

    return guess, n_iter, diff, converged


def sub_network_pf_singlebus(
    sub_network: SubNetwork,
    snapshots: Sequence | None = None,
    skip_pre: float = False,
    distribute_slack: bool = False,
    slack_weights: str | pd.Series = "p_set",
    linear: bool = False,
) -> tuple[int, float, bool]:
    """
    Non-linear power flow for a sub-network consiting of a single bus.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of n.snapshots on which to run
        the power flow, defaults to n.snapshots
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating dependent values and finding bus controls.
    distribute_slack : bool, default False
        If ``True``, distribute the slack power across generators proportional to generator dispatch by default
        or according to the distribution scheme provided in ``slack_weights``.
        If ``False`` only the slack generator takes up the slack.
    slack_weights : pandas.Series|str, default 'p_set'
        Distribution scheme describing how to determine the fraction of the total slack power
        a bus of the sub-network takes up. Default is to distribute proportional to generator dispatch
        ('p_set'). Another option is to distribute proportional to (optimised) nominal capacity ('p_nom' or 'p_nom_opt').
        Custom weights can be provided via a pandas.Series/dict
        that has the generators of the single bus as index/keys.
    """

    sns = as_index(sub_network.n, snapshots, "snapshots", "snapshot")
    n = sub_network.n
    logger.info(
        f"Balancing power on single-bus sub-network {sub_network} for snapshots {snapshots}"
    )

    if not skip_pre:
        find_bus_controls(sub_network)
        _allocate_pf_outputs(n, linear=False)

    if isinstance(slack_weights, dict):
        slack_weights = pd.Series(slack_weights)

    buses_o = sub_network.buses_o

    _calculate_controllable_nodal_power_balance(sub_network, n, sns, buses_o)

    v_mag_pu_set = get_as_dense(n, "Bus", "v_mag_pu_set", sns)
    n.buses_t.v_mag_pu.loc[sns, sub_network.slack_bus] = v_mag_pu_set.loc[
        :, sub_network.slack_bus
    ]
    n.buses_t.v_ang.loc[sns, sub_network.slack_bus] = 0.0

    if distribute_slack:
        for bus, group in sub_network.generators().groupby("bus"):
            if slack_weights in ["p_nom", "p_nom_opt"]:
                if all(n.generators[slack_weights] == 0):
                    msg = f"Invalid slack weights! Generator attribute {slack_weights} is always zero."
                    raise ValueError(msg)
                bus_generator_shares = (
                    n.generators[slack_weights].loc[group.index].pipe(normed).fillna(0)
                )
            elif slack_weights == "p_set":
                generators_t_p_choice = get_as_dense(n, "Generator", slack_weights, sns)
                if generators_t_p_choice.isna().all().all():
                    msg = (
                        f"Invalid slack weights! Generator attribute {slack_weights}"
                        f" is always NaN."
                    )
                    raise ValueError(msg)
                if (generators_t_p_choice == 0).all().all():
                    msg = (
                        f"Invalid slack weights! Generator attribute {slack_weights}"
                        f" is always zero."
                    )
                    raise ValueError(msg)

                bus_generator_shares = (
                    generators_t_p_choice.loc[sns, group.index]
                    .apply(normed, axis=1)
                    .fillna(0)
                )
            else:
                bus_generator_shares = slack_weights.pipe(normed).fillna(0)
            n.generators_t.p.loc[sns, group.index] += (
                bus_generator_shares.multiply(
                    -n.buses_t.p.loc[sns, bus], axis=0
                )
            )  # fmt: skip
    else:
        n.generators_t.p.loc[sns, sub_network.slack_generator] -= (
            n.buses_t.p.loc[sns, sub_network.slack_bus]
        )  # fmt: skip

    n.generators_t.q.loc[sns, sub_network.slack_generator] -= (
        n.buses_t.q.loc[sns, sub_network.slack_bus]
    )  # fmt: skip

    n.buses_t.p.loc[sns, sub_network.slack_bus] = 0.0
    n.buses_t.q.loc[sns, sub_network.slack_bus] = 0.0

    return 0, 0.0, True  # dummy substitute for newton raphson output


def sub_network_pf(
    sub_network: SubNetwork,
    snapshots: Sequence | None = None,
    skip_pre: bool = False,
    x_tol: float = 1e-6,
    use_seed: bool = False,
    distribute_slack: bool = False,
    slack_weights: pd.Series | dict | str = "p_set",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Non-linear power flow for connected sub-network.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of n.snapshots on which to run
        the power flow, defaults to n.snapshots
    skip_pre: bool, default False
        Skip the preliminary steps of computing topology, calculating dependent values and finding bus controls.
    x_tol: float
        Tolerance for Newton-Raphson power flow.
    use_seed : bool, default False
        Use a seed for the initial guess for the Newton-Raphson algorithm.
    distribute_slack : bool, default False
        If ``True``, distribute the slack power across generators proportional to generator dispatch by default
        or according to the distribution scheme provided in ``slack_weights``.
        If ``False`` only the slack generator takes up the slack.
    slack_weights : pandas.Series|str, default 'p_set'
        Distribution scheme describing how to determine the fraction of the total slack power
        a bus of the sub-network takes up. Default is to distribute proportional to generator dispatch
        ('p_set'). Another option is to distribute proportional to (optimised) nominal capacity ('p_nom' or 'p_nom_opt').
        Custom weights can be provided via a pandas.Series/dict
        that has the buses or the generators of the sub-network as index/keys.
        When using custom weights with buses as index/keys the slack power of a bus is distributed
        among its generators in proportion to their nominal capacity (``p_nom``) if given, otherwise evenly.

    Returns
    -------
    Tuple of three pandas.Series indicating number of iterations,
    remaining error, and convergence status for each snapshot
    """

    if not isinstance(slack_weights, (str, pd.Series, dict)):
        msg = (
            f"Type of 'slack_weights' must be string, pd.Series or dict. Got "
            f"{type(slack_weights)}."
        )
        raise TypeError(msg)

    if isinstance(slack_weights, dict):
        slack_weights = pd.Series(slack_weights)
    elif isinstance(slack_weights, str):
        valid_strings = ["p_nom", "p_nom_opt", "p_set"]
        if slack_weights not in valid_strings:
            msg = (
                f"String value for 'slack_weights' must be one of {valid_strings}. "
                f"Is {slack_weights}."
            )
            raise ValueError(msg)

    sns = as_index(sub_network.n, snapshots, "snapshots", "snapshot")
    logger.info(
        "Performing non-linear load-flow on {} sub-network {} for snapshots {}".format(
            sub_network.n.sub_networks.at[sub_network.name, "carrier"],
            sub_network,
            sns,
        )
    )

    # _sub_network_prepare_pf(sub_network, snapshots, skip_pre, calculate_Y)
    n = sub_network.n

    if not skip_pre:
        calculate_dependent_values(n)
        find_bus_controls(sub_network)
        _allocate_pf_outputs(n, linear=False)

    # get indices for the components on this sub-network
    branches_i = sub_network.branches_i(active_only=True)
    buses_o = sub_network.buses_o
    sn_buses = sub_network.buses().index
    sn_generators = sub_network.generators().index

    generator_slack_weights_b = False
    bus_slack_weights_b = False
    if isinstance(slack_weights, pd.Series):
        if all(i in sn_generators for i in slack_weights.index):
            generator_slack_weights_b = True
        elif all(i in sn_buses for i in slack_weights.index):
            bus_slack_weights_b = True
        else:
            raise ValueError(
                "Custom slack weights pd.Series/dict must only have the",
                "generators or buses of the sub-network as index/keys.",
            )

    if not skip_pre and len(branches_i) > 0:
        calculate_Y(sub_network, skip_pre=True)

    _calculate_controllable_nodal_power_balance(sub_network, n, sns, buses_o)

    def f(
        guess: np.ndarray,
        distribute_slack: bool = False,
        slack_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        last_pq = -1 if distribute_slack else None
        n.buses_t.v_ang.loc[now, sub_network.pvpqs] = guess[: len(sub_network.pvpqs)]
        n.buses_t.v_mag_pu.loc[now, sub_network.pqs] = guess[
            len(sub_network.pvpqs) : last_pq
        ]

        v_mag_pu = n.buses_t.v_mag_pu.loc[now, buses_o]
        v_ang = n.buses_t.v_ang.loc[now, buses_o]
        V = v_mag_pu * np.exp(1j * v_ang)

        if distribute_slack:
            slack_power = slack_weights * guess[-1]
            mismatch = V * np.conj(sub_network.Y * V) - s + slack_power
        else:
            mismatch = V * np.conj(sub_network.Y * V) - s

        if distribute_slack:
            F = r_[real(mismatch)[:], imag(mismatch)[1 + len(sub_network.pvs) :]]
        else:
            F = r_[real(mismatch)[1:], imag(mismatch)[1 + len(sub_network.pvs) :]]

        return F

    def dfdx(
        guess: np.ndarray,
        distribute_slack: bool = False,
        slack_weights: np.ndarray | None = None,
    ) -> csr_matrix:
        last_pq = -1 if distribute_slack else None
        n.buses_t.v_ang.loc[now, sub_network.pvpqs] = guess[: len(sub_network.pvpqs)]
        n.buses_t.v_mag_pu.loc[now, sub_network.pqs] = guess[
            len(sub_network.pvpqs) : last_pq
        ]

        v_mag_pu = n.buses_t.v_mag_pu.loc[now, buses_o]
        v_ang = n.buses_t.v_ang.loc[now, buses_o]

        V = v_mag_pu * np.exp(1j * v_ang)

        index = r_[: len(buses_o)]

        # make sparse diagonal matrices
        V_diag = csr_matrix((V, (index, index)))
        V_norm_diag = csr_matrix((V / abs(V), (index, index)))
        I_diag = csr_matrix((sub_network.Y * V, (index, index)))

        dS_dVa = 1j * V_diag * np.conj(I_diag - sub_network.Y * V_diag)

        dS_dVm = V_norm_diag * np.conj(I_diag) + V_diag * np.conj(
            sub_network.Y * V_norm_diag
        )

        J10 = dS_dVa[1 + len(sub_network.pvs) :, 1:].imag
        J11 = dS_dVm[1 + len(sub_network.pvs) :, 1 + len(sub_network.pvs) :].imag

        if distribute_slack:
            J00 = dS_dVa[:, 1:].real
            J01 = dS_dVm[:, 1 + len(sub_network.pvs) :].real
            J02 = csr_matrix(slack_weights, (1, 1 + len(sub_network.pvpqs))).T
            J12 = csr_matrix((1, len(sub_network.pqs))).T
            J_P_blocks = [J00, J01, J02]
            J_Q_blocks = [J10, J11, J12]
        else:
            J00 = dS_dVa[1:, 1:].real
            J01 = dS_dVm[1:, 1 + len(sub_network.pvs) :].real
            J_P_blocks = [J00, J01]
            J_Q_blocks = [J10, J11]

        J = svstack([shstack(J_P_blocks), shstack(J_Q_blocks)], format="csr")

        return J

    # Set what we know: slack V and v_mag_pu for PV buses
    v_mag_pu_set = get_as_dense(n, "Bus", "v_mag_pu_set", sns)
    n.buses_t.v_mag_pu.loc[sns, sub_network.pvs] = v_mag_pu_set.loc[:, sub_network.pvs]
    n.buses_t.v_mag_pu.loc[sns, sub_network.slack_bus] = v_mag_pu_set.loc[
        :, sub_network.slack_bus
    ]
    n.buses_t.v_ang.loc[sns, sub_network.slack_bus] = 0.0

    if not use_seed:
        n.buses_t.v_mag_pu.loc[sns, sub_network.pqs] = 1.0
        n.buses_t.v_ang.loc[sns, sub_network.pvpqs] = 0.0

    slack_args = {"distribute_slack": distribute_slack}
    slack_variable_b = 1 if distribute_slack else 0

    if distribute_slack:
        if isinstance(slack_weights, str) and slack_weights == "p_set":
            generators_t_p_choice = get_as_dense(n, "Generator", slack_weights, sns)
            bus_generation = generators_t_p_choice.rename(columns=n.generators.bus)
            slack_weights_calc = (
                pd.DataFrame(
                    bus_generation.T.groupby(bus_generation.columns).sum().T,
                    columns=buses_o,
                )
                .apply(normed, axis=1)
                .fillna(0)
            )

        elif isinstance(slack_weights, str) and slack_weights in ["p_nom", "p_nom_opt"]:
            if all(n.generators[slack_weights] == 0):
                msg = (
                    f"Invalid slack weights! Generator attribute {slack_weights} is "
                    f"always zero."
                )
                raise ValueError(msg)

            slack_weights_calc = (
                n.generators.groupby("bus")[slack_weights]
                .sum()
                .reindex(buses_o)
                .pipe(normed)
                .fillna(0)
            )

        elif generator_slack_weights_b:
            # convert generator-based slack weights to bus-based slack weights
            slack_weights_calc = (
                slack_weights.rename(n.generators.bus)
                .groupby(slack_weights.index.name)
                .sum()
                .reindex(buses_o)
                .pipe(normed)
                .fillna(0)
            )

        elif bus_slack_weights_b:
            # take bus-based slack weights
            slack_weights_calc = slack_weights.reindex(buses_o).pipe(normed).fillna(0)

    ss = np.empty((len(sns), len(buses_o)), dtype=complex)
    roots = np.empty(
        (
            len(sns),
            len(sub_network.pvpqs) + len(sub_network.pqs) + slack_variable_b,
        )
    )
    iters = pd.Series(0, index=sns)
    diffs = pd.Series(index=sns, dtype=float)
    convs = pd.Series(False, index=sns)
    for i, now in enumerate(sns):
        p = n.buses_t.p.loc[now, buses_o]
        q = n.buses_t.q.loc[now, buses_o]
        ss[i] = s = p + 1j * q

        # Make a guess for what we don't know: V_ang for PV and PQs and v_mag_pu for PQ buses
        guess = r_[
            n.buses_t.v_ang.loc[now, sub_network.pvpqs],
            n.buses_t.v_mag_pu.loc[now, sub_network.pqs],
        ]

        if distribute_slack:
            guess = np.append(guess, [0])  # for total slack power
            if isinstance(slack_weights, str) and slack_weights == "p_set":
                # snapshot-dependent slack weights
                slack_args["slack_weights"] = slack_weights_calc.loc[now]
            else:
                slack_args["slack_weights"] = slack_weights_calc

        # Now try and solve
        roots[i], n_iter, diff, converged = newton_raphson_sparse(
            f,
            guess,
            dfdx,
            x_tol=x_tol,
            **slack_args,  # type: ignore
        )
        iters[now] = n_iter
        diffs[now] = diff
        convs[now] = converged
    if not convs.all():
        not_converged = sns[~convs]
        logger.warning(f"Power flow did not converge for {list(not_converged)}.")

    # now set everything
    if distribute_slack:
        last_pq = -1
    else:
        last_pq = None
    n.buses_t.v_ang.loc[sns, sub_network.pvpqs] = roots[:, : len(sub_network.pvpqs)]
    n.buses_t.v_mag_pu.loc[sns, sub_network.pqs] = roots[
        :, len(sub_network.pvpqs) : last_pq
    ]

    v_mag_pu = n.buses_t.v_mag_pu.loc[sns, buses_o].values
    v_ang = n.buses_t.v_ang.loc[sns, buses_o].values

    V = v_mag_pu * np.exp(1j * v_ang)

    # add voltages to branches
    buses_indexer = buses_o.get_indexer
    branch_bus0 = []
    branch_bus1 = []
    for c in sub_network.iterate_components(n.passive_branch_components):
        branch_bus0 += list(c.static.query("active").bus0)
        branch_bus1 += list(c.static.query("active").bus1)
    v0 = V[:, buses_indexer(branch_bus0)]
    v1 = V[:, buses_indexer(branch_bus1)]

    i0 = np.empty((len(sns), sub_network.Y0.shape[0]), dtype=complex)
    i1 = np.empty((len(sns), sub_network.Y1.shape[0]), dtype=complex)
    for i, now in enumerate(sns):
        i0[i] = sub_network.Y0 * V[i]
        i1[i] = sub_network.Y1 * V[i]

    s0 = pd.DataFrame(v0 * np.conj(i0), columns=branches_i, index=sns)
    s1 = pd.DataFrame(v1 * np.conj(i1), columns=branches_i, index=sns)
    for c in sub_network.iterate_components(n.passive_branch_components):
        s0t = s0.loc[:, c.name]
        s1t = s1.loc[:, c.name]
        n.dynamic(c.name).p0.loc[sns, s0t.columns] = s0t.values.real
        n.dynamic(c.name).q0.loc[sns, s0t.columns] = s0t.values.imag
        n.dynamic(c.name).p1.loc[sns, s1t.columns] = s1t.values.real
        n.dynamic(c.name).q1.loc[sns, s1t.columns] = s1t.values.imag

    s_calc = np.empty((len(sns), len(buses_o)), dtype=complex)
    for i in np.arange(len(sns)):
        s_calc[i] = V[i] * np.conj(sub_network.Y * V[i])
    slack_index = buses_o.get_loc(sub_network.slack_bus)
    if distribute_slack:
        n.buses_t.p.loc[sns, sn_buses] = s_calc.real[:, buses_indexer(sn_buses)]
    else:
        n.buses_t.p.loc[sns, sub_network.slack_bus] = s_calc[:, slack_index].real
    n.buses_t.q.loc[sns, sub_network.slack_bus] = s_calc[:, slack_index].imag
    n.buses_t.q.loc[sns, sub_network.pvs] = s_calc[
        :, buses_indexer(sub_network.pvs)
    ].imag

    # set shunt impedance powers
    shunt_impedances_i = sub_network.shunt_impedances_i()
    if len(shunt_impedances_i):
        # add voltages
        shunt_impedances_v_mag_pu = v_mag_pu[
            :, buses_indexer(n.shunt_impedances.loc[shunt_impedances_i, "bus"])
        ]
        n.shunt_impedances_t.p.loc[sns, shunt_impedances_i] = (
            shunt_impedances_v_mag_pu**2
        ) * n.shunt_impedances.loc[shunt_impedances_i, "g_pu"].values
        n.shunt_impedances_t.q.loc[sns, shunt_impedances_i] = (
            shunt_impedances_v_mag_pu**2
        ) * n.shunt_impedances.loc[shunt_impedances_i, "b_pu"].values

    # let slack generator take up the slack
    if distribute_slack:
        distributed_slack_power = (
            n.buses_t.p.loc[sns, sn_buses] - ss[:, buses_indexer(sn_buses)].real
        )
        for bus, group in sub_network.generators().groupby("bus"):
            if isinstance(slack_weights, str) and slack_weights == "p_set":
                generators_t_p_choice = get_as_dense(n, "Generator", slack_weights, sns)
                bus_generator_shares = (
                    generators_t_p_choice.loc[sns, group.index]
                    .apply(normed, axis=1)
                    .fillna(0)
                )
                n.generators_t.p.loc[sns, group.index] += (
                    bus_generator_shares.multiply(
                        distributed_slack_power.loc[sns, bus], axis=0
                    )
                )  # fmt: skip
            else:
                if generator_slack_weights_b:
                    bus_generator_shares = (
                        slack_weights.loc[group.index].pipe(normed).fillna(0)  # type: ignore
                    )
                else:
                    bus_generators_p_nom = n.generators.p_nom.loc[group.index]
                    # distribute evenly if no p_nom given
                    if all(bus_generators_p_nom) == 0:
                        bus_generators_p_nom = 1
                    bus_generator_shares = bus_generators_p_nom.pipe(normed).fillna(0)
                n.generators_t.p.loc[sns, group.index] += (
                    distributed_slack_power.loc[
                        sns, bus
                    ].apply(lambda row: row * bus_generator_shares)
                )  # fmt: skip
    else:
        n.generators_t.p.loc[sns, sub_network.slack_generator] += (
            n.buses_t.p.loc[sns, sub_network.slack_bus] - ss[:, slack_index].real
        )

    # set the Q of the slack and PV generators
    n.generators_t.q.loc[sns, sub_network.slack_generator] += (
        n.buses_t.q.loc[sns, sub_network.slack_bus] - ss[:, slack_index].imag
    )

    n.generators_t.q.loc[sns, n.buses.loc[sub_network.pvs, "generator"]] += np.asarray(
        n.buses_t.q.loc[sns, sub_network.pvs]
        - ss[:, buses_indexer(sub_network.pvs)].imag
    )

    return iters, diffs, convs


@deprecated_common_kwargs
def network_lpf(
    n: Network, snapshots: Sequence | None = None, skip_pre: bool = False
) -> None:
    """
    Linear power flow for generic network.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of n.snapshots on which to run
        the power flow, defaults to n.snapshots
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.

    Returns
    -------
    None
    """
    sns = as_index(n, snapshots, "snapshots", "snapshot")
    _network_prepare_and_run_pf(n, sns, skip_pre, linear=True)


@deprecated_common_kwargs
def apply_line_types(n: Network) -> None:
    """
    Calculate line electrical parameters x, r, b, g from standard types.
    """
    lines_with_types_b = n.lines.type != ""
    if lines_with_types_b.zsum() == 0:
        return

    missing_types = pd.Index(
        n.lines.loc[lines_with_types_b, "type"].unique()
    ).difference(n.line_types.index)
    if not missing_types.empty:
        msg = (
            f'The type(s) {", ".join(missing_types)} do(es) not exist in '
            f"n.line_types"
        )
        raise ValueError(msg)

    # Get a copy of the lines data
    lines = n.lines.loc[lines_with_types_b, ["type", "length", "num_parallel"]].join(
        n.line_types, on="type"
    )

    for attr in ["r", "x"]:
        lines[attr] = (
            lines[attr + "_per_length"] * lines["length"] / lines["num_parallel"]
        )
    lines["b"] = (
        2
        * np.pi
        * 1e-9
        * lines["f_nom"]
        * lines["c_per_length"]
        * lines["length"]
        * lines["num_parallel"]
    )

    # now set calculated values on live lines
    for attr in ["r", "x", "b"]:
        n.lines.loc[lines_with_types_b, attr] = lines[attr]


@deprecated_common_kwargs
def apply_transformer_types(n: Network) -> None:
    """
    Calculate transformer electrical parameters x, r, b, g from standard types.
    """
    trafos_with_types_b = n.transformers.type != ""
    if trafos_with_types_b.zsum() == 0:
        return

    missing_types = pd.Index(
        n.transformers.loc[trafos_with_types_b, "type"].unique()
    ).difference(n.transformer_types.index)
    if not missing_types.empty:
        msg = (
            f'The type(s) {", ".join(missing_types)} do(es) not exist in '
            f"n.transformer_types"
        )
        raise ValueError(msg)

    # Get a copy of the transformers data
    # (joining pulls in "phase_shift", "s_nom", "tap_side" from TransformerType)
    t = n.transformers.loc[
        trafos_with_types_b, ["type", "tap_position", "num_parallel"]
    ].join(n.transformer_types, on="type")

    t["r"] = t["vscr"] / 100.0
    t["x"] = np.sqrt((t["vsc"] / 100.0) ** 2 - t["r"] ** 2)

    # NB: b and g are per unit of s_nom
    t["g"] = t["pfe"] / (1000.0 * t["s_nom"])

    # for some bizarre reason, some of the standard types in pandapower have i0^2 < g^2
    t["b"] = -np.sqrt(((t["i0"] / 100.0) ** 2 - t["g"] ** 2).clip(lower=0))

    for attr in ["r", "x"]:
        t[attr] /= t["num_parallel"]

    for attr in ["b", "g"]:
        t[attr] *= t["num_parallel"]

    # deal with tap positions

    t["tap_ratio"] = 1.0 + (t["tap_position"] - t["tap_neutral"]) * (
        t["tap_step"] / 100.0
    )

    # now set calculated values on live transformers
    for attr in ["r", "x", "g", "b", "phase_shift", "s_nom", "tap_side", "tap_ratio"]:
        n.transformers.loc[trafos_with_types_b, attr] = t[attr]

    # TODO: status, rate_A


def wye_to_delta(
    z1: float,
    z2: float,
    z3: float,
) -> tuple[float, float, float]:
    """
    Follows http://home.earthlink.net/~w6rmk/math/wyedelta.html.

    Parameters
    ----------
        z1 (int, float, or complex): First impedance value
        z2 (int, float, or complex): Second impedance value
        z3 (int, float, or complex): Third impedance value

    Returns
    -------
        tuple[float, float, float]:
            A tuple containing the three transformed impedance values
    """
    summand = z1 * z2 + z2 * z3 + z3 * z1
    return (summand / z2, summand / z1, summand / z3)


@deprecated_common_kwargs
def apply_transformer_t_model(n: Network) -> None:
    """
    Convert given T-model parameters to PI-model parameters using wye-delta
    transformation.
    """
    z_series = n.transformers.r_pu + 1j * n.transformers.x_pu
    y_shunt = n.transformers.g_pu + 1j * n.transformers.b_pu

    ts_b = (n.transformers.model == "t") & (y_shunt != 0.0)

    if ts_b.zsum() == 0:
        return

    za, zb, zc = wye_to_delta(
        z_series.loc[ts_b] / 2, z_series.loc[ts_b] / 2, 1 / y_shunt.loc[ts_b]
    )

    n.transformers.loc[ts_b, "r_pu"] = real(zc)
    n.transformers.loc[ts_b, "x_pu"] = imag(zc)
    n.transformers.loc[ts_b, "g_pu"] = real(2 / za)
    n.transformers.loc[ts_b, "b_pu"] = imag(2 / za)


@deprecated_common_kwargs
def calculate_dependent_values(n: Network) -> None:
    """
    Calculate per unit impedances and append voltages to lines and shunt
    impedances.
    """
    apply_line_types(n)
    apply_transformer_types(n)

    n.lines["v_nom"] = n.lines.bus0.map(n.buses.v_nom)
    n.lines.loc[n.lines.carrier == "", "carrier"] = n.lines.bus0.map(n.buses.carrier)

    n.lines["x_pu"] = n.lines.x / (n.lines.v_nom**2)
    n.lines["r_pu"] = n.lines.r / (n.lines.v_nom**2)
    n.lines["b_pu"] = n.lines.b * n.lines.v_nom**2
    n.lines["g_pu"] = n.lines.g * n.lines.v_nom**2
    n.lines["x_pu_eff"] = n.lines["x_pu"]
    n.lines["r_pu_eff"] = n.lines["r_pu"]

    # convert transformer impedances from base power s_nom to base = 1 MVA
    n.transformers["x_pu"] = n.transformers.x / n.transformers.s_nom
    n.transformers["r_pu"] = n.transformers.r / n.transformers.s_nom
    n.transformers["b_pu"] = n.transformers.b * n.transformers.s_nom
    n.transformers["g_pu"] = n.transformers.g * n.transformers.s_nom
    n.transformers["x_pu_eff"] = n.transformers["x_pu"] * n.transformers["tap_ratio"]
    n.transformers["r_pu_eff"] = n.transformers["r_pu"] * n.transformers["tap_ratio"]

    apply_transformer_t_model(n)

    n.shunt_impedances["v_nom"] = n.shunt_impedances["bus"].map(n.buses.v_nom)
    n.shunt_impedances["b_pu"] = n.shunt_impedances.b * n.shunt_impedances.v_nom**2
    n.shunt_impedances["g_pu"] = n.shunt_impedances.g * n.shunt_impedances.v_nom**2

    n.links.loc[n.links.carrier == "", "carrier"] = n.links.bus0.map(n.buses.carrier)

    n.stores.loc[n.stores.carrier == "", "carrier"] = n.stores.bus.map(n.buses.carrier)

    update_linkports_component_attrs(n)


def find_slack_bus(sub_network: SubNetwork) -> None:
    """
    Find the slack bus in a connected sub-network.
    """
    gens = sub_network.generators()

    if len(gens) == 0:
        #        logger.warning("No generators in sub-network {}, better hope power is already balanced".format(sub_network.name))
        sub_network.slack_generator = None
        sub_network.slack_bus = sub_network.buses_i()[0]

    else:
        slacks = gens[gens.control == "Slack"].index

        if len(slacks) == 0:
            sub_network.slack_generator = gens.index[0]
            sub_network.n.generators.loc[sub_network.slack_generator, "control"] = (
                "Slack"
            )
            logger.debug(
                f"No slack generator found in sub-network {sub_network.name}, using {sub_network.slack_generator} as the slack generator"
            )

        elif len(slacks) == 1:
            sub_network.slack_generator = slacks[0]
        else:
            sub_network.slack_generator = slacks[0]
            sub_network.n.generators.loc[slacks[1:], "control"] = "PV"
            logger.debug(
                f"More than one slack generator found in sub-network {sub_network.name}, using {sub_network.slack_generator} as the slack generator"
            )

        sub_network.slack_bus = gens.bus[sub_network.slack_generator]

    # also put it into the dataframe
    sub_network.n.sub_networks.at[sub_network.name, "slack_bus"] = sub_network.slack_bus

    logger.debug(
        f"Slack bus for sub-network {sub_network.name} is {sub_network.slack_bus}"
    )


def find_bus_controls(sub_network: SubNetwork) -> None:
    """
    Find slack and all PV and PQ buses for a sub_network.

    This function also fixes sub_network.buses_o, a DataFrame ordered by
    control type.
    """
    n = sub_network.n

    find_slack_bus(sub_network)

    gens = sub_network.generators()
    buses_i = sub_network.buses_i()

    # default bus control is PQ
    n.buses.loc[buses_i, "control"] = "PQ"

    # find all buses with one or more gens with PV
    pvs = gens[gens.control == "PV"].index.to_series()
    if len(pvs) > 0:
        pvs = pvs.groupby(gens.bus).first()
        n.buses.loc[pvs.index, "control"] = "PV"
        n.buses.loc[pvs.index, "generator"] = pvs

    n.buses.loc[sub_network.slack_bus, "control"] = "Slack"
    n.buses.loc[sub_network.slack_bus, "generator"] = sub_network.slack_generator

    buses_control = n.buses.loc[buses_i, "control"]
    sub_network.pvs = buses_control.index[buses_control == "PV"]
    sub_network.pqs = buses_control.index[buses_control == "PQ"]

    sub_network.pvpqs = sub_network.pvs.append(sub_network.pqs)

    # order buses
    sub_network.buses_o = sub_network.pvpqs.insert(0, sub_network.slack_bus)


def calculate_B_H(sub_network: SubNetwork, skip_pre: bool = False) -> None:
    """
    Calculate B and H matrices for AC or DC sub-networks.
    """
    n = sub_network.n

    if not skip_pre:
        calculate_dependent_values(n)
        find_bus_controls(sub_network)

    if n.sub_networks.at[sub_network.name, "carrier"] == "DC":
        attribute = "r_pu_eff"
    else:
        attribute = "x_pu_eff"

    # following leans heavily on pypower.makeBdc

    z = np.concatenate(
        [
            (c.static.loc[c.static.query("active").index, attribute]).values
            for c in sub_network.iterate_components(n.passive_branch_components)
        ]
    )
    # susceptances
    b = np.divide(1.0, z, out=np.full_like(z, np.inf), where=z != 0)

    if np.isnan(b).any():
        logger.warning(
            "Warning! Some series impedances are zero - this will cause a singularity in LPF!"
        )
    b_diag = csr_matrix((b, (r_[: len(b)], r_[: len(b)])))

    # incidence matrix
    sub_network.K = sub_network.incidence_matrix(busorder=sub_network.buses_o)

    sub_network.H = b_diag * sub_network.K.T

    # weighted Laplacian
    sub_network.B = sub_network.K * sub_network.H

    phase_shift = np.concatenate(
        [
            (
                (c.static.loc[c.static.query("active").index, "phase_shift"]).values
                * np.pi
                / 180.0
                if c.name == "Transformer"
                else np.zeros((len(c.static.query("active").index),))
            )
            for c in sub_network.iterate_components(n.passive_branch_components)
        ]
    )
    sub_network.p_branch_shift = np.multiply(-b, phase_shift, where=b != np.inf)

    sub_network.p_bus_shift = sub_network.K * sub_network.p_branch_shift


def calculate_PTDF(sub_network: SubNetwork, skip_pre: bool = False) -> None:
    """
    Calculate the Power Transfer Distribution Factor (PTDF) for sub_network.

    Sets sub_network.PTDF as a (dense) numpy array.

    Parameters
    ----------
    sub_network : pypsa.SubNetwork
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating dependent values,
        finding bus controls and computing B and H.
    """
    if not skip_pre:
        calculate_B_H(sub_network)

    # calculate inverse of B with slack removed

    n_pvpq = len(sub_network.pvpqs)
    index = np.r_[:n_pvpq]

    identity = csc_matrix((np.ones(n_pvpq), (index, index)))

    B_inverse = spsolve(csc_matrix(sub_network.B[1:, 1:]), identity)

    # exception for two-node networks, where B_inverse is a 1d array
    if issparse(B_inverse):
        B_inverse = B_inverse.toarray()
    elif B_inverse.shape == (1,):
        B_inverse = B_inverse.reshape((1, 1))

    # add back in zeroes for slack
    B_inverse = np.hstack((np.zeros((n_pvpq, 1)), B_inverse))
    B_inverse = np.vstack((np.zeros(n_pvpq + 1), B_inverse))

    sub_network.PTDF = sub_network.H * B_inverse


def calculate_Y(
    sub_network: SubNetwork, skip_pre: bool = False, active_branches_only: bool = True
) -> None:
    """
    Calculate bus admittance matrices for AC sub-networks.
    """
    if not skip_pre:
        calculate_dependent_values(sub_network.n)

    if sub_network.n.sub_networks.at[sub_network.name, "carrier"] != "AC":
        logger.warning("Non-AC networks not supported for Y!")
        return

    branches = sub_network.branches()
    buses_o = sub_network.buses_o

    if active_branches_only:
        branches = branches[branches.active]

    n = sub_network.n

    # following leans heavily on pypower.makeYbus
    # Copyright Richard Lincoln, Ray Zimmerman, BSD-style licence

    num_branches = len(branches)
    num_buses = len(buses_o)

    y_se = 1 / (branches["r_pu"] + 1.0j * branches["x_pu"])

    y_sh = branches["g_pu"] + 1.0j * branches["b_pu"]

    tau = branches["tap_ratio"].fillna(1.0)

    # catch some transformers falsely set with tau = 0 by pypower
    tau[tau == 0] = 1.0

    # define the HV tap ratios
    tau_hv = pd.Series(1.0, branches.index)
    tau_hv[branches.tap_side == 0] = tau[branches.tap_side == 0]

    # define the LV tap ratios
    tau_lv = pd.Series(1.0, branches.index)
    tau_lv[branches.tap_side == 1] = tau[branches.tap_side == 1]

    phase_shift = np.exp(1.0j * branches["phase_shift"].fillna(0.0) * np.pi / 180.0)

    # build the admittance matrix elements for each branch
    Y11 = (y_se + 0.5 * y_sh) / tau_lv**2
    Y10 = -y_se / tau_lv / tau_hv / phase_shift
    Y01 = -y_se / tau_lv / tau_hv / np.conj(phase_shift)
    Y00 = (y_se + 0.5 * y_sh) / tau_hv**2

    # bus shunt impedances
    b_sh = (
        n.shunt_impedances.b_pu.groupby(n.shunt_impedances.bus)
        .sum()
        .reindex(buses_o, fill_value=0.0)
    )
    g_sh = (
        n.shunt_impedances.g_pu.groupby(n.shunt_impedances.bus)
        .sum()
        .reindex(buses_o, fill_value=0.0)
    )
    Y_sh = g_sh + 1.0j * b_sh

    # get bus indices
    bus0 = buses_o.get_indexer(branches.bus0)
    bus1 = buses_o.get_indexer(branches.bus1)

    # connection matrices
    C0 = csr_matrix(
        (ones(num_branches), (np.arange(num_branches), bus0)), (num_branches, num_buses)
    )
    C1 = csr_matrix(
        (ones(num_branches), (np.arange(num_branches), bus1)), (num_branches, num_buses)
    )

    # build Y{0, 1} such that Y{0, 1} * V is the vector complex branch currents

    i = r_[np.arange(num_branches), np.arange(num_branches)]
    sub_network.Y0 = csr_matrix(
        (r_[Y00, Y01], (i, r_[bus0, bus1])), (num_branches, num_buses)
    )
    sub_network.Y1 = csr_matrix(
        (r_[Y10, Y11], (i, r_[bus0, bus1])), (num_branches, num_buses)
    )

    # now build bus admittance matrix
    sub_network.Y = (
        C0.T * sub_network.Y0
        + C1.T * sub_network.Y1
        + csr_matrix((Y_sh, (np.arange(num_buses), np.arange(num_buses))))
    )


def aggregate_multi_graph(sub_network: SubNetwork) -> None:
    """
    Aggregate branches between same buses and replace with a single branch with
    aggregated properties (e.g. s_nom is summed, length is averaged).
    """
    n = sub_network.n

    count = 0
    seen = []
    graph = sub_network.graph()
    for u, v in graph.edges():
        if (u, v) in seen:
            continue
        line_objs = list(graph.adj[u][v].keys())
        if len(line_objs) > 1:
            lines = n.lines.loc[[line[1] for line in line_objs]]
            attr_inv = ["x", "r"]
            attr_sum = ["s_nom", "b", "g", "s_nom_max", "s_nom_min"]
            attr_mean = ["capital_cost", "length", "terrain_factor"]

            aggregated = {attr: 1.0 / (1.0 / lines[attr]).sum() for attr in attr_inv}
            for attr in attr_sum:
                aggregated[attr] = lines[attr].sum()

            for attr in attr_mean:
                aggregated[attr] = lines[attr].mean()

            count += len(line_objs) - 1

            # remove all but first line
            for line in line_objs[1:]:
                n.remove("Line", line[1])

            rep = line_objs[0]

            for key, value in aggregated.items():
                setattr(rep, key, value)

            seen.append((u, v))

    logger.info(
        "Removed %d excess lines from sub-network %s and replaced with aggregated lines",
        count,
        sub_network.name,
    )


def find_tree(sub_network: SubNetwork, weight: str = "x_pu") -> None:
    """
    Get the spanning tree of the graph, choose the node with the highest degree
    as a central "tree slack" and then see for each branch which paths from the
    slack to each node go through the branch.
    """
    branches_bus0 = sub_network.branches()["bus0"]
    branches_i = branches_bus0.index
    buses_i = sub_network.buses_i()

    graph = sub_network.graph(weight=weight, inf_weight=1.0)
    sub_network.tree = nx.minimum_spanning_tree(graph)

    # find bus with highest degree to use as slack
    tree_slack_bus, slack_degree = max(sub_network.tree.degree(), key=itemgetter(1))
    logger.debug("Tree slack bus is %s with degree %d.", tree_slack_bus, slack_degree)

    # determine which buses are supplied in tree through branch from slack

    # matrix to store tree structure
    sub_network.T = dok_matrix((len(branches_i), len(buses_i)))

    for j, bus in enumerate(buses_i):
        path = nx.shortest_path(sub_network.tree, bus, tree_slack_bus)
        for i in range(len(path) - 1):
            branch = next(iter(graph[path[i]][path[i + 1]].keys()))
            branch_i = branches_i.get_loc(branch)
            sign = +1 if branches_bus0.iat[branch_i] == path[i] else -1
            sub_network.T[branch_i, j] = sign


def find_cycles(sub_network: SubNetwork, weight: str = "x_pu") -> None:
    """
    Find all cycles in the sub_network and record them in sub_network.C.

    networkx collects the cycles with more than 2 edges; then the 2-edge
    cycles from the MultiGraph must be collected separately (for cases
    where there are multiple lines between the same pairs of buses).

    Cycles with infinite impedance are skipped.
    """
    branches_bus0 = sub_network.branches()["bus0"]
    branches_i = branches_bus0.index

    # reduce to a non-multi-graph for cycles with > 2 edges
    mgraph = sub_network.graph(weight=weight, inf_weight=False)
    graph = nx.Graph(mgraph)

    cycles = nx.cycle_basis(graph)

    # number of 2-edge cycles
    num_multi = len(mgraph.edges()) - len(graph.edges())

    sub_network.C = dok_matrix((len(branches_bus0), len(cycles) + num_multi))

    for j, cycle in enumerate(cycles):
        for i in range(len(cycle)):
            branch = next(iter(mgraph[cycle[i]][cycle[(i + 1) % len(cycle)]].keys()))
            branch_i = branches_i.get_loc(branch)
            sign = +1 if branches_bus0.iat[branch_i] == cycle[i] else -1
            sub_network.C[branch_i, j] += sign

    # counter for multis
    c = len(cycles)

    # add multi-graph 2-edge cycles for multiple branches between same pairs of buses
    for u, v in graph.edges():
        bs = list(mgraph[u][v].keys())
        if len(bs) > 1:
            first = bs[0]
            first_i = branches_i.get_loc(first)
            for b in bs[1:]:
                b_i = branches_i.get_loc(b)
                sign = (
                    -1 if branches_bus0.iat[b_i] == branches_bus0.iat[first_i] else +1
                )
                sub_network.C[first_i, c] = 1
                sub_network.C[b_i, c] = sign
                c += 1


def sub_network_lpf(
    sub_network: SubNetwork,
    snapshots: Sequence | None = None,
    skip_pre: bool = False,
) -> None:
    """
    Linear power flow for connected sub-network.

    Parameters
    ----------
    snapshots : list-like|single snapshot
        A subset or an elements of n.snapshots on which to run
        the power flow, defaults to n.snapshots
    skip_pre : bool, default False
        Skip the preliminary steps of computing topology, calculating
        dependent values and finding bus controls.

    Returns
    -------
    None
    """
    sns = as_index(sub_network.n, snapshots, "snapshots", "snapshot")
    logger.info(
        "Performing linear load-flow on %s sub-network %s for snapshot(s) %s",
        sub_network.n.sub_networks.at[sub_network.name, "carrier"],
        sub_network,
        snapshots,
    )

    n = sub_network.n

    if not skip_pre:
        calculate_dependent_values(n)
        find_bus_controls(sub_network)
        _allocate_pf_outputs(n, linear=True)

    # get indices for the components on this sub-network
    buses_o = sub_network.buses_o
    branches_i = sub_network.branches_i(active_only=True)

    # allow all shunt impedances to dispatch as set
    shunt_impedances_i = sub_network.shunt_impedances_i()
    n.shunt_impedances_t.p.loc[sns, shunt_impedances_i] = n.shunt_impedances.g_pu.loc[
        shunt_impedances_i
    ].values

    # allow all one ports to dispatch as set
    for c in sub_network.iterate_components(n.controllable_one_port_components):
        c_p_set = get_as_dense(n, c.name, "p_set", sns, c.static.query("active").index)
        n.dynamic(c.name).p.loc[sns, c.static.query("active").index] = c_p_set

    # set the power injection at each node
    n.buses_t.p.loc[sns, buses_o] = sum(
        [
            (
                (
                    c.dynamic.p.loc[sns, c.static.query("active").index]
                    * c.static.loc[c.static.query("active").index, "sign"]
                )
                .T.groupby(c.static.loc[c.static.query("active").index, "bus"])
                .sum()
                .T.reindex(columns=buses_o, fill_value=0.0)
            )
            for c in sub_network.iterate_components(n.one_port_components)
        ]
        + [
            -c.dynamic[f"p{str(i)}"]
            .loc[sns]
            .T.groupby(c.static[f"bus{str(i)}"])
            .sum()
            .T.reindex(columns=buses_o, fill_value=0)
            for c in n.iterate_components(n.controllable_branch_components)
            for i in [int(col[3:]) for col in c.static.columns if col[:3] == "bus"]
        ]
    )

    if not skip_pre and len(branches_i) > 0:
        calculate_B_H(sub_network, skip_pre=True)

    v_diff = np.zeros((len(sns), len(buses_o)))
    if len(branches_i) > 0:
        p = n.buses_t["p"].loc[sns, buses_o].values - sub_network.p_bus_shift
        v_diff[:, 1:] = spsolve(sub_network.B[1:, 1:], p[:, 1:].T).T
        flows = (
            pd.DataFrame(v_diff * sub_network.H.T, columns=branches_i, index=sns)
            + sub_network.p_branch_shift
        )

        for c in sub_network.iterate_components(n.passive_branch_components):
            f = flows.loc[:, c.name]
            n.dynamic(c.name).p0.loc[sns, f.columns] = f
            n.dynamic(c.name).p1.loc[sns, f.columns] = -f

    if n.sub_networks.at[sub_network.name, "carrier"] == "DC":
        n.buses_t.v_mag_pu.loc[sns, buses_o] = 1 + v_diff
        n.buses_t.v_ang.loc[sns, buses_o] = 0.0
    else:
        n.buses_t.v_ang.loc[sns, buses_o] = v_diff
        n.buses_t.v_mag_pu.loc[sns, buses_o] = 1.0

    # set slack bus power to pick up remained
    slack_adjustment = (
        -n.buses_t.p.loc[sns, buses_o[1:]].sum(axis=1).fillna(0.0)
        - n.buses_t.p.loc[sns, buses_o[0]]
    )
    n.buses_t.p.loc[sns, buses_o[0]] += slack_adjustment

    # let slack generator take up the slack
    if sub_network.slack_generator is not None:
        n.generators_t.p.loc[sns, sub_network.slack_generator] += (
            slack_adjustment
        )  # fmt: skip


@deprecated_common_kwargs
def network_batch_lpf(n: Network, snapshots: Sequence | None = None) -> None:
    """
    Batched linear power flow with numpy.dot for several snapshots.
    """
    raise NotImplementedError("Batch linear power flow not supported yet.")
