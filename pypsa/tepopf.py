from .opf import (define_generator_variables_constraints,
                  define_storage_variables_constraints,
                  define_store_variables_constraints,
                  define_link_flows,
                  define_passive_branch_flows,
                  define_passive_branch_flows_with_angles,
                  define_passive_branch_flows_with_kirchhoff,
                  define_sub_network_cycle_constraints,
                  define_passive_branch_constraints,
                  define_nodal_balances,
                  define_nodal_balance_constraints, # needs modification
                  define_sub_network_balance_constraints,
                  define_global_constraints,
                  define_linear_objective, # needs modification
                  extract_optimisation_results, # may need modification
                  network_lopf_prepare_solver,
                  network_lopf_solve)

# preprocessing:
def infer_candidates(candidate_investments=True):
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """

    if candidate_investments:
        candidate_lines_to_investment()

    return 0


def candidate_lines_to_investment():
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """
    pass


def bigm(formulation):
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """

    if formulation == "angles":
        m = bigm_for_angles()
    elif formulation == "kirchhoff":
        m = bigm_for_kirchhoff()
    else:
        raise NotImplementedError("Calculating Big-M for formulation `{}` not implemented.\
                                   Try `angles` or `kirchhoff`.")

    return m

def bigm_for_angles():
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """

    pass

def bigm_for_kirchhoff():
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """

    pass

# formulation
def define_integer_branch_extension_variables(network, snapshots):
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """

    pass

def define_integer_passive_branch_flows(network, snapshots):
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """
    
    if formulation == "angles":
        define_integer_passive_branch_flows_with_angles(network, snapshots)
    elif formulation == "kirchhoff":
        define_integer_passive_branch_flows_with_kirchhoff(network, snapshots)

def define_integer_passive_branch_flows_with_angles(network, snapshots):
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """

    pass

def define_integer_passive_branch_flows_with_kirchhoff(network, snapshots):
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """

    pass

def define_integer_passive_branch_constraints(network, snapshots): 
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """

    pass


def network_teplopf_build_model():
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """

    pass


def network_teplopf():
    """
    Description

    Parameters
    ----------

    Returns
    -------
    None
    """

    pass
