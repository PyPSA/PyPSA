############
Optimisation
############

Optimization functions which can be called within a :class:`pypsa.Network` via
``n.optimize`` or ``n.optimization.func``. For example ``n.optimize.create_model()``.

Statistic methods
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _source/

    ~pypsa.optimization.optimize.OptimizationAccessor.__call__
    ~pypsa.optimization.optimize.OptimizationAccessor.create_model
    ~pypsa.optimization.optimize.OptimizationAccessor.solve_model
    ~pypsa.optimization.optimize.OptimizationAccessor.assign_solution
    ~pypsa.optimization.optimize.OptimizationAccessor.assign_duals
    ~pypsa.optimization.optimize.OptimizationAccessor.post_processing
    ~pypsa.optimization.optimize.OptimizationAccessor.optimize_transmission_expansion_iteratively
    ~pypsa.optimization.optimize.OptimizationAccessor.optimize_security_constrained
    ~pypsa.optimization.optimize.OptimizationAccessor.optimize_with_rolling_horizon 
    ~pypsa.optimization.optimize.OptimizationAccessor.optimize_mga
    ~pypsa.optimization.optimize.OptimizationAccessor.optimize_and_run_non_linear_powerflow
    ~pypsa.optimization.optimize.OptimizationAccessor.fix_optimal_capacities 
    ~pypsa.optimization.optimize.OptimizationAccessor.fix_optimal_dispatch 
    ~pypsa.optimization.optimize.OptimizationAccessor.add_load_shedding 
