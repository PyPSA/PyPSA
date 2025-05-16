################
Troubleshooting
################

Minimum Working Examples (MWEs)
====================================================

If you are experiencing problems with PyPSA, please create a MWE and share it when asking for help.

A MWE is a small example that reproduces the problem you are experiencing.
It should be as small as possible, but still reproduce the problem.
Often, when creating a MWE, you will find the problem yourself,
and if not it will be much easier for others to help you.

For details on what a MWE is and how to create one, see this `Stack Overflow post <https://stackoverflow.com/help/minimal-reproducible-example>`_.


Library dependency issues
=========================

If you are experiencing problems with PyPSA or with the importing of
the libraries on which PyPSA depends, please first check that you are
working with the latest versions of all packages.

See :ref:`upgrading-pypsa`.


Consistency check on network
============================

A consistency check can be performed using the function :py:meth:`pypsa.Network.consistency_check`.


Problems with power flow convergence
====================================

If your ``n.pf()`` is not converging there are two possible reasons:

* The problem you have defined is not solvable (e.g. because in
  reality you would have a voltage collapse)
* The problem is solvable, but there are numerical instabilities in
  the solving algorithm (e.g. Newton-Raphson is known not to
  converge even for solvable problems; or the flat solution PyPSA
  uses as an initial guess is too far from the correction solution
  because of transformer phase-shifts)

There are some steps you can take to distinguish these two cases:

* Check the units you have used to define the problem are correct.
  If your units are out by a factor 1000
  (e.g. using kW instead of MW) don't be surprised if your problem is
  no longer solvable.
* Check with a linear power flow ``n.lpf()`` that all voltage
  angles differences across branches are less than 40 degrees. You can do this with the following code:

.. code:: python

   import pandas as pd
   import numpy as np

   now = n.snapshots[0]

   angle_diff = pd.Series(n.buses_t.v_ang.loc[now,n.lines.bus0].values -
                          n.buses_t.v_ang.loc[now,n.lines.bus1].values,
                          index=n.lines.index)

   (angle_diff*180/np.pi).describe()

* You can seed the non-linear power flow initial guess with the
  voltage angles from the linear power flow. This is advisable if you
  have transformers with phase shifts in the network, which lead to
  solutions far away from the flat initial guess of all voltage angles
  being zero. To seed the problem activate the ``use_seed`` switch:

.. code:: python

   n.lpf()
   n.pf(use_seed=True)


* Reduce all power values ``p_set`` and ``q_set`` of generators and
  loads to a fraction, e.g. 10%, solve the load flow and use it as a
  seed for the power at 20%, iteratively up to 100%.


Problems with optimisation convergence
======================================

If your ``n.optimize()`` is not converging here are some suggestions:

* Very small non-zero values, for example in
  ``n.generators_t.p_max_pu`` can confuse the
  optimiser. Consider e.g. removing values smaller than 0.001 with
  ``numpy.clip``.
* Open source solvers like GLPK and CBC struggle with large
  problems. Consider switching to a commercial solver like Gurobi,
  CPLEX or Xpress.
* Use the interior point or barrier method, and stop it from crossing
  over to the simplex algorithm once it is close to the solution. This
  will provide a good approximate solution. The settings for this
  behaviour in CPLEX and Gurobi can be found in the `PyPSA-Eur
  config.yaml
  <https://github.com/PyPSA/pypsa-eur/blob/master/config.default.yaml>`_.


Pitfalls & Gotchas
==================

Some attributes are generated dynamically and are therefore only
copies. If you change data in them, this will not update the original
data. They are all defined as functions to make this clear.

For example:

* ``n.branches()`` returns a DataFrame which is a concatenation
  of ``n.lines`` and ``n.transformers``
* ``sub_network.generators()`` returns a DataFrame consisting of
  generators in ``sub_network``


Reporting bugs/issues
=====================

If you found a bug, raise it as an issue on the `PyPSA Github Issues page
<https://github.com/PyPSA/PyPSA/issues>`_ or prepare a pull request.
