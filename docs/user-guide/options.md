<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

PyPSA has an options system that allows users to customise its global behaviour.

## Setting options

Options can be set by assigning values to attributes of the `pypsa.options` object. For example, to change the default solver used in optimizations to `"gurobi"` (see [Parameters options](#parameters-options)), run:
``` py
>>> pypsa.options.params.optimize.solver_name = "gurobi" # doctest: +SKIP
```

This will set the option **globally** for the current Python session.

You can also use a context manager to set options temporarily within a `with` block via [`pypsa.option_context`][pypsa.option_context]. This is useful if you want to change an option for a particular section of your code without altering the rest of it. For example:
``` py
>>> with pypsa.option_context(params.optimize.solver_name="gurobi"):
...     n.optimize()  # "gurobi" as the solver # doctest: +SKIP
...
>>> n.optimize()  # "highs" as the solver # doctest: +SKIP
```

To reset options to their default values, simply remove the assignment or use [pypsa.reset_option][].

Instead of setting the value and getting the value via attribute access, you can also use  [pypsa.set_option][] and [pypsa.get_option][].


## List of available options
Options are grouped into categories and sub-categories. You can run the [`describe()`][pypsa.options.describe] function on any category or sub-category to get a list of available options and their current values. To list all options just run
``` py
>>> pypsa.options.describe() # doctest: +ELLIPSIS
PyPSA Options
=============
...
```

!!! info

    The options system was recently introduced and more options will be added in future versions. If you have any suggestions for useful options, please open an issue on [GitHub](https://github.com/PyPSA/PyPSA/issues).

### General options

``` py
>>> pypsa.options.general.describe()
PyPSA Options
=============
allow_network_requests:
    Default: True
    Description: Allow PyPSA to make network requests. When False, all network requests
    (such as checking for version updates) are disabled. This may be needed
    in restricted environments, offline usage, or for security/privacy reasons.
    This only controls PyPSA's own network requests, dependencies may still
    make network requests independently.
```

### Parameters options

The `params` category allows to change the default parameters used in some PyPSA functions. For example the default solver used in optimizations is `highs`. When running [`n.optimize()`][pypsa.optimization.OptimizationAccessor.__call__] you would need to pass `solver_name="gurobi"` to use a different solver. Instead you can also
change the default globally by setting the option `pypsa.options.params.optimize.solver_name="gurobi"` as shown above.

``` py
>>> pypsa.options.params.describe()
PyPSA Options
=============
statistics.nice_names:
    Default: True
    Description: Default value for the 'nice_names' parameter in statistics module.
statistics.drop_zero:
    Default: True
    Description: Default value for the 'drop_zero' parameter in statistics module.
statistics.round:
    Default: 5
    Description: Default value for the 'round' parameter in statistics module.
add.return_names:
    Default: False
    Description: Default value for the 'return_names' parameter in Network.add method.
    If True, the add method returns the names of added components.
    If False, it returns None.
optimize.model_kwargs:
    Default: {}
    Description: Default value for the 'model_kwargs' parameter in optimization module.
optimize.solver_name:
    Default: highs
    Description: Default value for the 'solver_name' parameter in optimization module.
optimize.solver_options:
    Default: {}
    Description: Default value for the 'solver_options' parameter in optimization module.
```

### Warnings options

Turn off or on certain warnings.

``` py
>>> pypsa.options.warnings.describe()
PyPSA Options
=============
components_store_iter:
    Default: True
    Description: If False, suppresses the deprecation warning when iterating over components.
attribute_typos:
    Default: True
    Description: If False, suppresses warnings about potential typos in component attribute names. Note: warnings about unintended attributes (standard attributes for other components) will still be shown.
```

### API

Make changes to the PyPSA API. See <!-- md:guide components.md#new-components-class-api -->.

``` py
>>> pypsa.options.api.describe()  # doctest: +NORMALIZE_WHITESPACE
PyPSA Options
=============
new_components_api:
    Default: False
    Description: Activate the new components API, which replaces the static components data access
    	with the more flexible components class. This will just change the api and not any
    	functionality. Components class features are always available.
    	See `https://go.pypsa.org/new-components-api` for more details.
```


### Debug options

Options for developers to help with debugging.

``` py
>>> pypsa.options.debug.describe()
PyPSA Options
=============
runtime_verification:
    Default: False
    Description: Enable runtime verification of PyPSA's internal state. This is useful
        for debugging and development purposes. This will lead to overhead in
        performance and should not be used in production.
```
