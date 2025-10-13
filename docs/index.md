<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

![PyPSA Header Logo](assets/logo/logo-primary-light.svg#only-light)
![PyPSA Header Logo](assets/logo/logo-primary-dark.svg#only-dark)

# PyPSA: Python for Power System Analysis

[![PyPI version](https://img.shields.io/pypi/v/pypsa.svg)](https://pypi.python.org/pypi/pypsa)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pypsa.svg)](https://anaconda.org/conda-forge/pypsa)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FPyPSA%2FPyPSA%2Fmaster%2Fpyproject.toml)
[![REUSE status](https://api.reuse.software/badge/github.com/pypsa/pypsa)](https://api.reuse.software/info/github.com/pypsa/pypsa)
[![License](https://img.shields.io/pypi/l/pypsa.svg)](https://github.com/PyPSA/pypsa?tab=MIT-1-ov-file)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3946412.svg)](https://doi.org/10.5281/zenodo.3946412)
[![Discord](https://img.shields.io/discord/911692131440148490?logo=discord)](https://discord.gg/AnuJBk23FU)

PyPSA stands for **Python for Power System Analysis**. It is pronounced "pipes-ah" /ˈpaɪpsə/.

PyPSA is an open-source Python framework for optimising and simulating modern
power and energy systems that include features such as conventional generators
with unit commitment, variable wind and solar generation, hydro-electricity,
inter-temporal storage, coupling to other energy sectors, elastic demands, and
linearised power flow with loss approximations in DC and AC networks. PyPSA is
designed to scale well with large networks and long time series. It is made for
researchers, planners and utilities with basic coding aptitude who need a fast,
easy-to-use and transparent tool for power and energy system analysis.

Check out the [:octicons-gear-16: Features](features.md) for more information on the functionality.

!!! note

    PyPSA has many contributors, with the maintenance led by the [Department of Digital Transformation in
    Energy Systems](https://tu.berlin/en/ensys) at the [Technical University of
    Berlin](https://www.tu.berlin).  The project is currently supported by the 
    [German Research Foundation](https://www.dfg.de/en/) (grant number [`528775426`](https://gepris.dfg.de/gepris/projekt/528775426)).    
    Previous versions were developed at the [Karlsruhe
    Institute of Technology](http://www.kit.edu/english/index.php) funded by the
    [Helmholtz Association](https://www.helmholtz.de/en/), and
    at [FIAS](https://fias.uni-frankfurt.de/) funded by the [German Federal
    Ministry for Education and Research (BMBF)](https://www.bmbf.de/bmbf/en/).

    
## Quick Links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Getting started**

    ---

    [:octicons-arrow-right-24: Installation](installation.md)

    [:octicons-arrow-right-24: Quick Start](examples/example-1.ipynb)
    
-   :material-view-list:{ .lg .middle } **Overview**

    ---

    [:octicons-arrow-right-24: Features](features.md)

    [:octicons-arrow-right-24: Frequently Asked Questions](faq.md)

-   :material-new-box:{ .lg .middle } **Release Notes**

    ---

    Check out the latest features, bug fixes and improvements in the release notes.

    [:octicons-arrow-right-24: What's new](release-notes.md)

-   :material-scale-balance:{ .lg .middle } **Open Source, MIT**

    ---

    PyPSA is licensed under MIT and available on [GitHub](https://www.github.com/PyPSA/PyPSA).

    [:octicons-arrow-right-24: License](license.md)

</div>

## Sections

<div class="grid cards" markdown>

-   :material-bookshelf:{ .lg .middle } **User Guide**

    ---

    Find a detailed description of the PyPSA **design and architecture**, how to setup different **optimization problems** and how to use the **utility functions** here.
    

    [:octicons-arrow-right-24: User Guide](user-guide.md)

-   :material-notebook-multiple:{ .lg .middle } **Examples**

    ---
    
    Many different examples from setting up a **basic toy model** to **sector coupling** or **security-constrained optimization** can be found here.

    [:octicons-arrow-right-24: Examples](examples.md)

-   :octicons-code-16:{ .lg .middle } **API Reference**

    ---

    The API Reference is generated from the docstrings in the code. It contains a detailed description of **all classes and functions**, their parameters and how to use them.

    [:octicons-arrow-right-24: API Reference](network.md)

-   :fontawesome-solid-users:{ .lg .middle } **Contributing**

    ---

    PyPSA is an **open source project** and we welcome any contributions to **keep the project alive**. Find out how to **contribute here**. You don't need to be a developer to contribute.

    [:octicons-arrow-right-24: Contributing](contributing.md)

</div>

