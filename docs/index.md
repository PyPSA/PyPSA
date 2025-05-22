---
hide:
  - footer
---
![PyPSA Header Logo](assets/logo/logo-primary-light.svg#only-light)
![PyPSA Header Logo](assets/logo/logo-primary-dark.svg#only-dark)

# PyPSA: Python for Power System Analysis

[![PyPI version](https://img.shields.io/pypi/v/pypsa.svg)](https://pypi.python.org/pypi/pypsa)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/pypsa.svg)](https://anaconda.org/conda-forge/pypsa)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FPyPSA%2FPyPSA%2Fmaster%2Fpyproject.toml)
![Static Badge](https://img.shields.io/badge/latest-%23d10949?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTG9nbyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB2ZXJzaW9uPSIxLjEiIHZpZXdCb3g9IjAgMCA0NTAgNDUwIj4KICA8IS0tIEdlbmVyYXRvcjogQWRvYmUgSWxsdXN0cmF0b3IgMjkuNC4wLCBTVkcgRXhwb3J0IFBsdWctSW4gLiBTVkcgVmVyc2lvbjogMi4xLjAgQnVpbGQgMTUyKSAgLS0%2BCiAgPGRlZnM%2BCiAgICA8c3R5bGU%2BCiAgICAgIC5zdDAgewogICAgICAgIGZpbGw6ICMyOTkzYjU7CiAgICAgIH0KCiAgICAgIC5zdDEgewogICAgICAgIGZpbGw6ICMwYTg3NTQ7CiAgICAgIH0KCiAgICAgIC5zdDIgewogICAgICAgIGZpbGw6ICNmZmY7CiAgICAgIH0KCiAgICAgIC5zdDMgewogICAgICAgIGZpbGw6ICNkMTBhNDk7CiAgICAgIH0KCiAgICAgIC5zdDQgewogICAgICAgIGZpbGw6ICNmZmJmMDA7CiAgICAgIH0KCiAgICAgIC5zdDUgewogICAgICAgIGRpc3BsYXk6IG5vbmU7CiAgICAgIH0KICAgIDwvc3R5bGU%2BCiAgPC9kZWZzPgogIDxnIGlkPSJMZWZ0X1RyaWFuZ2xlIj4KICAgIDxwYXRoIGNsYXNzPSJzdDMiIGQ9Ik0yMTkuMzgsMTQyLjQ5bC0xMTEuNTQsMTg5LjYyYy05Ljg0LTEzLjkyLTI1LjYzLTIzLjM1LTQzLjY3LTI0LjU5bDExMi4wMi0xOTAuNDRjOS41OSwxNC4xMywyNS4yMiwyMy44Myw0My4xOCwyNS40MWgwWiIvPgogIDwvZz4KICA8ZyBpZD0iUmlnaHRfVHJpYW5nbGUiPgogICAgPHBhdGggY2xhc3M9InN0MyIgZD0iTTM4NC43MSwzMDcuMTFjLTEzLjE5LDEuMTctMjYuMDQsNi43OS0zNi4xNCwxNi44OS0yLjY2LDIuNjYtNS4wMSw1LjUxLTcuMDQsOC41MWwtMTExLjc4LTE5MC4wMmMxNy45Ni0xLjU3LDMzLjU5LTExLjI4LDQzLjE4LTI1LjQxbDExMS43NywxOTAuMDNoMFoiLz4KICA8L2c%2BCiAgPGcgaWQ9IkJvdHRvbV9UcmlhbmdsZSI%2BCiAgICA8cGF0aCBjbGFzcz0ic3QzIiBkPSJNMzM3LjA4LDM5MC4zN0gxMTMuMTZjMy40Ny03LjQ2LDUuMzktMTUuNzcsNS4zOS0yNC41NHMtMi4xMS0xNy44My01Ljg3LTI1LjU2aDIyNC4zOGMtNy41LDE1LjgxLTcuNSwzNC4yOSwwLDUwLjFoMFoiLz4KICA8L2c%2BCiAgPGcgaWQ9IlRvcF9NYXNrIiBjbGFzcz0ic3Q1Ij4KICAgIDxwYXRoIGNsYXNzPSJzdDIiIGQ9Ik0yODMuMjUsODQuMjZjMCwxMi4yMS0zLjczLDIzLjU1LTEwLjEyLDMyLjk0LTkuNjMsMTQuMTgtMjUuMzIsMjMuOTItNDMuMzUsMjUuNTEtMS43Mi4xNS0zLjQ1LjIzLTUuMjEuMjNzLTMuNDktLjA4LTUuMjEtLjIzYy0xOC4wMy0xLjU4LTMzLjcyLTExLjMyLTQzLjM1LTI1LjUxLTYuMzktOS4zOS0xMC4xMi0yMC43My0xMC4xMi0zMi45NCwwLTMyLjQsMjYuMjctNTguNjcsNTguNjctNTguNjdzNTguNjcsMjYuMjcsNTguNjcsNTguNjdoLjAyWiIvPgogIDwvZz4KICA8ZyBpZD0iTGVmdF9NYXNrIiBjbGFzcz0ic3Q1Ij4KICAgIDxwYXRoIGNsYXNzPSJzdDIiIGQ9Ik0xMTguNzgsMzY1LjMyYzAsOC44LTEuOTQsMTcuMTQtNS40MSwyNC42My05LjMsMjAuMS0yOS42NiwzNC4wNC01My4yNiwzNC4wNC0zMi40LDAtNTguNjgtMjYuMjctNTguNjgtNTguNjdzMjYuMjctNTguNjcsNTguNjctNTguNjdjMS4zNywwLDIuNzMuMDUsNC4wOC4xNCwxOC4xMSwxLjI0LDMzLjk2LDEwLjcsNDMuODQsMjQuNjgsMS44NCwyLjU4LDMuNDYsNS4zMiw0Ljg2LDguMiwzLjc3LDcuNzUsNS44OSwxNi40NSw1Ljg5LDI1LjY2aC4wMVoiLz4KICA8L2c%2BCiAgPGcgaWQ9IlJpZ2h0X01hc2siIGNsYXNzPSJzdDUiPgogICAgPHBhdGggY2xhc3M9InN0MiIgZD0iTTQzMS4zNyw0MDYuODFjLTIyLjkyLDIyLjkyLTYwLjA3LDIyLjkyLTgyLjk3LDAtNC45LTQuOS04Ljc0LTEwLjQ0LTExLjU0LTE2LjM1LTcuNTMtMTUuODgtNy41My0zNC40MywwLTUwLjI5LDEuMjctMi42OCwyLjc3LTUuMyw0LjQ4LTcuOCwyLjA0LTMuMDEsNC40LTUuODgsNy4wNy04LjU0LDEwLjEzLTEwLjEzLDIzLjA0LTE1Ljc5LDM2LjI4LTE2Ljk2LDE2LjctMS40OCwzMy45MSw0LjE3LDQ2LjcsMTYuOTYsMjIuOTIsMjIuOTEsMjIuOTIsNjAuMDYsMCw4Mi45N2gtLjAyWiIvPgogIDwvZz4KICA8cGF0aCBpZD0iUmlnaHRfQ2lyY2xlIiBjbGFzcz0ic3QwIiBkPSJNMzg5Ljg5LDQxNS40M2MxMy4yOSwwLDI2LjAzLTUuMjgsMzUuNDMtMTQuNjcsOS4zOS05LjM5LDE0LjY3LTIyLjEzLDE0LjY3LTM1LjQzcy01LjI4LTI2LjAzLTE0LjY3LTM1LjQzYy05LjM5LTkuMzktMjIuMTMtMTQuNjctMzUuNDMtMTQuNjdzLTI2LjAzLDUuMjgtMzUuNDMsMTQuNjdjLTkuNCw5LjM5LTE0LjY3LDIyLjEzLTE0LjY3LDM1LjQzczUuMjgsMjYuMDMsMTQuNjcsMzUuNDNjOS4zOSw5LjM5LDIyLjEzLDE0LjY3LDM1LjQzLDE0LjY3aDBaIi8%2BCiAgPHBhdGggaWQ9IlRvcF9DaXJjbGUiIGNsYXNzPSJzdDQiIGQ9Ik0yMjQuNTcsMTM0LjM3YzEzLjI5LDAsMjYuMDMtNS4yOCwzNS40My0xNC42Nyw5LjM5LTkuMzksMTQuNjctMjIuMTMsMTQuNjctMzUuNDNzLTUuMjgtMjYuMDMtMTQuNjctMzUuNDNjLTkuMzktOS4zOS0yMi4xMy0xNC42Ny0zNS40My0xNC42N3MtMjYuMDMsNS4yOC0zNS40MywxNC42N2MtOS4zOSw5LjM5LTE0LjY3LDIyLjEzLTE0LjY3LDM1LjQzczUuMjgsMjYuMDMsMTQuNjcsMzUuNDNjOS4zOSw5LjM5LDIyLjEzLDE0LjY3LDM1LjQzLDE0LjY3aDBaIi8%2BCiAgPHBhdGggaWQ9IkxlZnRfQ2lyY2xlIiBjbGFzcz0ic3QxIiBkPSJNNjAuMTEsNDE1LjQzYzEzLjI5LDAsMjYuMDMtNS4yOCwzNS40My0xNC42Nyw5LjM5LTkuMzksMTQuNjctMjIuMTMsMTQuNjctMzUuNDNzLTUuMjgtMjYuMDMtMTQuNjctMzUuNDNjLTkuMzktOS4zOS0yMi4xMy0xNC42Ny0zNS40My0xNC42N3MtMjYuMDMsNS4yOC0zNS40MywxNC42N2MtOS4zOSw5LjM5LTE0LjY3LDIyLjEzLTE0LjY3LDM1LjQzczUuMjgsMjYuMDMsMTQuNjcsMzUuNDNjOS4zOSw5LjM5LDIyLjEzLDE0LjY3LDM1LjQzLDE0LjY3WiIvPgo8L3N2Zz4%3D&label=PyPSA&labelColor=%23293036&link=https%3A%2F%2Fpypsa.readthedocs.io)
[![Tests](https://github.com/PyPSA/PyPSA/actions/workflows/test.yml/badge.svg)](https://github.com/PyPSA/PyPSA/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/pypsa/badge/?version=latest)](https://pypsa.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/PyPSA/PyPSA/master.svg)](https://results.pre-commit.ci/latest/github/PyPSA/PyPSA/master)
[![Code coverage](https://codecov.io/gh/PyPSA/PyPSA/branch/master/graph/badge.svg?token=kCpwJiV6Jr)](https://codecov.io/gh/PyPSA/PyPSA)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/pypi/l/pypsa.svg)](https://github.com/PyPSA/pypsa?tab=MIT-1-ov-file)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.3946412.svg)](https://doi.org/10.5281/zenodo.3946412)
[![Discord](https://img.shields.io/discord/911692131440148490?logo=discord)](https://discord.gg/AnuJBk23FU)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/PyPSA/pypsa?tab=coc-ov-file)

PyPSA stands for "Python for Power System Analysis". It is pronounced "pipes-ah".

PyPSA is an open source toolbox for simulating and optimising modern power and
energy systems that include features such as conventional generators with unit
commitment, variable wind and solar generation, storage units, coupling to other
energy sectors, and mixed alternating and direct current networks. PyPSA is
designed to scale well with large networks and long time series.

This project is maintained by the [Department of Digital Transformation in
Energy Systems](https://tub-ensys.github.io) at the [Technical University of
Berlin](https://www.tu.berlin). Previous versions were developed by the Energy
System Modelling group at the [Institute for Automation and Applied
Informatics](https://www.iai.kit.edu/english/index.php) at the [Karlsruhe
Institute of Technology](http://www.kit.edu/english/index.php) funded by the
[Helmholtz Association](https://www.helmholtz.de/en/), and by the [Renewable
Energy
Group](https://fias.uni-frankfurt.de/physics/schramm/renewable-energy-system-and-network-analysis/)
at [FIAS](https://fias.uni-frankfurt.de/) to carry out simulations for the
[CoNDyNet project](https://fias.institute/en/projects/condynet/), financed by the [German Federal
Ministry for Education and Research (BMBF)](https://www.bmbf.de/bmbf/en/)
as part of the [Stromnetze Research
Initiative](http://forschung-stromnetze.info/projekte/grundlagen-und-konzepte-fuer-effiziente-dezentrale-stromnetze/).


## Quick Links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Getting started**

    ---

    [:octicons-arrow-right-24: Installation](installation.md)

    [:octicons-arrow-right-24: Quick Start](quick-start.md)
    
    [:octicons-arrow-right-24: First Time Users Guide](first-time-users.md)

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

    [:octicons-arrow-right-24: API Reference](api.md)

-   :material-brain:{ .lg .middle } **Resources**

    ---

    There are many resources available from various sources on **Energy System Modelling** and **PyPSA**. Find **learning materials** here.

    [:octicons-arrow-right-24: Resources](resources.md)

-   :fontawesome-solid-users:{ .lg .middle } **Contributing**

    ---

    PyPSA is an **open source project** and we welcome any contributions to **keep the project alive**. Find out how to **contribute here**. You don't need to be a developer to contribute.

    [:octicons-arrow-right-24: Contributing](contributing.md)

-   :octicons-tools-16:{ .lg .middle } **More Tools**

    ---

    PyPSA is just one of many tools available for energy system modelling. Check out other **tools**, **projects** and **models** in the PyPSA universe.

    [:octicons-arrow-right-24: More Tools](more-tools.md)

</div>

