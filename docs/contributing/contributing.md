<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Instructions for Contributing

First of all, thank you for thinking about contributing to PyPSA! 

We strongly welcome anyone interested in PyPSA and all its related projects, be it
with new ideas, suggestions, submitting bug reports or contributing code changes.

**How to contribute**

* [Code Contributions](#code): Implement new features, fix bugs, or improve the performance.
* [Documentation Contributions](#documentation): Improve the documentation by adding new sections, fixing typos, or clarifying existing content.
* [Example Contributions](#examples): Showcase your work, which could be useful for others.

**Where to go for help**

* To **discuss** with other PyPSA users, organise projects, share news, and get in touch with the community you can use the [Discord server](https://discord.gg/AnuJBk23FU).
* For **troubleshooting**, please check the [troubleshooting](troubleshooting.md) in the documentation.
* For **guidelines to contribute** to PyPSA, stay right here.

## Code

**Contribution workflow in a nutshell**

1. Fork the repository on GitHub
2. Clone your fork: `git clone https://github.com/<your-username>/PyPSA.git`
3. Fetch the upstream tags `git fetch --tags https://github.com/PyPSA/PyPSA.git`
4. Install with dependencies in editable mode: `pip install -e .[dev]`
5. Setup linter and formatter, e.g `pre-commit install` (see [Code Style](#style))
6. Write your code (preferably on a new branch)
7. Run tests: `pytest` (see [Testing](#testing))
8. Push your changes to your fork and create a pull request on GitHub

<!-- TODO: What to work on, TODO, which issues, labeling etc. -->

### Style

**pre-commit**

We run a couple of tools via [pre-commit](https://pre-commit.com) to ensure a 
consistent code style and to catch common programming errors or bad practices before
they are committed. Don't worry, you can just start coding and the pre-commit will 
tell you if something is not right.

It is already installed with the development dependencies, but you can also install it
manually via `pip install pre-commit` or `conda install -c conda-forge pre-commit`.

To use it automatically before every commit (recommended), just run once:

``` bash
pre-commit install
```

This will automatically check the changes which are staged before you commit them.

To manually run it, use:

``` bash
pre-commit run --all
```

This will check all files in the repository.

**Ruff**

One of the tools that is run by pre-commit is [Ruff](https://docs.astral.sh/ruff),
which is our linter and formatter. It combines common tools like Flake8, Black, etc. 
Besides pre-commit, you can also run it via your CLI (see [Ruff installation](https://docs.astral.sh/ruff/installation/)) 
or IDE (e.g. VSCode [plugin](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)).
This will help you to keep your code clean and consistent already during development.

Ruff is also already installed with the development dependencies, but you can also install it
manually using `pip install ruff`.

To use the linter in your CLI, run:

``` bash
ruff check . --fix
```

This will check all files in the repository and gives you hints on what to improve. The 
`--fix` flag will also automatically fix some of the issues, if possible. Some 
issues need to be fixed manually.

And to run the formatter, use:

``` bash
ruff format .
```

This will format all the files in the repository and immediately apply the changes to 
them. It is basically [the same](https://docs.astral.sh/ruff/faq/#how-does-ruffs-formatter-compare-to-black)
as Black. 

!!! note

    It is not mandatory to use either Ruff or pre-commit. We will also be running it in our CI/CD pipeline. But it's highly recommended, to make everyone's life easier.

### Testing

Unit testing is performed with pytest which is installed with the development dependencies.

The tests can be found in the `test/` folder and can be run with:

``` bash
pytest
```

Or to run individual tests:

``` bash
pytest test_lpf_against_pypower.py
```

Power flow is tested against PYPOWER (the Python implementation of MATPOWER)
and pandapower.

Unit testing is also performed in the CI/CD pipeline, similar to the linting and formatting.

## Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org) and the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material) theme.

We strive to keep documentation useful and up to date for all PyPSA users. If
you encounter an area where documentation is not available or insufficient, we
very much welcome your contribution.

!!! note

    If you are not familiar with Markdown, consult the following [quick guide](https://www.markdownguide.org/basic-syntax/).

For bigger changes, we recommend to draft them locally. Just follow the steps in 
[Code Contributions](#code) to set up your local environment. In addition you can:

1. Also install the documentation dependencies via `pip install -e .[docs]`.
2. Make your changes in the corresponding `.md` file in the `docs` directory.
3. Compile your changes by running the following command in your terminal in the `doc` folder: `mkdocs serve`
   
!!! info

    If you are not editing example notebooks (`.ipynb` files), you may wanna comment out 
    the `mkdocs-jupyter` plugin in the `mkdocs.yml` file, to make the build faster. 
    In any case, you will end up with a link to a local server (e.g. `http://127.0.0.1:8000`).

For simple changes, you can also edit the documentation directly on GitHub:

1. If you are on the documentation page, click on the little paper icon with the pen on the top right. This will bring you straight to the source file.
2. Make your changes in the file.
3. Commit your changes and create a pull request.

## Examples

Nice examples are always welcome.

You can even submit your Jupyter notebook (`.ipynb`) directly
as an example. Please run the linter (see [Code Style](#style)) to ensure
that the notebook is clean and metadata is removed.

Then for every notebook:

1. Write the notebook (let's call it `foo.ipynb`) and place it
   in `docs/examples/foo.ipynb`.

2. Reference it in the configuration file `docs/mkdocs.yml` where the other examples are listed.

3. Commit your changes and create a pull request.
