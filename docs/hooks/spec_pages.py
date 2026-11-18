# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Add the formulation pages, generated from the YAML spec, to the docs.

The nav entry `Formulation (generated)` names the index page. This hook
expands it with the trait and component pages. The pages are rendered
once per process, so `mkdocs serve` shows a changed spec after a restart.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mkdocs.structure.files import File

from pypsa.optimization.spec._docs import INDEX, ROOT, pages

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files

TITLE = "Formulation (generated)"


def _section(folder: str) -> list[str]:
    return [f"{ROOT}/{path}" for path in pages() if path.startswith(f"{folder}/")]


def _expand(nav: Any) -> bool:
    """Replace the nav entry `TITLE` in *nav* by the generated pages, and return whether *nav* held it."""
    if isinstance(nav, list):
        return any(_expand(item) for item in nav)
    if not isinstance(nav, dict):
        return False
    if TITLE in nav:
        nav[TITLE] = [
            f"{ROOT}/{INDEX}",
            {"Traits": sorted(_section("traits"))},
            {"Components": _section("components")},
        ]
        return True
    return any(_expand(value) for value in nav.values())


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    """Expand the nav entry of the generated pages.

    Raises:
        ValueError: The nav lacks the entry `Formulation (generated)`.

    """
    if not _expand(config.nav):
        msg = f"The nav lacks the entry {TITLE!r}."
        raise ValueError(msg)
    return config


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Add the generated pages to *files*.

    Each page is written to a temporary file, since the plugin `autolinks`
    reads the source path of every page. Its name is unique, since `autolinks`
    resolves a link by the file name alone.
    """
    folder = Path(tempfile.mkdtemp(prefix="spec_pages_"))
    for path, text in pages().items():
        source = folder / f"formulation-{path.replace('/', '-')}"
        source.write_text(text)
        files.append(File.generated(config, f"{ROOT}/{path}", abs_src_path=str(source)))
    return files
