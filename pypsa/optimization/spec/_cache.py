# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Keep the rendered and lowered spec on disk, so the next process loads it.

Rendering and lowering parse every expression of the spec, which takes
longer than building a small model. A file is named after everything they
read: the YAML files, the formulation and placeholder values of each
component, the code that renders, and the versions of what the pickle holds.
A change to any of them names another file, so a stale one is never read.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import pickle
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import user_cache_dir

from pypsa._options import options
from pypsa.optimization.spec import _render

if TYPE_CHECKING:
    from collections.abc import Iterator
    from importlib.resources.abc import Traversable

    from linopy.spec import Lowered
    from mathspec.spec import Spec

    #: The spec, the PyPSA names of its entries, and the spec lowered for linopy.
    Rendered = tuple[Spec, _render.Names, Lowered]

logger = logging.getLogger(__name__)


def _cache_root() -> Path:
    """Return the folder of the rendered specs."""
    return Path(user_cache_dir("pypsa")) / "spec"


def _files(folder: Traversable, prefix: str = "") -> Iterator[tuple[str, bytes]]:
    """Yield the path and content of each YAML file under *folder*, in path order."""
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        path = f"{prefix}{entry.name}"
        if entry.is_dir():
            yield from _files(entry, f"{path}/")
        elif entry.name.endswith(".yaml"):
            yield path, entry.read_bytes()


def _key(root: Traversable, patch: Traversable | None) -> str:
    """Return the name of the rendering of *root* under *patch*, a digest of all it reads."""
    import pypsa  # noqa: PLC0415

    components = _render.formulated(pypsa.Network())
    parts = [
        sys.version,
        *(version(package) for package in ("mathspec", "pydantic", "linopy")),
        inspect.getsource(_render),
        repr(str(patch) if patch is not None else None),
        *(repr((c.name, c._formulation, _render.symbols(c))) for c in components),
    ]
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
    for path, content in _files(root):
        digest.update(path.encode())
        digest.update(content)
    return digest.hexdigest()


def _read(path: Path) -> Rendered | None:
    """Return the rendering kept at *path*, or None where none is kept or it is cut short."""
    try:
        with path.open("rb") as file:
            return pickle.load(file)  # noqa: S301
    except FileNotFoundError:
        return None
    except (pickle.UnpicklingError, EOFError):
        logger.warning("The rendered spec at %s is unreadable; rendering anew.", path)
        return None


def _write(path: Path, rendered: Rendered) -> None:
    """Keep *rendered* at *path*, written whole or not at all."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(handle, "wb") as file:
            pickle.dump(rendered, file, protocol=pickle.HIGHEST_PROTOCOL)
        Path(name).replace(path)
    except OSError as error:
        logger.debug("The rendered spec is not kept at %s: %s", path, error)


def _rendered(root: Traversable, patch: Traversable | None) -> Rendered:
    """Return the spec of *root* under *patch*, its PyPSA names and the spec lowered."""
    from linopy.spec import lowered  # noqa: PLC0415

    spec, names = _render.render(root, patch)
    return spec, names, lowered(spec)


def render(root: Traversable, patch: Traversable | None = None) -> Rendered:
    """Return the rendered and lowered spec, from disk where an earlier process kept it."""
    if not options.optimization.spec_cache:
        return _rendered(root, patch)
    path = _cache_root() / f"{_key(root, patch)}.pkl"
    rendered = _read(path)
    if rendered is None:
        rendered = _rendered(root, patch)
        _write(path, rendered)
    return rendered
