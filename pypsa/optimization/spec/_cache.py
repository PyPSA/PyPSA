# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Keep each checked spec, so that a process reads it instead of parsing it again.

Parsing the spec is the bulk of a process's first build. A checked spec,
with what is derived from it, is kept in memory and, unless
`options.optimization.formulation_cache` is off, as a pickle in PyPSA's
user cache directory. Its key is the content it is checked from, with the
versions of the packages that check it and of the code that renders it,
so an edit to any of them misses the cache rather than reading a stale
spec.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import os
import pickle
import platform
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platformdirs import user_cache_dir

from pypsa._options import options

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

#: The most files kept on disk; the least recently used go first.
KEEP = 32

_MEMORY: dict[str, Any] = {}


def directory() -> Path:
    """Return the directory of the cached specs."""
    return Path(user_cache_dir("pypsa")) / "formulation"


@functools.cache
def _stamp() -> list[str]:
    code = [
        Path(__file__).with_name(f).read_bytes() for f in ("_render.py", "build.py")
    ]
    return [
        platform.python_version(),
        version("mathspec"),
        version("pydantic"),
        *(hashlib.sha256(c).hexdigest() for c in code),
    ]


def digest(key: Any) -> str:
    """Return the cache key of the spec checked from *key*, which serialises as JSON."""
    text = json.dumps([_stamp(), key], default=repr)
    return hashlib.sha256(text.encode()).hexdigest()


def _read(path: Path) -> Any:
    try:
        value = pickle.loads(path.read_bytes())  # noqa: S301  written by `_write`
    except FileNotFoundError:
        return None
    except (OSError, EOFError, AttributeError, ImportError, pickle.UnpicklingError):
        logger.warning("Ignoring the unreadable cached formulation %s.", path)
        return None
    with contextlib.suppress(OSError):
        path.touch()
    return value


def _write(path: Path, value: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.stem}.{os.getpid()}.part")
        part.write_bytes(pickle.dumps(value))
        part.replace(path)
        kept = sorted(path.parent.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
        for old in kept[:-KEEP]:
            old.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("Could not cache the formulation in %s: %s", path.parent, e)


def cached[T](key: str, check: Callable[[], T]) -> T:
    """Return the value under *key*, from memory, from disk or made by *check*."""
    if key in _MEMORY:
        return _MEMORY[key]
    on_disk = options.optimization.formulation_cache
    path = directory() / f"{key}.pkl"
    value = _read(path) if on_disk else None
    if value is None:
        value = check()
        if on_disk:
            _write(path, value)
    _MEMORY[key] = value
    return value
