# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Render the spec as pages of the docs, one per source file.

`system.yaml` gives the index page. Each file under `traits/` gives a
page, rendered for the first component that declares it. Each component
gives a page with all that holds for it: its traits, renamed, and its own
file under `components/`. An entry shows its math and its source, renamed,
under its key as anchor. A page closes with the entries it reads from
other pages and a legend of the symbols it uses or declares.

A hand-written page embeds entries by marker, see `marker`.
"""

from __future__ import annotations

import dataclasses
import functools
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import mathspec
import yaml
from mathspec._typesetting import _walk
from mathspec.program import ExpressionReference, Relation

import pypsa
from pypsa.optimization.spec._render import (
    formulated,
    load,
    renamed,
    renamer,
    snake,
    symbols,
)
from pypsa.optimization.spec.layers import SPEC, patch_folder, spec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from importlib.resources.abc import Traversable

    from mathspec.spec import Spec

    from pypsa.components.components import Components

#: The sections whose entries print as a line, in page order, with their kind.
LINES = {
    "variables": "Variable",
    "expressions": "Definition",
    "conditions": "Condition",
    "constraints": "Constraint",
    "piecewise": "Curve",
    "assumptions": "Assumption",
}
#: The sections a page takes entries from: those that print as a line, then those only its legend explains.
SECTIONS = (*LINES, "parameters", "relations")
#: The sections whose entries the legend explains, with their kind.
LEGEND = {
    "parameters": "parameter",
    "relations": "relation",
    "variables": "variable",
    "expressions": "definition",
    "conditions": "condition",
}
#: What each placeholder of a trait stands for, as the trait pages name it.
ROLES = {
    "COMPONENT": "Component",
    "UNIT": "Unit dimension",
    "NOM": "Nominal capacity",
    "POWER": "Power variable",
    "LEVEL": "Storage level",
    "CYCLIC": "Cyclic flag",
    "COEF": "Port coefficient",
}
INDEX = "index.md"
#: The folder of the generated pages in the docs.
ROOT = "user-guide/optimization/formulation"
#: What parts a line from its quantifier.
QUANTIFIER = r" \qquad "
#: The word that, as first argument of a marker, takes the spec of linearised unit commitment.
LINEARIZED = "linearized"

Groups = list[tuple[str, list["Entry"]]]


@dataclass(frozen=True)
class Entry:
    """One entry of a source file.

    Attributes:
        key: Its key in the rendered spec.
        section: The section it stands in, one of `SECTIONS`.
        source: Its entry as the file states it, renamed, with `name:` and `scaling:`.
        page: The page it is anchored on, empty for an entry of a patch, which no page shows.

    """

    key: str
    section: str
    source: dict[str, Any] | str
    page: str


class _Context(NamedTuple):
    spec: Spec
    table: mathspec.SymbolTable
    names: Mapping[str, str]
    indices: Mapping[str, str]
    sets: Mapping[str, str]
    upright: Callable[[str], str]
    owners: Mapping[str, str]


def _entries(path: Traversable, rename: Callable[[str], str], page: str) -> list[Entry]:
    """Return the entries of the source file at *path* in `SECTIONS`, renamed by *rename*."""
    if not path.is_file():
        return []
    data, names, scalings = load(path, rename)
    entries = []
    for section in SECTIONS:
        for key, body in data.get(section, {}).items():
            name = rename(key)
            source = renamed(body, rename)
            if isinstance(source, dict):
                named = names.get(section, {})
                source = ({"name": named[name]} if name in named else {}) | source
                if (section, name) in scalings:
                    source["scaling"] = scalings[section, name]
            entries.append(Entry(name, section, source, page))
    return entries


def _page_of(c: Components) -> str:
    return f"components/{snake(c.name)}.md"


def _read(node: Any) -> Iterator[str]:
    """Yield each name *node* holds, but in the body of a named expression or condition it reads."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, Mapping):
        for value in node.values():
            yield from _read(value)
    elif isinstance(node, tuple | list | frozenset):
        for value in node:
            yield from _read(value)
    elif is_dataclass(node):
        for f in fields(node):
            if f.name not in ("body", "description"):
                yield from _read(getattr(node, f.name))


def _link(page: str, target: str, anchor: str = "") -> str:
    """Return the link from *page* to *target*, at *anchor* where given.

    A link always holds a slash, since the plugin `autolinks` resolves one
    without by the file name alone.
    """
    relative = posixpath.relpath(target, posixpath.dirname(page) or ".")
    path = "" if target == page else relative if "/" in relative else f"./{relative}"
    return f"{path}#{anchor}" if anchor else path


def _cell(text: str | None) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def _row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: Iterable[Iterable[str]]) -> str:
    lines = [_row(header), _row("---" for _ in header), *map(_row, rows)]
    return "\n".join(lines)


def _math(latex: str, indent: str = "") -> str:
    return "\n".join(f"{indent}{line}" for line in ["$$", *latex.splitlines(), "$$"])


def _line(entry: Entry, ctx: _Context) -> str:
    return mathspec.typeset_line(
        ctx.spec, entry.key, "latex", symbols=ctx.table, inline_expressions=False
    )


def _body(entry: Entry, ctx: _Context, lead: str) -> str:
    """Return *lead* with the description of *entry*, then its math and source as tabs."""
    source = yaml.safe_dump(
        {entry.key: entry.source},
        sort_keys=False,
        allow_unicode=True,
        width=88,
        default_flow_style=False if isinstance(entry.source, str) else None,
    )
    code = "\n".join(f"    {row}" for row in ["```yaml", *source.splitlines(), "```"])
    described = isinstance(entry.source, dict) and entry.source.get("description")
    text = " ".join([lead, *(described or "").split()])
    math = _math(_line(entry, ctx), "    ")
    return f'{text}\n\n=== "Math"\n\n{math}\n\n=== "YAML"\n\n{code}\n'


def _block(entry: Entry, ctx: _Context) -> str:
    """Return the heading, description, math and source of *entry*."""
    lead = f"*{LINES[entry.section]}.*"
    return f"### `{entry.key}` {{#{entry.key}}}\n\n{_body(entry, ctx, lead)}"


def _over(entry: Any, sets: Mapping[str, str]) -> str:
    """Return the sets *entry* is indexed over, or for a relation the map or subset it is."""

    def product(dims: Iterable[str]) -> str:
        return r" \times ".join(sets[d] for d in dims)

    if not isinstance(entry, Relation):
        return product(entry.dims)
    if not entry.values:
        return rf"\subseteq {product(entry.dims)}"
    keys, values = (
        [entry.dim(r) for r in roles] for roles in (entry.key, entry.values)
    )
    return rf"{product(keys)} \to {product(values)}"


def _reads(entries: list[Entry], ctx: _Context) -> set[str]:
    """Return the keys of *entries* and the names they read."""
    program = ctx.spec.program
    return {e.key for e in entries} | {
        name for e in entries for name in _read(getattr(program, e.section)[e.key])
    }


def _symbols(read: set[str], ctx: _Context) -> str:
    """Return the tables of the sets and the named symbols among *read*."""
    program = ctx.spec.program
    sets = _table(
        ["Set", "Index", "Dimension", "Description"],
        (
            [
                f"${ctx.sets[d]}$",
                f"${ctx.indices[d]}$",
                f"`{d}`",
                _cell(entry.description),
            ]
            for d, entry in program.dimensions.items()
            if d in read
        ),
    )
    named = [
        (section, kind, name, getattr(program, section)[name])
        for section, kind in LEGEND.items()
        for name in sorted(read & set(getattr(program, section)))
    ]
    names = _table(
        ["Symbol", "Name", "Kind", "Over", "Description"],
        (
            [
                f"${ctx.upright(name) if section == 'relations' else ctx.names[name]}$",
                f"`{name}`",
                kind,
                f"${over}$" if (over := _over(entry, ctx.sets)) else "",
                _cell(entry.description),
            ]
            for section, kind, name, entry in named
        ),
    )
    return f"{sets}\n\n{names}"


def _legend(page: str, entries: list[Entry], ctx: _Context) -> str:
    """Return the links to what *entries* read from other pages, and the legend of the symbols they use or declare."""
    read = _reads(entries, ctx)
    here = {e.key for e in entries}
    uses = sorted(n for n in read - here if ctx.owners.get(n, page) != page)
    links = "\n".join(f"- [`{n}`]({_link(page, ctx.owners[n], n)})" for n in uses)
    parts = (
        [f"## Uses\n\nWhat this page reads from other pages.\n\n{links}"]
        if uses
        else []
    )
    return "\n\n".join([*parts, f"## Symbols\n\n{_symbols(read, ctx)}"])


def _render(page: str, title: str, intro: str, groups: Groups, ctx: _Context) -> str:
    """Return the page at *page*: title, intro, each group's lines under its heading, then the legend."""
    body = [
        f"## {heading}\n\n" + "\n".join(_block(e, ctx) for e in lines)
        for heading, group in groups
        if (lines := [e for e in group if e.section in LINES])
    ]
    entries = [e for _, group in groups for e in group]
    return "\n\n".join([f"# {title}", intro, *body, _legend(page, entries, ctx)]) + "\n"


def _by_section(entries: list[Entry]) -> Groups:
    return [
        (
            f"{LINES.get(section, section)}s",
            [e for e in entries if e.section == section],
        )
        for section in SECTIONS
    ]


def _index(system: list[Entry], components: list[Components], ctx: _Context) -> str:
    """Return the index page: the system file, the objective and the links to the other pages.

    Raises
    ------
    ValueError
        The spec does not optimize one named expression.

    """
    objective = ctx.spec.program.objective
    if objective is None or not isinstance(objective.expression, ExpressionReference):
        msg = "The spec does not optimize one named expression."
        raise ValueError(msg)
    sense = r"\min" if objective.sense == "minimize" else r"\max"
    traits = sorted({t for c in components for t in c._formulation})
    intro = "\n\n".join(
        [
            "These pages are generated from the YAML spec in `pypsa/optimization/spec/`, "
            "from which `n.optimize()` builds its model. Each entry shows its math and "
            "its source.",
            " ".join((ctx.spec.description or "").split()),
            "## Objective\n\n"
            + _math(rf"{sense} \; {ctx.names[objective.expression.name]}"),
            "## Traits\n\n" + "\n".join(f"- [{t}](traits/{t}.md)" for t in traits),
            "## Components\n\n"
            + "\n".join(f"- [{c.name}]({_page_of(c)})" for c in components),
        ]
    )
    return _render(INDEX, "Formulation", intro, _by_section(system), ctx)


def _component(c: Components, groups: Groups, ctx: _Context) -> str:
    """Return the page of component *c*, with the entries of each of its source files in *groups*."""
    page = _page_of(c)
    intro = (
        f"Everything `n.optimize()` states for the `{c.name}` component: the traits "
        f"it declares, renamed for it, and its own file `components/{snake(c.name)}.yaml`."
    )
    linked = [
        (f"Trait [{t}]({_link(page, f'traits/{t}.md')})", group)
        for t, group in groups[:-1]
    ]
    own = (f"Component file `components/{snake(c.name)}.yaml`", groups[-1][1])
    return _render(page, c.name, intro, [*linked, own], ctx)


def _trait(trait: str, entries: list[Entry], ctx: _Context) -> str:
    """Return the page of *trait*, its *entries* rendered for the first component that declares it."""
    page = f"traits/{trait}.md"
    declaring = formulated(pypsa.Network(), trait)
    found = [symbols(c) for c in declaring]
    roles = [r for r in ROLES if any(r in f for f in found)]
    names = _table(
        [ROLES[r] for r in roles],
        ([f"`{f[r]}`" if r in f else "" for r in roles] for f in found),
    )
    intro = (
        f"The trait `{trait}` states one formulation, in `traits/{trait}.yaml`, for "
        f"each component that declares it. This page shows it for "
        f"[`{declaring[0].name}`]({_link(page, _page_of(declaring[0]))}). Each "
        f"component gives the trait these names:\n\n{names}"
    )
    return _render(page, trait, intro, _by_section(entries), ctx)


@functools.cache
def _files(root: Traversable) -> tuple[list[Entry], dict[str, Groups]]:
    """Return the entries of the system file under *root*, and of each component its entries grouped by source file."""
    system = _entries(root / "system.yaml", lambda name: name, INDEX)
    groups = {
        c.name: [
            *(
                (t, _entries(root / f"traits/{t}.yaml", renamer(c), _page_of(c)))
                for t in c._formulation
            ),
            (
                "",
                _entries(
                    root / f"components/{snake(c.name)}.yaml", renamer(c), _page_of(c)
                ),
            ),
        ]
        for c in formulated(pypsa.Network())
    }
    return system, groups


def _every(root: Traversable) -> list[Entry]:
    system, groups = _files(root)  # type: ignore[arg-type]
    return [*system, *(e for g in groups.values() for _, group in g for e in group)]


@functools.cache
def _context(linearized_unit_commitment: bool = False) -> _Context:
    """Return how the spec, patched for linearised unit commitment on request, prints.

    Raises
    ------
    mathspec.errors.SchemaError
        `symbols.yaml` names a dimension or entry the spec lacks.

    """
    s = spec(linearized_unit_commitment)
    table = mathspec.SymbolTable.load(
        yaml.safe_load((SPEC / "symbols.yaml").read_text())
    )
    symbols_ = _walk(s, "latex", table, inline_expressions=False)
    return _Context(
        s,
        table,
        symbols_.symbols.name,
        symbols_.symbols.index,
        symbols_.symbols.set,
        symbols_.format.upright,
        {e.key: e.page for e in _every(SPEC) if e.section in LINES},
    )


def _first(entries: list[Entry]) -> dict[str, Entry]:
    """Return *entries* by key, the first of each key, so a line wins over a parameter of its name."""
    keyed: dict[str, Entry] = {}
    for e in entries:
        keyed.setdefault(e.key, e)
    return keyed


@functools.cache
def _keyed(linearized_unit_commitment: bool) -> dict[str, Entry]:
    """Return each entry by its key, those of the patch of linearised unit commitment on request."""
    keyed = _first(_every(SPEC))
    patch = patch_folder(linearized_unit_commitment)
    if patch is not None:
        patched = _first(_every(patch))
        keyed |= {k: dataclasses.replace(e, page="") for k, e in patched.items()}
    return keyed


def _lookup(keys: list[str], linearized_unit_commitment: bool) -> list[Entry]:
    """Return the entries of *keys*.

    Raises
    ------
    ValueError
        *keys* is empty or names an entry the spec lacks.

    """
    keyed = _keyed(linearized_unit_commitment)
    missing = [k for k in keys if k not in keyed]
    if not keys or missing:
        msg = f"The spec has no entries {missing or keys}."
        raise ValueError(msg)
    return [keyed[k] for k in keys]


def _anchored(entry: Entry, root: str, text: str) -> str:
    """Return *text*, linked to *entry* on its generated page under *root* where one shows it."""
    return f"[{text}]({root}/{entry.page}#{entry.key})" if entry.page else text


def _named(entry: Entry) -> str:
    named = isinstance(entry.source, dict) and entry.source.get("name") or entry.key
    return " / ".join(named.values()) if isinstance(named, dict) else named


def _snippet(entries: list[Entry], root: str, ctx: _Context) -> str:
    """Return each of *entries* with its kind, key, description, math and source.

    Raises
    ------
    ValueError
        An entry prints no line.

    """
    lineless = [e.key for e in entries if e.section not in LINES]
    if lineless:
        msg = f"The entries {lineless} print no line."
        raise ValueError(msg)
    return "\n".join(
        _body(e, ctx, f"*{LINES[e.section]}* {_anchored(e, root, f'`{e.key}`')}:")
        for e in entries
    )


def _constraints(entries: list[Entry], root: str, ctx: _Context) -> str:
    """Return a table of *entries*: the relation each states, without its quantifier, and its PyPSA name."""
    rows = (
        [
            f"${_cell(_line(e, ctx).split(QUANTIFIER, 1)[0])}$",
            _anchored(e, root, f"`{_named(e)}`"),
        ]
        for e in entries
    )
    return _table(["Constraint", "Name"], rows)


#: What each marker prints, by its kind.
MARKERS: dict[str, Callable[[list[Entry], str, _Context], str]] = {
    "spec": _snippet,
    "spec-legend": lambda entries, _, ctx: _symbols(_reads(entries, ctx), ctx),
    "spec-table": _constraints,
}


def marker(kind: str, args: str, page: str) -> str:
    """Return the markdown that a marker `<!-- md:KIND ARGS -->` on the docs page *page* stands for.

    *args* names entries by key, after the word `LINEARIZED` to take them
    from the spec of linearised unit commitment. The kind `spec` prints
    each entry as the generated pages do, `spec-legend` the legend of the
    symbols they use or declare, `spec-table` a table of their relations
    and PyPSA names.

    Raises
    ------
    ValueError
        *kind* is no marker, or *args* names no entry or one the spec lacks.

    """
    if kind not in MARKERS:
        msg = f"No marker {kind!r}, only {sorted(MARKERS)}."
        raise ValueError(msg)
    words = args.split()
    linearized = words[:1] == [LINEARIZED]
    entries = _lookup(words[linearized:], linearized)
    root = posixpath.relpath(ROOT, posixpath.dirname(page))
    return MARKERS[kind](entries, root, _context(linearized))


@functools.cache
def pages() -> dict[str, str]:
    """Return the formulation pages of the docs, keyed by their path relative to the formulation folder.

    Raises
    ------
    mathspec.errors.SchemaError
        `symbols.yaml` names a dimension or entry the spec lacks.

    """
    ctx = _context()
    system, groups = _files(SPEC)  # type: ignore[arg-type]
    components = formulated(pypsa.Network())
    examples: dict[str, list[Entry]] = {}
    for c in components:
        for t, group in groups[c.name][:-1]:
            examples.setdefault(t, group)
    return {
        INDEX: _index(system, components, ctx),
        **{_page_of(c): _component(c, groups[c.name], ctx) for c in components},
        **{f"traits/{t}.md": _trait(t, examples[t], ctx) for t in sorted(examples)},
    }
