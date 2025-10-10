# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""
Shortcode replacements for MkDocs Material theme and Griffe extension.

This module provides both MkDocs hooks for processing shortcodes in markdown pages
and a Griffe extension for processing shortcodes in API docstrings.
"""

from __future__ import annotations

import logging
import posixpath
import re
from re import Match
from typing import TYPE_CHECKING

try:
    import griffe

    GRIFFE_AVAILABLE = True
except ImportError:
    GRIFFE_AVAILABLE = False

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import File, Files
    from mkdocs.structure.pages import Page


# -----------------------------------------------------------------------------
# Shared Helper Functions
# -----------------------------------------------------------------------------


def clean_file_path(path: str) -> str:
    """Clean file paths for display."""
    import os

    # Remove file extension
    name = os.path.splitext(path)[0]

    # Get just the filename if it's a path
    name = os.path.basename(name)

    # Handle special case where folder and file have same name (e.g., components/components)
    parts = path.split("/")
    if len(parts) > 1 and parts[-1].startswith(parts[-2]):
        name = parts[-2]

    # Replace hyphens and underscores with spaces
    name = name.replace("-", " ").replace("_", " ")

    # Capitalize each word
    name = " ".join(word.capitalize() for word in name.split())

    return name


def _badge(text: str = "", type: str = "") -> str:
    """Create a badge with optional text and type."""
    classes = f"mdx-badge mdx-badge--{type}" if type else "mdx-badge"
    return "".join(
        [
            f'<span class="{classes}">',
            *([f'<span class="mdx-badge__text">{text}</span>'] if text else []),
            "</span>",
        ]
    )


# -----------------------------------------------------------------------------
# MkDocs Hooks
# -----------------------------------------------------------------------------


def _process_shortcodes(text: str, page: Page, files: Files):
    """Process shortcodes in text for MkDocs pages."""

    def replace(match: Match):
        type, args = match.groups()
        args = args.strip()

        # Check if this is a badge variant
        is_badge = type.startswith("badge-")
        if is_badge:
            type = type[6:]  # Remove "badge-" prefix

        # Handle different shortcode types
        if type == "version":
            return (
                _badge_for_version(args, page, files)
                if is_badge
                else _link_for_version(args, page, files)
            )
        elif type == "pr":
            return _badge_pr(args) if is_badge else _link_pr(args)
        elif type == "guide":
            return (
                _badge_guide(args, page, files)
                if is_badge
                else _link_guide(args, page, files)
            )
        elif type == "example":
            # Example is an alias for guide
            return (
                _badge_guide(args, page, files)
                if is_badge
                else _link_guide(args, page, files)
            )
        elif type == "api":
            return (
                _badge_api(args, page, files)
                if is_badge
                else _link_api(args, page, files)
            )

        # Otherwise, raise an error
        raise RuntimeError(f"Unknown shortcode: {type}")

    return re.sub(r"<!-- md:([\w-]+)(.*?) -->", replace, text, flags=re.I | re.M)


def on_page_markdown(markdown: str, *, page: Page, config: MkDocsConfig, files: Files):
    """Process shortcodes in the main markdown content."""
    return _process_shortcodes(markdown, page, files)


def on_env(env, config: MkDocsConfig, files: Files):
    """Hook into the Jinja2 environment to add a filter for processing shortcodes."""

    def process_docstring_shortcodes(text):
        # Create a dummy page object if needed for docstring processing
        # This allows shortcodes to work even when page context isn't available
        try:
            # Try to get the current page from context if available
            page = None
            # Process the shortcodes
            return _process_shortcodes(text, page, files)
        except:
            # If processing fails, return original text
            return text

    # Add filter to Jinja2 environment for use in templates
    env.filters["shortcodes"] = process_docstring_shortcodes
    return env


# -----------------------------------------------------------------------------
# Helper functions for MkDocs
# -----------------------------------------------------------------------------


def _resolve_path(path: str, page: Page, files: Files):
    """Resolve path of file relative to given page."""
    original_path = path
    path, anchor, *_ = f"{path}#".split("#")
    file = files.get_file_from_path(path)
    if file is None:
        logging.warning(f"File not found: {path} (referenced in {page.file.src_uri})")
        return original_path  # Return original path unchanged
    path = _resolve(file, page)
    return "#".join([path, anchor]) if anchor else path


def _resolve(file: File, page: Page):
    """Resolve path of file relative to given page."""
    try:
        path = posixpath.relpath(file.src_uri, page.file.src_uri)
        return posixpath.sep.join(path.split(posixpath.sep)[1:])
    except Exception as e:
        raise Exception(f"Cannot resolve path for {file} relative to {page}") from e


# -----------------------------------------------------------------------------
# Badge creation functions for MkDocs
# -----------------------------------------------------------------------------


def _badge_for_version(text: str, page: Page, files: Files):
    """Create badge for version."""
    path = f"release-notes.md#{text}"
    icon = "material-tag-outline"
    href = _resolve_path(path, page, files)
    return _badge(f"[:{icon}: {text}]({href} 'Minimum Version')")


def _link_for_version(text: str, page: Page, files: Files):
    """Create plain link for version."""
    path = f"release-notes.md#{text}"
    icon = "material-tag-outline"
    href = _resolve_path(path, page, files)
    return f"[:{icon}: {text}]({href} 'Minimum Version')"


def _badge_pr(text: str):
    """Create badge for pull request."""
    icon = "octicons-git-pull-request-16"
    href = f"https://github.com/PyPSA/PyPSA/pull/{text}/"
    return _badge(f"[:{icon}: {text}]({href} 'View Pull Request')")


def _link_pr(text: str):
    """Create plain link for pull request."""
    icon = "octicons-git-pull-request-16"
    href = f"https://github.com/PyPSA/PyPSA/pull/{text}/"
    return f"[:{icon}: {text}]({href} 'View Pull Request')"


def _badge_guide(text: str, page: Page, files: Files):
    """Create badge for user guide."""
    # Always start from user-guide directory
    path = f"user-guide/{text}"
    icon = "material-bookshelf"
    href = _resolve_path(path, page, files)
    display_text = clean_file_path(text)
    return _badge(f"[:{icon}: {display_text}]({href} 'View User Guide')")


def _link_guide(text: str, page: Page, files: Files):
    """Create plain link for user guide."""
    # Always start from user-guide directory
    path = f"user-guide/{text}"
    icon = "material-bookshelf"
    href = _resolve_path(path, page, files)
    display_text = clean_file_path(text)
    return f"[:{icon}: {display_text}]({href} 'View User Guide')"


def _badge_example(text: str, page: Page, files: Files):
    """Create badge for example."""
    path = "examples/examples.md"
    icon = "material-bookshelf"
    href = _resolve_path(path, page, files)
    href = href.replace("examples.md", f"{text}")
    display_text = clean_file_path(text)
    return _badge(f"[:{icon}: {display_text}]({href} 'View Example')")


def _link_example(text: str, page: Page, files: Files):
    """Create plain link for example."""
    path = "examples/examples.md"
    icon = "material-bookshelf"
    href = _resolve_path(path, page, files)
    href = href.replace("examples.md", f"{text}")
    display_text = clean_file_path(text)
    return f"[:{icon}: {display_text}]({href} 'View Example')"


def _badge_api(text: str, page: Page, files: Files):
    """Create badge for API reference."""
    path = "api/networks/network.md"
    icon = "octicons-code-16"
    href = _resolve_path(path, page, files)
    href = href.replace("networks/network.md", f"{text}")
    display_text = clean_file_path(text)
    return _badge(f"[:{icon}: {display_text}]({href} 'View API Reference')")


def _link_api(text: str, page: Page, files: Files):
    """Create plain link for API reference."""
    path = "api/networks/network.md"
    icon = "octicons-code-16"
    href = _resolve_path(path, page, files)
    href = href.replace("networks/network.md", f"{text}")
    display_text = clean_file_path(text)
    return f"[:{icon}: {display_text}]({href} 'View API Reference')"


# -----------------------------------------------------------------------------
# Griffe Extension for API Docstrings
# -----------------------------------------------------------------------------


if GRIFFE_AVAILABLE:

    class ProcessShortcodes(griffe.Extension):
        """Process shortcodes in docstrings."""

        def on_module(self, *, mod: griffe.Module, **kwargs):
            """Process shortcodes in module docstrings."""
            if mod.docstring and mod.docstring.value:
                mod.docstring.value = self._process_shortcodes(
                    mod.docstring.value, mod.canonical_path
                )

        def on_class(self, *, cls: griffe.Class, **kwargs):
            """Process shortcodes in class docstrings."""
            if cls.docstring and cls.docstring.value:
                cls.docstring.value = self._process_shortcodes(
                    cls.docstring.value, cls.canonical_path
                )

        def on_function(self, *, func: griffe.Function, **kwargs):
            """Process shortcodes in function docstrings."""
            if func.docstring and func.docstring.value:
                func.docstring.value = self._process_shortcodes(
                    func.docstring.value, func.canonical_path
                )

        def on_attribute(self, *, attr: griffe.Attribute, **kwargs):
            """Process shortcodes in attribute docstrings."""
            if attr.docstring and attr.docstring.value:
                attr.docstring.value = self._process_shortcodes(
                    attr.docstring.value, attr.canonical_path
                )

        def _calculate_depth(self, canonical_path: str) -> int:
            """Calculate the depth level based on the canonical path.

            For example:
            - pypsa.Network -> depth 2 (api/networks/network.md)
            - pypsa.components.Components -> depth 2 (api/components/components.md)
            - pypsa.components.Generators -> depth 3 (api/components/types/generators.md)
            """
            # Map canonical paths to their documentation structure
            # Component types (with specific type names) are at depth 3
            if canonical_path.startswith("pypsa.components._types."):
                # This is a component type like pypsa.components._types.generators.Generators
                return 3
            # Component types from public API are also at depth 3
            elif (
                canonical_path.startswith("pypsa.components.")
                and canonical_path.count(".") == 2
            ):
                # Check if it's a component type (e.g., pypsa.components.Generators)
                # but not the base Components class
                component_name = canonical_path.split(".")[-1]
                if component_name not in (
                    "Components",
                    "components",
                    "common",
                    "array",
                ):
                    return 3
            # Other API docs are typically at depth 2: api/<category>/<item>.md
            return 2

        def _process_shortcodes(self, text, canonical_path: str = ""):
            """Process shortcode patterns in text for API docstrings."""
            # Calculate depth for relative path construction
            depth = self._calculate_depth(canonical_path)
            path_prefix = "../" * depth

            def replace(match):
                type, args = match.groups()
                args = args.strip() if args else ""

                # Check if this is a badge variant
                is_badge = type.startswith("badge-")
                if is_badge:
                    type = type[6:]  # Remove "badge-" prefix

                # For API docstrings, use relative path resolution based on depth
                try:
                    if type == "version":
                        # Use relative path for version links
                        href = f"{path_prefix}release-notes.md#{args}"
                        icon = "material-tag-outline"
                        if is_badge:
                            return _badge(
                                f"[:{icon}: {args}]({href} 'Minimum Version')"
                            )
                        else:
                            return f"[:{icon}: {args}]({href} 'Minimum Version')"
                    elif type == "pr":
                        if is_badge:
                            return _badge_pr(args)
                        else:
                            return _link_pr(args)
                    elif type == "guide":
                        # Use relative path for guide links
                        href = f"{path_prefix}user-guide/{args}"
                        icon = "material-bookshelf"
                        display_text = clean_file_path(args)
                        if is_badge:
                            return _badge(
                                f"[:{icon}: {display_text}]({href} 'View User Guide')"
                            )
                        else:
                            return (
                                f"[:{icon}: {display_text}]({href} 'View User Guide')"
                            )
                    elif type == "example":
                        # Use relative path for example links
                        href = f"{path_prefix}examples/{args}"
                        icon = "material-notebook-multiple"
                        display_text = clean_file_path(args)
                        if is_badge:
                            return _badge(
                                f"[:{icon}: {display_text}]({href} 'View Example')"
                            )
                        else:
                            return f"[:{icon}: {display_text}]({href} 'View Example')"
                    elif type == "api":
                        # Use relative path for API links
                        href = f"{path_prefix}api/{args}"
                        icon = "octicons-code-16"
                        display_text = clean_file_path(args)
                        if is_badge:
                            return _badge(
                                f"[:{icon}: {display_text}]({href} 'View API Reference')"
                            )
                        else:
                            return f"[:{icon}: {display_text}]({href} 'View API Reference')"
                except Exception as e:
                    # Fallback to original if anything fails
                    print(
                        f"Warning: Shortcode processing failed for {type} {args}: {e}"
                    )
                    return match.group(0)

                # Return original if unknown type
                return match.group(0)

            # Process the shortcodes - allow hyphens in shortcode type
            return re.sub(
                r"<!-- md:([\w-]+)(.*?) -->", replace, text, flags=re.I | re.M
            )
