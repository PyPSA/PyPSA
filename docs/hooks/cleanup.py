# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re

try:
    import griffe

    GRIFFE_AVAILABLE = True
except ImportError:
    GRIFFE_AVAILABLE = False


def on_page_markdown(markdown, page, config, files):
    # Remove # doctest: +SKIP from code blocks
    pattern = r"(``` py.*?```)"

    def remove_doctest_skip(match):
        code_block = match.group(0)
        # Remove the doctest comments
        code_block = re.sub(r"\s*# doctest: \+SKIP", "", code_block)
        code_block = re.sub(r"\s*# doctest: \+ELLIPSIS", "", code_block)
        # Remove entire lines ending with # docs-hide
        code_block = re.sub(r"^.*# docs-hide\s*$", "", code_block, flags=re.MULTILINE)
        code_block = re.sub(r"^.*<BLANKLINE>\s*$", "", code_block, flags=re.MULTILINE)
        return code_block

    markdown = re.sub(pattern, remove_doctest_skip, markdown, flags=re.DOTALL)

    # Filter out Gurobi license messages from notebook output
    gurobi_patterns = [
        r"^.*Set parameter WLSAccessID.*$\n?",
        r"^.*Set parameter WLSSecret.*$\n?",
        r"^.*Set parameter LicenseID.*$\n?",
        r"^.*Academic license.*for non-commercial use only.*$\n?",
        r"^.*environment still referenced so free is deferred.*$\n?",
    ]
    for pattern in gurobi_patterns:
        markdown = re.sub(pattern, "", markdown, flags=re.MULTILINE)

    return markdown
