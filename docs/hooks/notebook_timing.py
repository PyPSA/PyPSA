# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""
MkDocs hook to time notebook execution during documentation builds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_timing_data: dict[str, dict] = {}


def on_pre_page(page, config, files):
    """Record start time before notebook processing."""
    src_path = page.file.src_path
    if src_path.endswith(".ipynb"):
        _timing_data[src_path] = {"start": time.perf_counter()}
    return page


def on_page_content(html, page, config, files):
    """Record end time after notebook is converted (including execution)."""
    src_path = page.file.src_path
    if src_path in _timing_data:
        elapsed = time.perf_counter() - _timing_data[src_path]["start"]
        _timing_data[src_path]["elapsed"] = elapsed
        # Print directly to stderr for visibility in build logs
        print(f"[notebook-timing] {elapsed:6.1f}s  {src_path}", file=sys.stderr)
    return html


def on_post_build(config):
    """Log summary of notebook execution times."""
    if not _timing_data:
        return

    # Sort by execution time (longest first)
    sorted_times = sorted(
        [(path, data.get("elapsed", 0)) for path, data in _timing_data.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    total_time = sum(t for _, t in sorted_times)

    print("\n" + "=" * 60, file=sys.stderr)
    print("NOTEBOOK EXECUTION TIME SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for path, elapsed in sorted_times:
        short_path = Path(path).name
        print(f"  {elapsed:6.1f}s  {short_path}", file=sys.stderr)

    print("-" * 60, file=sys.stderr)
    print(
        f"  {total_time:6.1f}s  TOTAL ({len(sorted_times)} notebooks)", file=sys.stderr
    )
    print("=" * 60 + "\n", file=sys.stderr)
