# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation

pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning"
)


@pytest.fixture
def animated_network(ac_dc_network):
    """Network with synthetic time-varying data for animation tests."""
    n = ac_dc_network
    snapshots = pd.date_range("2025-01-01", periods=5, freq="h")
    n.set_snapshots(snapshots)
    return n


@pytest.mark.skipif(os.name == "nt", reason="tcl_findLibrary on Windows")
def test_animate_basic(animated_network):
    n = animated_network
    line_width = pd.DataFrame(
        np.random.default_rng(42).random((len(n.snapshots), len(n.lines))) + 0.5,
        index=n.snapshots,
        columns=n.lines.index,
    )
    anim = n.plot.animate(line_width=line_width, geomap=False, fps=2)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


@pytest.mark.skipif(os.name == "nt", reason="tcl_findLibrary on Windows")
def test_animate_bus_sizes(animated_network):
    n = animated_network
    bus_size = pd.DataFrame(
        np.random.default_rng(42).random((len(n.snapshots), len(n.buses))) * 0.05,
        index=n.snapshots,
        columns=n.buses.index,
    )
    anim = n.plot.animate(bus_size=bus_size, geomap=False)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


@pytest.mark.skipif(os.name == "nt", reason="tcl_findLibrary on Windows")
def test_animate_default_snapshots(animated_network):
    n = animated_network
    line_width = pd.DataFrame(
        np.ones((len(n.snapshots), len(n.lines))),
        index=n.snapshots,
        columns=n.lines.index,
    )
    anim = n.plot.animate(line_width=line_width, geomap=False)
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


@pytest.mark.skipif(os.name == "nt", reason="tcl_findLibrary on Windows")
def test_animate_snapshot_subset(animated_network):
    n = animated_network
    line_width = pd.DataFrame(
        np.ones((len(n.snapshots), len(n.lines))),
        index=n.snapshots,
        columns=n.lines.index,
    )
    anim = n.plot.animate(
        snapshots=n.snapshots[:3], line_width=line_width, geomap=False
    )
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


@pytest.mark.skipif(os.name == "nt", reason="tcl_findLibrary on Windows")
def test_animate_timestamp(animated_network):
    n = animated_network
    anim = n.plot.animate(geomap=False)
    anim._init_draw()
    anim._step(0)
    ax = plt.gca()
    texts = [t for t in ax.texts if t.get_text() != ""]
    assert len(texts) > 0, "Timestamp text should be present on axis"
    plt.close("all")


@pytest.mark.skipif(os.name == "nt", reason="tcl_findLibrary on Windows")
def test_animate_timestamp_format(animated_network):
    n = animated_network
    anim = n.plot.animate(
        geomap=False, timestamp_kwargs={"format": "%H:%M", "fontsize": 14}
    )
    anim._init_draw()
    anim._step(0)
    ax = plt.gca()
    texts = [t for t in ax.texts if t.get_text() != ""]
    assert texts[0].get_text() == "00:00"
    plt.close("all")


@pytest.mark.skipif(os.name == "nt", reason="tcl_findLibrary on Windows")
def test_animate_static_and_dynamic_params(animated_network):
    n = animated_network
    line_width = pd.DataFrame(
        np.random.default_rng(42).random((len(n.snapshots), len(n.lines))) + 0.5,
        index=n.snapshots,
        columns=n.lines.index,
    )
    anim = n.plot.animate(
        line_width=line_width, bus_color="red", geomap=False
    )
    assert isinstance(anim, FuncAnimation)
    plt.close("all")


@pytest.mark.skipif(os.name == "nt", reason="tcl_findLibrary on Windows")
def test_animate_save(animated_network, tmp_path):
    n = animated_network
    filepath = str(tmp_path / "test_anim.gif")
    result = n.plot.animate(
        geomap=False, fps=2, path=filepath, writer="pillow"
    )
    assert result is None
    assert (tmp_path / "test_anim.gif").exists()
