"""The paper's spatial figures drawn from the campaign's inspected runs (evaluation topology 202, final episode, beta
0.95): the spatial filmstrip, the rounds per node, the GIF of the fields around the switch, and the dynamics of the
mean gradient around the switch over the training seeds.

The MLP's runs record the spatial snapshots of the paper (as its TD3 did); the GNN's runs record the attention
snapshots, which hold the same fields from 85 s on and share the topology of the MLP's run.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from experiments.campaign import LEARNED, TRAIN_SEEDS
from . import style
from .campaign_snapshots import EMBEDDINGS, INSPECTED_EVALUATION_SEED, SNAPSHOT_BETA
from .figures import spatial
from .campaign_paper import architecture
from .figures.common import method_band

# The snapshots and the communication edges of the inspected MLP run, as the paper's figure functions read them.
PAPER_SNAPSHOTS = f"spatial_snapshots_weight-{SNAPSHOT_BETA}.csv"
PAPER_EDGES = f"spatial_edges_weight-{SNAPSHOT_BETA}.csv"
ZOOM_S = (85.0, 165.0)
REFERENCE_PERIODS = (3.0, 5.0)
GIF = "spatial_wave_animation.gif"


def inspected_files(home: Path, jobs: list[dict], finals: dict[int, str], algorithm: int,
                    training_seed: int) -> dict[str, Path]:
    """The snapshot files of the static run of an architecture's final variant on the inspected topology."""
    job = next(job for job in jobs if job["family"] == "static" and job["algorithm"] == algorithm
               and job["variant"] == finals[algorithm] and job["beta"] == SNAPSHOT_BETA
               and job["training_seed"] == training_seed and job["evaluation_seed"] == INSPECTED_EVALUATION_SEED)
    return {path.name: path for path in (home / "exports" / job["key"]).rglob("*.csv") if "spatial" in path.parts}


def shim(folder: Path, snapshot_file: Path, edge_file: Path) -> SimpleNamespace:
    """A folder laid out as the paper's ``spatial`` exports (at beta 0.95), for the paper's figure functions."""
    (folder / PAPER_SNAPSHOTS).symlink_to(snapshot_file.resolve())
    (folder / PAPER_EDGES).symlink_to(edge_file.resolve())
    return SimpleNamespace(spatial_dir=folder)


def spatial_figures(home: Path, jobs: list[dict], finals: dict[int, str], out: Path) -> None:
    """Filmstrip, rounds per node and GIF of the MLP (as the paper's TD3); filmstrip and GIF of the GNN."""
    seed = TRAIN_SEEDS["test"][0]
    mlp = inspected_files(home, jobs, finals, LEARNED[1], seed)
    gnn = inspected_files(home, jobs, finals, LEARNED[0], seed)
    with tempfile.TemporaryDirectory() as folder:
        data = shim(Path(folder), mlp[PAPER_SNAPSHOTS], mlp[PAPER_EDGES])
        spatial.spatial_filmstrip(data, out / "filmstrip")
        spatial.spatial_computation(data, out / "static")
        spatial.spatial_wave_animation(data, out / "filmstrip")
    with tempfile.TemporaryDirectory() as folder:
        data = shim(Path(folder), gnn[EMBEDDINGS], mlp[PAPER_EDGES])
        drawn = Path(folder) / "drawn"
        spatial.spatial_filmstrip(data, drawn)
        spatial.spatial_wave_animation(data, drawn)
        for name in ("spatial_filmstrip.pdf", "spatial_filmstrip.png", GIF):
            stem, suffix = name.rsplit(".", 1)
            shutil.copy2(drawn / name, out / "filmstrip" / f"{stem}_gnn.{suffix}")


def network_means(path: Path) -> pd.DataFrame:
    """Network means over time of the gradient, of the ground truth and of the wake-up period."""
    frame = pd.read_csv(path)
    frame = frame[(frame.tEp >= ZOOM_S[0]) & (frame.tEp <= ZOOM_S[1])]
    means = frame.groupby("tEp")[["gradient", "groundTruth", "nextWakeUp"]].mean()
    return means.assign(deviation=(means.gradient - means.groundTruth).abs())


def seed_traces(home: Path, jobs: list[dict], finals: dict[int, str], algorithm: int) -> dict[int, pd.DataFrame]:
    name = EMBEDDINGS if algorithm == LEARNED[0] else PAPER_SNAPSHOTS
    return {seed: network_means(inspected_files(home, jobs, finals, algorithm, seed)[name])
            for seed in TRAIN_SEEDS["test"]}


def source_switch_dynamics(home: Path, jobs: list[dict], finals: dict[int, str], out: Path) -> None:
    """Mean gradient vs. ground truth (first seed), its deviation and the wake-up period around the switch."""
    fig, axes = style.figure("double", 3.9, 3, 1, sharex=True)
    gradient_ax, deviation_ax, period_ax = axes
    grid = np.arange(ZOOM_S[0], ZOOM_S[1] + 0.5, 1.0)
    shown = TRAIN_SEEDS["test"][0]
    for algorithm in LEARNED:
        method = architecture(algorithm, finals)
        traces = seed_traces(home, jobs, finals, algorithm)

        def matrix(column: str) -> np.ndarray:
            return np.stack([np.interp(grid, trace.index, trace[column]) for trace in traces.values()])

        gradient_ax.plot(grid, np.interp(grid, traces[shown].index, traces[shown].gradient), color=method.color,
                         linewidth=1.0)
        method_band(deviation_ax, grid, matrix("deviation"), method, markevery=6, nonnegative=True)
        method_band(period_ax, grid, matrix("nextWakeUp"), method, markevery=6)
        truth = traces[shown]
    gradient_ax.plot(grid, np.interp(grid, truth.index, truth.groundTruth), color=style.INK, linestyle=":",
                     linewidth=1.1, zorder=5)
    for period in REFERENCE_PERIODS:
        period_ax.plot(grid, np.full_like(grid, period), color=style.fixed_period_color(period), linestyle="--",
                       linewidth=0.9)
    gradient_ax.set_ylabel(f"Mean gradient (m)\ntraining seed {shown}")
    deviation_ax.set_ylabel("$|\\bar g - \\bar g^*|$ (m)")
    period_ax.set_ylabel("Mean wake-up period (s)")
    period_ax.set_ylim(0.4, 5.4)
    period_ax.set_xlabel("Time within the final episode (s), test topology "
                         f"{INSPECTED_EVALUATION_SEED}, {len(TRAIN_SEEDS['test'])} training seeds")
    period_ax.set_xlim(*ZOOM_S)
    for index, ax in enumerate(axes):
        style.mark_source_switch(ax, label=ax is period_ax, label_y=0.03)
        style.panel_label(ax, "abc"[index])
    handles = [style.legend_handle(architecture(algorithm, finals)) for algorithm in LEARNED]
    handles += [Line2D([], [], color=style.fixed_period_color(period), linestyle="--", linewidth=0.9,
                       label=f"Fixed {period:g} s") for period in REFERENCE_PERIODS]
    handles.append(Line2D([], [], color=style.INK, linestyle=":", linewidth=1.1, label="Ground truth $\\bar g^*$"))
    style.figure_legend(fig, handles, ncol=3)
    style.save(fig, out / "static", "source_switch_dynamics")
