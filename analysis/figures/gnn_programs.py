"""GNN responses to a source move or elected-leader failure on programs unseen during training.

All rows use seed 2, beta=0.95, episode 16. Arrows point to the most-attended
neighbour; embedding colours use a common PCA fitted across all programs.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from .. import config, style
from ..dataset import Dataset
from .explain import pca
from .gnn import ARROW_SHRINK, FILM_TIMES_S
from .spatial import _draw_sources

ELECTION_FILM_TIMES_S = (95.0, 103.0, 115.0, 150.0, 198.0)
# The run shown in the titles: the paper's generalization snapshots (other campaigns pass their own).
PAPER_RUN = "seed 2, episode 16"

PROGRAM_FOLDERS = {"gradient": "gradient", "hop": "hopGradient", "broadcast": "broadcast", "collect": "collect", "leaderElection": "leaderElection"}


def load_program_embeddings(data: Dataset) -> dict[str, pd.DataFrame]:
    frames = {}
    for program, folder in PROGRAM_FOLDERS.items():
        path = data.spatial_dir / "generalization" / folder / "gnn_spatiotemporal_embeddings.csv"
        if path.exists():
            frames[program] = pd.read_csv(path)
    if not frames:
        warnings.warn("No program attention snapshots: run the GNN generalization simulations (see README).", stacklevel=2)
        return {}
    missing = set(PROGRAM_FOLDERS) - frames.keys()
    if missing:
        warnings.warn(f"Program attention snapshots missing: {', '.join(sorted(missing))}", stacklevel=2)
    columns = [c for c in next(iter(frames.values())).columns if c.startswith("h_")]
    joined = pd.concat(list(frames.values()), ignore_index=True)
    scores, _ = pca(joined[columns].to_numpy())
    if np.std(joined.nextWakeUp) > 0 and np.std(scores[:, 0]) > 0:
        if np.corrcoef(scores[:, 0], -joined.nextWakeUp)[0, 1] < 0:
            scores[:, 0] *= -1
    start = 0
    for program, frame in frames.items():
        frames[program] = frame.assign(pc1=scores[start:start + len(frame), 0])
        start += len(frame)
    return frames


def _snapshot(frame: pd.DataFrame, requested: float) -> pd.DataFrame:
    times = np.unique(frame.tEp)
    nearest = times[np.argmin(np.abs(times - requested))]
    if abs(nearest - requested) > 1.1:
        raise ValueError(f"No snapshot within 1.1 s of {requested:g} s")
    return frame[np.isclose(frame.tEp, nearest)].drop_duplicates("nodeId").sort_values("nodeId")


def _norm(frames: dict[str, pd.DataFrame]) -> TwoSlopeNorm:
    # Include every value, so colours never hide outliers by clipping.
    values = np.concatenate([f.pc1.to_numpy() for f in frames.values()])
    return TwoSlopeNorm(vcenter=0, vmin=min(float(values.min()), -1e-3), vmax=max(float(values.max()), 1e-3))


def _map(ax, nodes, values, cmap, **kwargs):
    points = ax.scatter(nodes.x, nodes.y, c=values, cmap=cmap, s=12,
                        linewidths=0.2, edgecolors=style.INK_MUTED, zorder=3, **kwargs)
    _draw_sources(ax, nodes)
    ax.set_aspect("equal")
    ax.grid(False)
    return points


def _leader_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    return nodes[nodes.isLeader > 0.5]


def _leaders(ax, nodes):
    elected = _leader_nodes(nodes)
    ax.scatter(elected.x, elected.y, s=26, marker="s", facecolors="none",
               edgecolors="#ae3903", linewidths=0.8, zorder=4)


def _failed_nodes(frame: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    initial = frame[np.isclose(frame.tEp, frame.tEp.min())].drop_duplicates("nodeId")
    return initial[~initial.nodeId.isin(nodes.nodeId)]


def _mark_failed(ax, frame, nodes):
    failed = _failed_nodes(frame, nodes)
    ax.scatter(failed.x, failed.y, s=30, marker="x", color=style.INK,
               linewidths=1.2, zorder=6)


def _arrows(ax, nodes):
    ax.quiver(nodes.x, nodes.y,
              (nodes.bestNbrX - nodes.x) * ARROW_SHRINK,
              (nodes.bestNbrY - nodes.y) * ARROW_SHRINK,
              angles="xy", scale_units="xy", scale=1,
              color=style.INK_SECONDARY, width=0.006,
              headwidth=3.5, headlength=4, zorder=2)


def _bar(fig, points, axes, label):
    bar = fig.colorbar(points, ax=axes, fraction=0.018, pad=0.01)
    bar.set_label(label, fontsize=7)
    bar.outline.set_visible(False)


def gnn_program_attention_filmstrip(data: Dataset, out: Path, run: str = PAPER_RUN) -> None:
    """One row per program, shared embedding colours and attention arrows."""
    frames = load_program_embeddings(data)
    if not frames:
        return
    norm = _norm(frames)
    fig, axes = style.figure("double", 1.5 * len(frames) + 0.5,
                             len(frames), len(FILM_TIMES_S), sharex=True, sharey=True, squeeze=False, layout="none")
    # Fixed-aspect maps and a colourbar spanning five rows need explicit margins.
    fig.subplots_adjust(left=0.09, right=0.92, bottom=0.055, top=0.93, wspace=0.06, hspace=0.16)
    for row, (program, frame) in enumerate(frames.items()):
        for col, t in enumerate(FILM_TIMES_S):
            ax = axes[row, col]
            nodes = _snapshot(frame, t)
            points = _map(ax, nodes, nodes.pc1, style.DIVERGING, norm=norm)
            _arrows(ax, nodes)
            if program == "leaderElection":
                _leaders(ax, nodes)
                _mark_failed(ax, frame, nodes)
            if row == 0:
                ax.set_title(f"$t$ = {nodes.tEp.iloc[0]:g} s")
            if row == len(frames) - 1:
                ax.set_xlabel("$x$ (m)")
        axes[row, 0].set_ylabel(f"{config.PROGRAM_SHORT_LABELS[program]}\n$y$ (m)")
    _bar(fig, points, axes.ravel().tolist(), "Embedding PC1 (shared PCA)")
    fig.suptitle(f"GNN: {run}, $\\beta=0.95$; source move / elected-leader failure at 100 s\n"
                 "arrows: most-attended neighbour; stars: source / initial boost; squares: leaders; crosses: failed node", fontsize=7)
    # savefig restores a None engine between formats; keep the manual layout active.
    with mpl.rc_context({"figure.constrained_layout.use": False}):
        style.save(fig, out, "gnn_program_attention_filmstrip")


def _leader_id_colours(shown: pd.DataFrame):
    identifiers = np.unique(np.concatenate([shown.gradient, shown.groundTruth])).astype(int)
    cmap = mpl.colormaps["tab20"].resampled(len(identifiers))
    norm = mpl.colors.BoundaryNorm(np.arange(len(identifiers) + 1) - 0.5, len(identifiers))
    return identifiers, cmap, norm


def gnn_program_response_filmstrips(data: Dataset, out: Path, run: str = PAPER_RUN) -> None:
    """Output, chosen round rate, error, and embedding/attention for each program."""
    frames = load_program_embeddings(data)
    if not frames:
        return
    norm = _norm(frames)
    for program, frame in frames.items():
        times = ELECTION_FILM_TIMES_S if program == "leaderElection" else FILM_TIMES_S
        snapshots = [_snapshot(frame, t) for t in times]
        shown = pd.concat(snapshots, ignore_index=True)
        output_range = (float(min(shown.gradient.min(), shown.groundTruth.min())),
                        float(max(shown.gradient.max(), shown.groundTruth.max())))
        if output_range[0] == output_range[1]:
            output_range = (output_range[0] - 0.5, output_range[1] + 0.5)
        error_max = max(float(shown.error.max()), 1e-3)
        rates = 1 / shown.nextWakeUp
        rate_range = (min(0.2, float(rates.min())), max(1.0, float(rates.max())))
        fig, axes = style.figure("double", 6.6, 4, len(FILM_TIMES_S), sharex=True, sharey=True, layout="none")
        fig.subplots_adjust(left=0.07, right=0.90, bottom=0.05, top=0.92, wspace=0.05, hspace=0.18)
        election = program == "leaderElection"
        identifiers, id_cmap, id_norm = _leader_id_colours(shown) if election else (None, None, None)
        labels = ("Leader ID" if election else "Program output", "Round rate (Hz)",
                  "Wrong leader ID (0/1)" if election else "Absolute error", "Embedding PC1 (shared PCA)")
        for col, nodes in enumerate(snapshots):
            output = (np.searchsorted(identifiers, nodes.gradient), id_cmap, dict(norm=id_norm)) if election else (
                nodes.gradient, style.SEQUENTIAL, dict(vmin=output_range[0], vmax=output_range[1]))
            rows = (output,
                    (1 / nodes.nextWakeUp, style.SEQUENTIAL_2, dict(vmin=rate_range[0], vmax=rate_range[1])),
                    (nodes.error, style.SEQUENTIAL_3, dict(vmin=0, vmax=error_max)),
                    (nodes.pc1, style.DIVERGING, dict(norm=norm)))
            for row, (values, cmap, options) in enumerate(rows):
                points = _map(axes[row, col], nodes, values, cmap, **options)
                if program == "leaderElection":
                    _mark_failed(axes[row, col], frame, nodes)
                if row == 3:
                    _arrows(axes[row, col], nodes)
                    if program == "leaderElection":
                        _leaders(axes[row, col], nodes)
                if col == len(snapshots) - 1:
                    if election and row == 0:
                        bar = fig.colorbar(points, ax=axes[row, :].tolist(), fraction=0.018, pad=0.01,
                                           ticks=np.arange(len(identifiers)))
                        bar.ax.set_yticklabels([str(identifier) for identifier in identifiers])
                        bar.set_label(labels[row], fontsize=7)
                        bar.outline.set_visible(False)
                    else:
                        _bar(fig, points, axes[row, :].tolist(), labels[row])
            axes[0, col].set_title(f"$t$ = {nodes.tEp.iloc[0]:g} s")
            axes[-1, col].set_xlabel("$x$ (m)")
        for row in range(4):
            axes[row, 0].set_ylabel("$y$ (m)")
        if program == "leaderElection":
            event = f"elected leader removed at 100 s; grain {config.LEADER_ELECTION_GRAIN_M:g} m; crosses: failed node"
        else:
            event = "source moves at 100 s; stars: source"
        fig.suptitle(f"{config.PROGRAM_LABELS[program]}: GNN, {run}, $\\beta=0.95$\n"
                     f"{event}; arrows: most-attended neighbour", fontsize=7)
        with mpl.rc_context({"figure.constrained_layout.use": False}):
            style.save(fig, out, f"gnn_response_filmstrip_{program}")


FIGURES = (gnn_program_attention_filmstrip, gnn_program_response_filmstrips)
