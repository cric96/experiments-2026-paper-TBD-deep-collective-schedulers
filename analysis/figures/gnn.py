"""Spatio-temporal analysis of the GNN policy's node embeddings (seed 2, beta = 0.95, final episode).

Data: ``spatial/gnn_spatiotemporal_embeddings.csv`` -- every 1 s between 85 s and
160 s of the final episode, for each node: position, source flag, wake-up
period, the coordinates of its most-attended neighbour and 16 embedding
dimensions ``h_*``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as sps

from .. import config, style
from ..dataset import Dataset
from .explain import pca

FILM_TIMES_S = (95.0, 103.0, 108.0, 115.0, 138.0)
BEFORE_SWITCH_S, AFTER_SWITCH_S = 95.0, 140.0  # instants used to locate the old / new source
HOVMOLLER_WINDOW_S = (88.0, 150.0)
DISTANCE_BINS_M = np.linspace(0.0, 480.0, 25)
ZONE_RADIUS_M = 140.0
ARROW_SHRINK = 0.72  # attention arrows stop short of the neighbour
ORBIT_LABEL_TIMES_S = (106.0, 112.0, 120.0, 128.0)
# A time label is left out when its point is closer than this (a fraction of the ranges of the trajectory) to the
# steady state or to a labelled point: the labels would overlap.
ORBIT_LABEL_SPACING = 0.12


@dataclass
class WaveFront:
    """Linear fit of the time at which each node's PC1 peaks against its distance to the new source."""

    speed: float  # m/s
    low: float
    high: float
    onset: float  # time at which the front leaves the source (s)
    r: float


def load_embeddings(data: Dataset) -> pd.DataFrame | None:
    """GNN embedding snapshots with PC1/PC2 of the embedding (PC1 > 0 = active node); None if missing."""
    path = data.spatial_dir / "gnn_spatiotemporal_embeddings.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    scores, _ = pca(frame[[c for c in frame.columns if c.startswith("h_")]].to_numpy())
    if np.corrcoef(scores[:, 0], -frame.nextWakeUp)[0, 1] < 0:  # PC1 > 0 = active (short period)
        scores[:, 0] = -scores[:, 0]
    return frame.assign(pc1=scores[:, 0], pc2=scores[:, 1])


def _at(frame: pd.DataFrame, t: float) -> pd.DataFrame:
    times = np.unique(frame.tEp)
    nearest = times[np.argmin(np.abs(times - t))]
    return frame[np.isclose(frame.tEp, nearest)].drop_duplicates("nodeId").sort_values("nodeId")


def _source(frame: pd.DataFrame, t: float) -> np.ndarray:
    nodes = _at(frame, t)
    return nodes.loc[nodes.isSource > 0.5, ["x", "y"]].to_numpy()[0]


def _pc1_norm(frame: pd.DataFrame) -> TwoSlopeNorm:
    low, high = np.percentile(frame.pc1, [2, 98])
    return TwoSlopeNorm(vcenter=0.0, vmin=min(low, -1e-3), vmax=max(high, 1e-3))


def _pc1_colorbar(fig, mappable, ax, norm: TwoSlopeNorm, **kwargs) -> None:
    ticks = [v for v in np.arange(-2.0, 2.01, 0.5) if norm.vmin <= v <= norm.vmax]
    bar = fig.colorbar(mappable, ax=ax, ticks=ticks, format="%g", **kwargs)
    bar.set_label("Embedding PC1", fontsize=7)
    bar.outline.set_visible(False)


def wave_front(frame: pd.DataFrame) -> WaveFront:
    """Speed of the activity front after the switch, with a 95 % CI on the regression slope."""
    new_source = _source(frame, AFTER_SWITCH_S)
    after = frame[(frame.tEp >= config.SOURCE_SWITCH_S) & (frame.tEp <= HOVMOLLER_WINDOW_S[1])]
    peaks = after.loc[after.groupby("nodeId").pc1.idxmax()]
    distance = np.hypot(peaks.x - new_source[0], peaks.y - new_source[1])
    fit = sps.linregress(peaks.tEp, distance)
    half = sps.t.ppf(0.5 + config.CONFIDENCE / 2, len(peaks) - 2) * fit.stderr
    return WaveFront(fit.slope, fit.slope - half, fit.slope + half, -fit.intercept / fit.slope, fit.rvalue)


def gnn_attention_filmstrip(data: Dataset, out: Path) -> None:
    """Embedding PC1 over the network and each node's most-attended neighbour, around the switch."""
    frame = load_embeddings(data)
    if frame is None:
        return
    norm = _pc1_norm(frame)
    fig, axes = style.figure("double", 1.85, 1, len(FILM_TIMES_S), sharex=True, sharey=True)
    for ax, t in zip(axes, FILM_TIMES_S):
        nodes = _at(frame, t)
        points = ax.scatter(
            nodes.x,
            nodes.y,
            c=nodes.pc1.clip(norm.vmin, norm.vmax),
            cmap=style.DIVERGING,
            norm=norm,
            s=12,
            linewidths=0.2,
            edgecolors=style.INK_MUTED,
            zorder=3,
        )
        ax.quiver(
            nodes.x,
            nodes.y,
            (nodes.bestNbrX - nodes.x) * ARROW_SHRINK,
            (nodes.bestNbrY - nodes.y) * ARROW_SHRINK,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=style.INK_SECONDARY,
            width=0.006,
            headwidth=3.5,
            headlength=4,
            zorder=2,
        )
        sources = nodes[nodes.isSource > 0.5]
        ax.plot(
            sources.x,
            sources.y,
            marker="*",
            markersize=8,
            color=style.INK,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linestyle="none",
            zorder=5,
            clip_on=False,
        )
        ax.set_title(f"$t$ = {t:g} s")
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("$x$ (m)")
    axes[0].set_ylabel("$y$ (m)")
    _pc1_colorbar(fig, points, axes.tolist(), norm, fraction=0.015, pad=0.01)
    style.save(fig, out, "gnn_attention_filmstrip")


def labelled_times(orbit: pd.DataFrame, steady: tuple[float, float]) -> list[float]:
    """The times of the trajectory nearest to ORBIT_LABEL_TIMES_S whose points stand apart from the steady state and
    from the points labelled before them: a trajectory back at the steady state needs no time label."""
    spans = (np.ptp(orbit.pc1), np.ptp(orbit.pc2))
    placed, times = [steady], []
    for target in ORBIT_LABEL_TIMES_S:
        nearest = orbit.index[np.argmin(np.abs(orbit.index - target))]
        point = (orbit.pc1[nearest], orbit.pc2[nearest])
        if all(np.hypot((point[0] - x) / spans[0], (point[1] - y) / spans[1]) >= ORBIT_LABEL_SPACING
               for x, y in placed):
            placed.append(point)
            times.append(nearest)
    return times


def gnn_latent_dynamics(data: Dataset, out: Path) -> None:
    """Space-time diagram with the fitted wave front, PC1 per zone, and the network's latent orbit."""
    frame = load_embeddings(data)
    if frame is None:
        return
    new_source, old_source = _source(frame, AFTER_SWITCH_S), _source(frame, BEFORE_SWITCH_S)
    window = frame[(frame.tEp >= HOVMOLLER_WINDOW_S[0]) & (frame.tEp <= HOVMOLLER_WINDOW_S[1])]
    window = window.assign(
        to_new=np.hypot(window.x - new_source[0], window.y - new_source[1]),
        to_old=np.hypot(window.x - old_source[0], window.y - old_source[1]),
    )
    times = np.unique(window.tEp)
    fig, (space_ax, zone_ax, orbit_ax) = style.figure("double", 2.3, 1, 3)
    fig.get_layout_engine().set(wspace=0.08)

    window = window.assign(bin=np.digitize(window.to_new, DISTANCE_BINS_M) - 1)
    grid = window.pivot_table(index="bin", columns="tEp", values="pc1", aggfunc="mean")
    grid = grid.reindex(index=range(len(DISTANCE_BINS_M) - 1))
    norm = _pc1_norm(frame)
    image = space_ax.imshow(
        grid.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap=style.DIVERGING,
        norm=norm,
        extent=[times[0], times[-1], DISTANCE_BINS_M[0], DISTANCE_BINS_M[-1]],
    )
    front = wave_front(frame)
    front_times = np.array([front.onset, front.onset + DISTANCE_BINS_M[-1] / front.speed])
    space_ax.plot(
        front_times,
        (front_times - front.onset) * front.speed,
        color=style.INK,
        linestyle="--",
        linewidth=0.8,
        label=f"front {front.speed:.1f} m/s\n[{front.low:.1f}, {front.high:.1f}]",
    )
    space_ax.set_xlim(times[0], times[-1])
    space_ax.legend(loc="lower right", fontsize=6, handlelength=1.2)
    space_ax.set_xlabel("Time (s)")
    space_ax.set_ylabel("Distance to the new source (m)")
    space_ax.grid(False)
    _pc1_colorbar(fig, image, space_ax, norm, fraction=0.05, pad=0.02)

    zones = (
        (
            f"< {ZONE_RADIUS_M:g} m from new source",
            window.to_new < ZONE_RADIUS_M,
            {"color": style.INK, "linestyle": "-"},
        ),
        (
            f"< {ZONE_RADIUS_M:g} m from old source",
            window.to_old < ZONE_RADIUS_M,
            {"color": style.INK, "linestyle": "--"},
        ),
        (
            "elsewhere",
            (window.to_new >= ZONE_RADIUS_M) & (window.to_old >= ZONE_RADIUS_M),
            {"color": style.INK_MUTED, "linestyle": "-"},
        ),
    )
    for label, mask, line in zones:
        series = window[mask].groupby("tEp").pc1.mean()
        zone_ax.plot(series.index, series.to_numpy(), linewidth=1.0, label=label, **line)
    style.mark_source_switch(zone_ax)
    zone_ax.set_xlabel("Time (s)")
    zone_ax.set_ylabel("Mean embedding PC1")
    zone_ax.set_ylim(top=zone_ax.get_ylim()[1] + 0.6)  # room for the legend above the curves
    zone_ax.legend(loc="upper right", fontsize=5.5, handlelength=1.5)

    orbit = window.groupby("tEp")[["pc1", "pc2", "nextWakeUp"]].mean()
    orbit_ax.plot(orbit.pc1, orbit.pc2, color=style.INK_MUTED, linewidth=0.6, zorder=1)
    points = orbit_ax.scatter(
        orbit.pc1, orbit.pc2, c=orbit.nextWakeUp, cmap=style.SEQUENTIAL, s=10, zorder=2, linewidths=0
    )
    steady = orbit.loc[orbit.index <= config.SOURCE_SWITCH_S]
    steady_point = (steady.pc1.mean(), steady.pc2.mean())
    for nearest in labelled_times(orbit, steady_point):
        style.direct_label(orbit_ax, orbit.pc1[nearest], orbit.pc2[nearest], f"{nearest:g} s", dx=3, dy=3, fontsize=5.5)
    style.direct_label(orbit_ax, *steady_point, "steady state", dx=2, dy=9, fontsize=5.5)
    orbit_ax.set_xlabel("Network-mean PC1")
    orbit_ax.set_ylabel("Network-mean PC2")
    bar = fig.colorbar(points, ax=orbit_ax, fraction=0.05, pad=0.02)
    bar.set_label("Mean wake-up period (s)", fontsize=7)
    bar.outline.set_visible(False)
    for ax, letter in zip((space_ax, zone_ax, orbit_ax), "abc"):
        style.panel_label(ax, letter)
    style.save(fig, out, "gnn_latent_dynamics")


FIGURES = (gnn_attention_filmstrip, gnn_latent_dynamics)
