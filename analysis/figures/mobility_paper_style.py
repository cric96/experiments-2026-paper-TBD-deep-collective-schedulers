"""The figures of the static paper (``charts/paper``) for one scenario of the mobile experiment.

The trade-off, spatial and attention figures are the paper's own functions on the mobile runs
(``mobility_paper.paper_dataset``): same metrics over the whole episode, same fixed-period frontier (1-5 s), same
bootstrap intervals over seeds, same inspected run (seed 2, last episode). The mobile exports have no mean gradient, so
the two dynamics figures show the mean absolute error of the nodes instead of the deviation of the mean gradient,
averaged over the 10 evaluation episodes instead of the final one. In the dynamic scenario the nodes move, so the maps
are triangulated and linked again at every instant (``moving_filmstrip``), and ``attention_on_motion`` shows whether
the GNN looks at the neighbours that move.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.tri import Triangulation
from scipy.spatial.distance import cdist

from .. import config, style
from ..dataset import Dataset
from ..mobility_paper import PAPER_ALGORITHMS, PAPER_FIXED, PAPER_SCENARIOS, REGION, PaperData, attention_on_motion
from ..mobility_regions import COMMUNICATION_RANGE_M
from ..methods import TD3, BY_KEY
from . import gnn, spatial, tradeoff
from .common import mean_band, method_band

ZOOM_S = (80.0, 165.0)  # window around the event, as in the paper
EPISODE_START_S = 4.0  # skips the first frame, as in the paper
REFERENCE_PERIODS = (3.0, 5.0)
EVENT_LABELS = {"static": "source moves", "dynamic": "region moves"}
EVENT_FIGURES = {"static": "source_switch_dynamics", "dynamic": "region_event_dynamics"}
MOVING_FILM_TIMES_S = (95.0, 105.0, 120.0, 145.0, 165.0)  # before, while and after the region moves
ATTENTION_WINDOW_S = (86.0, 160.0)
LEGEND_HEADROOM = 0.35  # room above the curves for the legend
PERIOD_LABEL = "Mean wake-up period (s)"
ERROR_LABEL = "Mean absolute error (m)"
TIME_LABEL = "Time within the evaluation episodes (s)"
ALGORITHM_OF = {key: algorithm for algorithm, key in PAPER_ALGORITHMS.items()}
FIXED_OF = {period: algorithm for algorithm, period in PAPER_FIXED.items()}


def _fixed_style(period: float) -> dict:
    return {"color": style.fixed_period_color(period), "linestyle": "--", "linewidth": 0.9}


def _matrix(series: pd.DataFrame, algorithm: int, beta: float, column: str, window: tuple[float, float]) -> tuple:
    """Seeds x frames matrix of ``column`` ("period" is the inverse of the round rate) within ``window``."""
    rows = series[(series.algorithm == algorithm) & np.isclose(series.beta, beta)
                  & series.t.between(*window)]
    values = rows.assign(period=1.0 / rows.rate).pivot_table(index="seed", columns="t", values=column)
    return values.columns.to_numpy(), values.to_numpy()


def _event(ax, name: str, label: bool) -> None:
    ax.axvline(config.SOURCE_SWITCH_S, color=style.INK_MUTED, linewidth=0.7, linestyle=":", zorder=1)
    if name == "dynamic":
        ax.axvspan(config.SOURCE_SWITCH_S, REGION.motion_end_s, color=style.PHASE_SHADE, linewidth=0, zorder=0)
    if label:
        ax.text(config.SOURCE_SWITCH_S, 0.03, f" {EVENT_LABELS[name]}", transform=ax.get_xaxis_transform(),
                fontsize=6, color=style.INK_MUTED, ha="left", va="bottom")


def source_switch_dynamics(series: pd.DataFrame, name: str, out: Path) -> None:
    """Error and wake-up period around the event (beta = 0.95): the learners against two fixed periods."""
    fig, (error_ax, period_ax) = style.figure("double", 2.8, 2, 1, sharex=True)
    for method in (TD3, BY_KEY["gnn"]):
        for ax, column in ((error_ax, "absoluteError"), (period_ax, "period")):
            times, matrix = _matrix(series, ALGORITHM_OF[method.key], config.FOCUS_WEIGHT, column, ZOOM_S)
            method_band(ax, times, matrix, method, markevery=3, nonnegative=True)
    for period in REFERENCE_PERIODS:
        times, matrix = _matrix(series, FIXED_OF[period], config.FOCUS_WEIGHT, "absoluteError", ZOOM_S)
        mean_band(error_ax, times, matrix, band=False, **_fixed_style(period))
        period_ax.plot(times, np.full_like(times, period), **_fixed_style(period))
    error_ax.set_ylabel(ERROR_LABEL)
    period_ax.set_ylabel(PERIOD_LABEL)
    period_ax.set_ylim(0.8, 5.4)
    period_ax.set_xlabel(TIME_LABEL)
    period_ax.set_xlim(*ZOOM_S)
    for index, ax in enumerate((error_ax, period_ax)):
        _event(ax, name, label=ax is period_ax)
        style.panel_label(ax, "ab"[index])
    handles = [style.legend_handle(method) for method in (TD3, BY_KEY["gnn"])]
    handles += [Line2D([], [], label=f"Fixed {period:g} s", **_fixed_style(period)) for period in REFERENCE_PERIODS]
    style.figure_legend(fig, handles, ncol=4)
    style.save(fig, out, EVENT_FIGURES[name])


def beta_effect_intra_episode(series: pd.DataFrame, name: str, out: Path) -> None:
    """How beta shapes the wake-up schedule and the error of TD3 through the episode."""
    fig, (period_ax, error_ax) = style.figure("single", 3.3, 2, 1, sharex=True)
    window = (EPISODE_START_S, config.EPISODE_LENGTH_S)
    for weight in config.WEIGHTS:
        color = style.BETA_COLORS[weight]
        times, periods = _matrix(series, ALGORITHM_OF[TD3.key], weight, "period", window)
        mean_band(period_ax, times, periods, color, f"{weight:g}", linewidth=1.0)
        times, errors = _matrix(series, ALGORITHM_OF[TD3.key], weight, "absoluteError", window)
        mean_band(error_ax, times, errors, color, linewidth=1.0, nonnegative=True)
    period_ax.set_ylabel(PERIOD_LABEL)
    period_ax.set_ylim(0.8, 5.2)
    error_ax.set_ylabel(ERROR_LABEL)
    error_ax.set_xlabel(TIME_LABEL)
    error_ax.set_xlim(0, config.EPISODE_LENGTH_S)
    for index, ax in enumerate((period_ax, error_ax)):
        _event(ax, name, label=ax is error_ax)
        style.panel_label(ax, "ab"[index])
    handles, labels = period_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), title=f"{TD3.short}, $\\beta$ =")
    style.save(fig, out, "beta_effect_intra_episode")


class _Frame:
    """What ``spatial._field`` draws over: the triangulation and the links of the nodes at one instant."""

    def __init__(self, nodes: pd.DataFrame) -> None:
        positions = nodes[["x", "y"]].to_numpy()
        self.triangulation = Triangulation(nodes.x, nodes.y)
        upper = np.triu(cdist(positions, positions) <= COMMUNICATION_RANGE_M, k=1)
        first, second = np.nonzero(upper)
        self.edges = pd.DataFrame(dict(xU=positions[first, 0], yU=positions[first, 1], xV=positions[second, 0],
                                       yV=positions[second, 1]))


def moving_filmstrip(dataset: Dataset, out: Path) -> None:
    """The paper's spatial filmstrip (gradient, round rate, error) before, while and after the region moves; the
    nodes that moved in the last second are circled."""
    snapshots = spatial.Snapshots.load(dataset)
    if snapshots is None:
        return
    fig, axes = style.figure("double", 4.6, 3, len(MOVING_FILM_TIMES_S), sharex=True, sharey=True)
    for column, time in enumerate(MOVING_FILM_TIMES_S):
        nodes, previous = snapshots.at(time), snapshots.at(time - 1.0)
        moved = np.hypot(nodes.x - previous.x, nodes.y - previous.y) > 0.0
        for row, (values, cmap, value_range, label) in enumerate(spatial._rows(nodes)):
            ax = axes[row, column]
            filled = spatial._field(ax, _Frame(nodes), nodes, values, cmap, value_range)
            ax.scatter(nodes.x[moved], nodes.y[moved], s=12, facecolors="none", edgecolors=style.INK, linewidths=0.5,
                       zorder=4)
            if column == len(MOVING_FILM_TIMES_S) - 1:
                bar = fig.colorbar(filled, ax=axes[row, :].tolist(), fraction=0.02, pad=0.01,
                                   ticks=np.linspace(*value_range, 5), format="%g")
                bar.set_label(label, fontsize=7)
                bar.outline.set_visible(False)
        axes[0, column].set_title(f"$t$ = {time:g} s")
        axes[-1, column].set_xlabel("$x$ (m)")
    for row in range(3):
        axes[row, 0].set_ylabel("$y$ (m)")
    style.save(fig, out, "spatial_filmstrip")


def attention_to_motion(dataset: Dataset, out: Path) -> None:
    """(a) Among the still nodes next to a moving one, how often the GNN attends most to a moving neighbour, against a
    random choice; (b) the embedding PC1 (> 0: active) of the moving and of the still nodes."""
    frame = gnn.load_embeddings(dataset)
    if frame is None:
        return
    shares = attention_on_motion(frame)
    shares = shares[shares.tEp.between(*ATTENTION_WINDOW_S)]
    fig, (share_ax, pc1_ax) = style.figure("single", 3.2, 2, 1, sharex=True)
    gnn_method = BY_KEY["gnn"]
    share_ax.plot(shares.tEp, shares.attended, color=gnn_method.color, linewidth=1.0, label="most attended neighbour")
    share_ax.plot(shares.tEp, shares.chance, color=style.INK_MUTED, linewidth=0.9, linestyle="--",
                  label="random neighbour")
    share_ax.set_ylabel("Share that moves")
    share_ax.set_ylim(0, 1)
    share_ax.legend(loc="upper left", ncol=2, fontsize=6)
    pc1_ax.plot(shares.tEp, shares.pc1_moving, color=style.INK, linewidth=1.0, label="moving nodes")
    pc1_ax.plot(shares.tEp, shares.pc1_still, color=style.INK_MUTED, linewidth=1.0, label="still nodes")
    pc1_ax.set_ylabel("Mean embedding PC1")
    pc1_ax.set_xlabel("Time within the last episode (s)")
    low, high = pc1_ax.get_ylim()
    pc1_ax.set_ylim(low, high + LEGEND_HEADROOM * (high - low))
    pc1_ax.legend(loc="upper left", ncol=2, fontsize=6)
    for index, ax in enumerate((share_ax, pc1_ax)):
        _event(ax, "dynamic", label=ax is pc1_ax)
        style.panel_label(ax, "ab"[index])
    style.save(fig, out, "attention_on_motion")


def _series(data: PaperData, name: str) -> pd.DataFrame:
    if name == "static":
        return data.static_series
    _, level = PAPER_SCENARIOS[name]
    return data.region_network[np.isclose(data.region_network.diffusion, level)]


def render(dataset: Dataset, data: PaperData, name: str, out: Path) -> None:
    """The paper's figures of scenario ``name`` ("static" or "dynamic") into ``out``."""
    style.apply()
    out.mkdir(parents=True, exist_ok=True)
    for figure in tradeoff.FIGURES:
        figure(dataset, out)
    series = _series(data, name)
    source_switch_dynamics(series, name, out)
    beta_effect_intra_episode(series, name, out)
    gnn.gnn_attention_filmstrip(dataset, out)
    if name == "static":
        for figure in (*spatial.FIGURES, gnn.gnn_latent_dynamics):
            figure(dataset, out)
    else:
        moving_filmstrip(dataset, out)
        attention_to_motion(dataset, out)
    plt.close("all")
