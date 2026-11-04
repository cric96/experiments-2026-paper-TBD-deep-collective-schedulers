"""Figures of the mobile experiment (``analysis.mobility_paper``) besides the static paper's ones.

* ``static/rate_and_error``: round rate and error through the episode in the static network, learners against two
  fixed periods, over the error of the nodes cut off from the source;
* ``dynamic/region_dynamics``, ``region_routes``, ``region_snapshot``: where the rounds and the error go while the
  region moves (see ``figures.mobility_regions``);
* ``recap/gains``: gain over the fixed periods per scenario and phase, paired by seed, whole error and nodes connected
  to the source;
* ``recap/pareto_frontiers``: error vs. rounds of every beta against the fixed periods, per phase, static and dynamic;
* ``recap/tradeoff_vs_beta``: rounds and error (relative to a fixed 3 s period of the same seed) as functions of beta.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

from .. import config, style
from ..mobility_paper import (RECAP, REFERENCE_FIXED, REGION, STATIC_LEVEL, STATIC_VARIANT, SWEEP_LEVEL, PaperData,
                              beta_response)
from ..mobility_pareto import FIXED_PERIODS, METHODS
from ..mobility_static import PHASE_BY_KEY
from ..mobility_variants import CONNECTED, FOCUS_BETA, HEURISTIC, LEARNERS
from ..stats import t_interval
from .mobility_final import (BRACKET, ERROR_SCALE_M, FLOOR_COLOUR, TOPOLOGY_CARRIER, _fixed_colour, _floor_handle,
                             _frontier_panel, _learner_handles, _metres, _source_changes)
from .mobility_regions import region_dynamics, region_routes, region_snapshot
from .mobility_static import _mean_series, _rate_series
from .mobility_variants import GAIN_LIMIT, _draw_cells, _gain_colourbar

SCHEDULERS = (*LEARNERS, HEURISTIC)
LABELS = {0: "GNN", 1: "TD3", HEURISTIC: "Heuristic"}  # the paper's names; one training only
FRONTIER_PHASES = ("tracking", "event", "settled")
GAIN_ROWS = (  # (variant, level, label)
    (STATIC_VARIANT, STATIC_LEVEL, "Static: source switch"),
    (REGION.variant, 1.0, "Dynamic: region at 1 m/s"),
    (REGION.variant, 2.0, "Dynamic: region at 2 m/s"),
)
GAIN_COLUMNS = (("bootstrap", "0–40 s\nnew source"), ("tracking", "40–100 s\nquiet"),
                ("event", "100–150 s\nthe event"), ("settled", "150–200 s\nafter it"))
SWEEP_ROWS = (("Static", STATIC_LEVEL), ("Dynamic, 1 m/s", SWEEP_LEVEL))
FLOOR_MARGIN = 0.12  # room below the floor, as a share of the span between the floor and the top
BETA_OFFSET = 0.08  # horizontal shift between the schedulers at the same beta
RATE_TICKS = (0.2, 0.3, 0.5, 1.0, 2.0)


def _sweep_metrics(data: PaperData, level: float) -> pd.DataFrame:
    if np.isclose(level, STATIC_LEVEL):
        return data.static_metrics
    return data.region_metrics[np.isclose(data.region_metrics.diffusion, level)]


# --- static network -----------------------------------------------------------------------------------------


def static_dynamics(series: pd.DataFrame, out: Path) -> None:
    """Round rate (top) and mean error (bottom) through the static evaluation episodes, beta = 0.95."""
    data = series[np.isclose(series.beta, FOCUS_BETA)]
    fig, (rate_ax, error_ax) = style.figure("single", height=3.3, nrows=2, sharex=True)
    times, unreachable = _mean_series(data, TOPOLOGY_CARRIER, "unreachable")
    error_ax.fill_between(times, 0, unreachable * ERROR_SCALE_M, color=FLOOR_COLOUR, linewidth=0, zorder=1)
    for algorithm in BRACKET:
        period = FIXED_PERIODS[algorithm]
        rate_ax.axhline(1 / period, color=_fixed_colour(period), linestyle="--", linewidth=0.9, zorder=2)
        style.direct_label(rate_ax, config.EPISODE_LENGTH_S, 1 / period, f"{period:g} s", dx=-1, dy=3, va="bottom",
                           ha="right")
        error_ax.plot(*_mean_series(data, algorithm, "absoluteError"), color=_fixed_colour(period), linestyle="--",
                      linewidth=0.9, zorder=2)
    for algorithm in LEARNERS:
        line = style.line_style(METHODS[algorithm], linewidth=1.0, markevery=10)
        rate_ax.plot(*_rate_series(data, algorithm), **line)
        error_ax.plot(*_mean_series(data, algorithm, "absoluteError"), **line)
    for ax in (rate_ax, error_ax):
        _source_changes(ax, label=ax is rate_ax)
    low, high = rate_ax.get_ylim()
    rate_ax.set_ylim(low, high + 0.2 * (high - low))
    rate_ax.set_ylabel("Rounds per node per second\n(4 s moving average)")
    error_ax.set_ylabel("Mean absolute error (m)")
    error_ax.set_xlabel("Time within the evaluation episode (s)")
    error_ax.set_ylim(bottom=0)
    error_ax.set_xlim(0, config.EPISODE_LENGTH_S)
    handles = _learner_handles(heuristic=False, labels=LABELS) + [
        Line2D([], [], color=_fixed_colour(FIXED_PERIODS[algorithm]), linestyle="--", linewidth=0.9,
               label=f"Fixed {FIXED_PERIODS[algorithm]:g} s") for algorithm in BRACKET] + [_floor_handle()]
    style.figure_legend(fig, handles, ncol=2)
    style.save(fig, out, "rate_and_error")


# --- gains --------------------------------------------------------------------------------------------------


def _gain_cells(table: pd.DataFrame, algorithm: int, estimate: str) -> pd.DataFrame:
    rows = table[(table.algorithm == algorithm) & (table.estimate == estimate)]
    keys = [(variant, level, phase) for variant, level, _ in GAIN_ROWS for phase, _ in GAIN_COLUMNS]
    return rows.set_index(["variant", "level", "phase"]).reindex(keys)[["mean", "low", "high"]]


def gains(table: pd.DataFrame, out: Path) -> None:
    """Gain over the fixed periods at beta = 0.95, per scenario (rows) and phase (columns)."""
    fig, axes = style.figure("double", height=3.0, nrows=2, ncols=2, sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-GAIN_LIMIT, vcenter=0.0, vmax=GAIN_LIMIT)
    for row, (estimate, name) in enumerate((("whole", "whole error"), (CONNECTED, "nodes connected to the source"))):
        for column, algorithm in enumerate(LEARNERS):
            ax = axes[row, column]
            _draw_cells(ax, _gain_cells(table, algorithm, estimate), (len(GAIN_ROWS), len(GAIN_COLUMNS)), norm)
            ax.set_title(f"({'abcd'[row * 2 + column]}) {LABELS[algorithm]}, {name}", loc="left", fontsize=7, pad=3)
    axes[0, 0].set_yticks(range(len(GAIN_ROWS)), [label for *_, label in GAIN_ROWS])
    for ax in axes[:, 0]:
        ax.tick_params(axis="y", labelsize=6.5)
    for ax in axes[1]:
        ax.set_xticks(range(len(GAIN_COLUMNS)), [label for _, label in GAIN_COLUMNS], fontsize=6)
    _gain_colourbar(fig, axes, norm)
    style.save(fig, out, "gains")


# --- multi-objective ------------------------------------------------------------------------------------------


def _zoom_to_floor(ax, rows: pd.DataFrame, phase: str) -> None:
    """Starts the error axis just below the floor of the nodes cut off from the source, where every schedule ends."""
    floor = _metres(rows[rows.algorithm == TOPOLOGY_CARRIER], "unreachable", phase).mean()
    top = ax.get_ylim()[1]
    ax.set_ylim(floor - FLOOR_MARGIN * (top - floor), top)


def pareto_frontiers(data: PaperData, out: Path) -> None:
    """Mean error vs. rounds of every beta (lines join the betas of a scheduler) against the fixed periods."""
    fig, axes = style.figure(height=4.0, nrows=len(SWEEP_ROWS), ncols=len(FRONTIER_PHASES))
    note = axes[0, -1]
    for row, (name, level) in enumerate(SWEEP_ROWS):
        metrics = _sweep_metrics(data, level)
        for column, phase in enumerate(FRONTIER_PHASES):
            ax = axes[row, column]
            rows = metrics[metrics.phase == phase]
            _frontier_panel(ax, rows, phase)
            _zoom_to_floor(ax, rows, phase)
            window = PHASE_BY_KEY[phase]
            ax.set_title(f"({'abcdef'[row * len(FRONTIER_PHASES) + column]}) {name}, {window.start:g}–{window.end:g} s",
                         loc="left", fontsize=7, pad=2)
            ax.set_xlabel(f"Rounds in {window.start:g}–{window.end:g} s (log scale)")
        axes[row, 0].set_ylabel("Mean absolute error (m)")
    note.text(0.99, 0.98, "each line joins β = 0.5, 0.8, 0.95, 0.99, 1\n(left to right: more rounds)",
              transform=note.transAxes, ha="right", va="top", fontsize=6, color=style.INK_MUTED)
    handles = _learner_handles(labels=LABELS) + [
        Line2D([], [], **style.fixed_period_style(label="Fixed period (0.5–5 s)")), _floor_handle()]
    style.figure_legend(fig, handles, ncol=5)
    style.save(fig, out, "pareto_frontiers")


def _beta_panel(ax, metrics: pd.DataFrame, phase: str, quantity: str) -> None:
    values = beta_response(metrics, phase, quantity)
    fixed = values[values.algorithm.isin(BRACKET)].groupby("algorithm").value.mean()
    for algorithm, mean in fixed.items():
        period = FIXED_PERIODS[algorithm]
        ax.axhline(mean, color=_fixed_colour(period), linestyle="--", linewidth=0.8, zorder=1)
        style.direct_label(ax, len(config.WEIGHTS) - 0.55, mean, f"{period:g} s", dx=0, dy=2, va="bottom", ha="right")
    for index, algorithm in enumerate(SCHEDULERS):
        rows = values[values.algorithm == algorithm]
        intervals = [t_interval(rows[np.isclose(rows.beta, beta)].value.to_numpy()) for beta in config.WEIGHTS]
        x = np.arange(len(config.WEIGHTS)) + (index - 1) * BETA_OFFSET
        means = np.array([interval.mean for interval in intervals])
        errors = [means - [interval.low for interval in intervals], [interval.high for interval in intervals] - means]
        ax.errorbar(x, means, yerr=errors, fmt="none", ecolor=METHODS[algorithm].color, elinewidth=0.7, capsize=1.2)
        ax.plot(x, means, **style.line_style(METHODS[algorithm], linewidth=0.9))
    ax.set_xticks(range(len(config.WEIGHTS)), [f"{beta:g}" for beta in config.WEIGHTS])
    ax.set_xlim(-0.4, len(config.WEIGHTS) - 0.4)


def tradeoff_vs_beta(data: PaperData, out: Path) -> None:
    """Rows: static, dynamic at 1 m/s; columns: rounds after 40 s, then the error of each phase; x: beta."""
    reference = f"{FIXED_PERIODS[REFERENCE_FIXED]:g} s"
    panels = (("post_bootstrap", "rate", "Rounds per node\nper second, 40–200 s"),
              *((phase, "error", f"Error over a fixed {reference}\nperiod, {PHASE_BY_KEY[phase].start:g}–"
                 f"{PHASE_BY_KEY[phase].end:g} s") for phase in FRONTIER_PHASES))
    fig, axes = style.figure(height=3.9, nrows=len(SWEEP_ROWS), ncols=len(panels))
    for row, (name, level) in enumerate(SWEEP_ROWS):
        metrics = _sweep_metrics(data, level)
        for column, (phase, quantity, label) in enumerate(panels):
            ax = axes[row, column]
            _beta_panel(ax, metrics, phase, quantity)
            ax.set_ylabel(label, fontsize=6.5)
            if quantity == "rate":
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(FixedLocator(RATE_TICKS))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_major_formatter(lambda value, _: f"{value:g}")
            else:
                ax.yaxis.set_major_formatter(lambda value, _: f"×{value:.2g}")
            ax.set_title(f"({'abcdefgh'[row * len(panels) + column]})" + (f" {name}" if column == 0 else ""),
                         loc="left", fontsize=7, pad=2)
        for ax in axes[row]:
            ax.set_xlabel("Trade-off weight β")
    handles = _learner_handles(labels=LABELS) + [
        Line2D([], [], color=_fixed_colour(FIXED_PERIODS[algorithm]), linestyle="--", linewidth=0.8,
               label=f"Fixed {FIXED_PERIODS[algorithm]:g} s") for algorithm in BRACKET]
    style.figure_legend(fig, handles, ncol=5)
    style.save(fig, out, "tradeoff_vs_beta")


def render(data: PaperData, out: Path) -> None:
    """The figures of the mobile experiment that the static paper has no counterpart for."""
    style.apply()
    static_dynamics(data.static_series, out / "static")
    region_dynamics(REGION, data.region_series, out / "dynamic", LABELS)
    region_routes(REGION, data.routes, out / "dynamic", LABELS)
    region_snapshot(REGION, data.frame, out / "dynamic")
    gains(data.gains, out / RECAP)
    pareto_frontiers(data, out / RECAP)
    tradeoff_vs_beta(data, out / RECAP)
    plt.close("all")
