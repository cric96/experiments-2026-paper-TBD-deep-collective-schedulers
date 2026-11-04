"""The final configuration of the mobile experiment (``analysis.mobility_variants``) and the steps that led to it.

The final configuration is the ``dependants`` variant: waypoint mobility, warm start and the three local features.
With the warm start every episode starts with a new source (the gradient of the previous one is still there), so
0-40 s is a source change like the one at 100 s, not a bootstrap from scratch.

* ``final_dynamics``: round rate and error through the evaluation episodes, over the error of the nodes cut off from
  the source, which no schedule can correct;
* ``final_frontiers``: error vs. rounds per phase and speed, down to that floor;
* ``error_floor``: how the scenario changes moved the floor and the room left to the schedule;
* ``feature_gains``: the gain of each feature step, per seed, on the whole error and on the connected nodes.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .. import config, style
from ..mobility_pareto import FIXED_PERIODS, METHODS
from ..mobility_static import BOOTSTRAP_END_S, PHASE_BY_KEY, fixed_reference
from ..mobility_variants import BROWNIAN, CONNECTED, FOCUS_BETA, HEURISTIC, LEARNERS, VARIANT_BY_KEY
from ..stats import t_band, t_interval
from .mobility_static import _log_rounds_axis, _mean_series, _rate_series

FINAL = "dependants"
FEATURE_STEPS = ("warm-start", "own-speed", "unexplained-change", "dependants")
SCENARIO_STEPS = ("baseline", "short-retention", "waypoint-mobility", "warm-start")
ERROR_SCALE_M = 500.0  # normalization of the error: a node cut off from the source counts this much
FASTEST_FIXED, SLOWEST_FIXED = 10, 7  # 0.5 s and 5 s
BRACKET = (4, 5)  # fixed 2 s and 3 s: the round rates of the learned policies lie between them
TOPOLOGY_CARRIER = FASTEST_FIXED  # any scheduler: the motion, hence the reachability, does not depend on it
FLOOR_COLOUR = "#d9d7d0"
RANGE_COLOUR = "#c9d7e8"
SHADE = "#f2f2ef"
FRONTIER_PHASES = ("tracking", "switch")
FEATURE_PANELS = (
    ("gain_tracking", "Tracking, 1 m/s"),
    ("gain_switch", "After the switch, 1 m/s"),
    ("gain_post_static", "40–200 s, static"),
    ("gain_post_focus", "40–200 s, 1 m/s"),
    ("gain_post_fastest", "40–200 s, 2 m/s"),
)
OFFSET = 0.17  # horizontal shift between the two learners at the same step
JITTER = 0.05  # horizontal spread of the per-seed dots


def _level_name(variant_key: str, index: int) -> str:
    variant = VARIANT_BY_KEY[variant_key]
    if index == 0:
        return "static"
    level = variant.levels[index]
    return f"D = {level:g} m²/s" if variant.levels == BROWNIAN else f"{level:g} m/s"


def _fixed_colour(period: float) -> str:
    return style.fixed_period_color(min(max(period, 1.0), 5.0))


def _learner_handles(heuristic: bool = True, labels: Mapping[int, str] | None = None) -> list:
    algorithms = (*LEARNERS, HEURISTIC) if heuristic else LEARNERS
    return [style.legend_handle(METHODS[algorithm], **({"label": labels[algorithm]} if labels else {}))
            for algorithm in algorithms]


def _floor_handle() -> Patch:
    return Patch(facecolor=FLOOR_COLOUR, edgecolor="none", label="Nodes cut off from the source (any schedule)")


# --- dynamics -------------------------------------------------------------------------------------------


def _source_changes(ax, label: bool) -> None:
    ax.axvspan(0, BOOTSTRAP_END_S, color=SHADE, linewidth=0, zorder=0)
    style.mark_source_switch(ax, label=label, label_y=0.97)
    if label:
        ax.text(BOOTSTRAP_END_S / 2, 0.97, "new source", transform=ax.get_xaxis_transform(), fontsize=6,
                color=style.INK_MUTED, ha="center", va="top")


def final_dynamics(series: pd.DataFrame, out: Path) -> None:
    """Round rate (top) and mean error (bottom) through the evaluation episodes, beta = 0.95, per speed."""
    levels = VARIANT_BY_KEY[FINAL].levels
    fig, axes = style.figure(height=3.6, nrows=2, ncols=len(levels), sharex=True, sharey="row")
    focus = series[np.isclose(series.beta, FOCUS_BETA)]
    for column, level in enumerate(levels):
        data = focus[np.isclose(focus.diffusion, level)]
        rate_ax, error_ax = axes[:, column]
        times, unreachable = _mean_series(data, TOPOLOGY_CARRIER, "unreachable")
        error_ax.fill_between(times, 0, unreachable * ERROR_SCALE_M, color=FLOOR_COLOUR, linewidth=0, zorder=1)
        for algorithm in BRACKET:
            period = FIXED_PERIODS[algorithm]
            rate_ax.axhline(1 / period, color=_fixed_colour(period), linestyle="--", linewidth=0.9, zorder=2)
            style.direct_label(rate_ax, config.EPISODE_LENGTH_S, 1 / period, f"{period:g} s", dx=-1, dy=3,
                               va="bottom", ha="right")
            error_ax.plot(*_mean_series(data, algorithm, "absoluteError"), color=_fixed_colour(period),
                          linestyle="--", linewidth=0.9, zorder=2)
        for algorithm in LEARNERS:
            line = style.line_style(METHODS[algorithm], linewidth=1.0, markevery=10)
            rate_ax.plot(*_rate_series(data, algorithm), **line)
            error_ax.plot(*_mean_series(data, algorithm, "absoluteError"), **line)
        for ax in (rate_ax, error_ax):
            _source_changes(ax, label=ax is rate_ax and column == 0)
        rate_ax.set_title(f"({'abc'[column]}) {_level_name(FINAL, column).capitalize()}", loc="left", fontsize=8)
        error_ax.set_xlabel("Time within the evaluation episode (s)")
    axes[0, 0].set_ylabel("Rounds per node per second\n(4 s moving average)")
    axes[1, 0].set_ylabel("Mean absolute error (m)")
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].set_xlim(0, config.EPISODE_LENGTH_S)
    handles = _learner_handles(heuristic=False) + [
        Line2D([], [], color=_fixed_colour(FIXED_PERIODS[algorithm]), linestyle="--", linewidth=0.9,
               label=f"Fixed {FIXED_PERIODS[algorithm]:g} s") for algorithm in BRACKET] + [_floor_handle()]
    style.figure_legend(fig, handles, ncol=5)
    style.save(fig, out, "final_dynamics")


# --- frontiers ------------------------------------------------------------------------------------------


def _metres(data: pd.DataFrame, column: str, phase: str) -> pd.Series:
    """A per-node phase integral of the normalized error as a mean error in metres (capped at the scale)."""
    return data[column] / PHASE_BY_KEY[phase].duration * ERROR_SCALE_M


def _frontier_panel(ax, data: pd.DataFrame, phase: str) -> None:
    floor = _metres(data[data.algorithm == TOPOLOGY_CARRIER], "unreachable", phase).mean()
    ax.axhspan(0, floor, color=FLOOR_COLOUR, linewidth=0, zorder=0)
    fixed = fixed_reference(data).assign(metres=lambda rows: _metres(rows, "error", phase))
    fixed = fixed.groupby("period")[["rounds", "metres"]].mean().sort_values("rounds")
    ax.plot(fixed.rounds, fixed.metres, **style.fixed_period_style(label="Fixed period (0.5–5 s)"))
    for period, point in fixed.iterrows():
        style.direct_label(ax, point.rounds, point.metres, f"{period:g} s", dx=2, dy=2, va="bottom", ha="left")
    for algorithm in (*LEARNERS, HEURISTIC):
        rows = data[data.algorithm == algorithm].assign(metres=lambda rows: _metres(rows, "error", phase))
        means = rows.groupby("beta")[["rounds", "metres"]].mean()
        ax.plot(means.rounds, means.metres, **style.line_style(METHODS[algorithm], linewidth=0.9))
    _log_rounds_axis(ax)
    ax.set_ylim(0, fixed.metres.max() * 1.12)


def final_frontiers(metrics: pd.DataFrame, out: Path) -> None:
    """Mean error vs. rounds in the tracking and switch phases, per speed; the grey band is the floor."""
    levels = VARIANT_BY_KEY[FINAL].levels
    fig, axes = style.figure(height=4.2, nrows=len(FRONTIER_PHASES), ncols=len(levels))
    for row, phase in enumerate(FRONTIER_PHASES):
        for column, level in enumerate(levels):
            ax = axes[row, column]
            _frontier_panel(ax, metrics[np.isclose(metrics.diffusion, level) & (metrics.phase == phase)], phase)
            title = f"({'abcdef'[row * len(levels) + column]}) {_level_name(FINAL, column).capitalize()}"
            ax.set_title(title, loc="left", fontsize=7.5, pad=2)
            ax.set_xlabel(f"Rounds in {PHASE_BY_KEY[phase].start:g}–{PHASE_BY_KEY[phase].end:g} s (log scale)")
        axes[row, 0].set_ylabel(f"Mean error (m)\n{PHASE_BY_KEY[phase].label.lower()}")
    axes[0, 1].text(0.99, 0.98, "learners: $\\beta$ = 0.5, 0.95, 1\n(swept at 1 m/s only)",
                    transform=axes[0, 1].transAxes, ha="right", va="top", fontsize=6, color=style.INK_MUTED)
    handles = _learner_handles() + [Line2D([], [], **style.fixed_period_style(label="Fixed period (0.5–5 s)")),
                                    _floor_handle()]
    style.figure_legend(fig, handles, ncol=5)
    style.save(fig, out, "final_frontiers")


# --- error floor and room left to the schedule, per scenario ---------------------------------------------


def _floor_row(key: str, metrics: pd.DataFrame, level_index: int, phase: str) -> dict:
    level = VARIANT_BY_KEY[key].levels[level_index]
    data = metrics[np.isclose(metrics.diffusion, level) & (metrics.phase == phase)]

    def metres(algorithm: int, column: str = "error") -> float:
        return _metres(data[data.algorithm == algorithm], column, phase).mean()

    return dict(scenario=key, level=level_index, floor=metres(TOPOLOGY_CARRIER, "unreachable"),
                fastest=metres(FASTEST_FIXED), slowest=metres(SLOWEST_FIXED),
                learner=metres(LEARNERS[0]) if (data.algorithm == LEARNERS[0]).any() else np.nan)


def _floor_panel(ax, rows: pd.DataFrame) -> None:
    gap = 0.6
    x = rows.index.to_numpy() + gap * (rows.index // 3)
    ax.bar(x, rows.floor, width=0.62, color=FLOOR_COLOUR, linewidth=0, zorder=1)
    ax.bar(x, rows.slowest - rows.fastest, bottom=rows.fastest, width=0.62, color=RANGE_COLOUR, linewidth=0, zorder=1)
    for column in ("fastest", "slowest"):
        ax.plot(x, rows[column], **style.fixed_period_style(linestyle="none", marker="_", markersize=7))
    ax.plot(x, rows.learner, **style.line_style(METHODS[LEARNERS[0]], linestyle="none", markersize=4))
    ax.set_xticks(x, [_level_name(row.scenario, row.level) for row in rows.itertuples()], rotation=45, ha="right",
                  rotation_mode="anchor", fontsize=6)
    for index, key in enumerate(SCENARIO_STEPS):
        ax.text(x[index * 3 + 1], 1.0, VARIANT_BY_KEY[key].label, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=6.5, color=style.INK_SECONDARY)
    ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
    ax.set_ylim(bottom=0)


def error_floor(loaded: dict, out: Path) -> None:
    """Per scenario and level: the floor, the range between the 0.5 s and the 5 s period, and the GNN."""
    fig, axes = style.figure(height=2.9, nrows=1, ncols=len(FRONTIER_PHASES))
    for index, (ax, phase) in enumerate(zip(axes, FRONTIER_PHASES)):
        rows = pd.DataFrame([_floor_row(key, loaded[key][0], level, phase) for key in SCENARIO_STEPS
                             for level in range(3)])
        _floor_panel(ax, rows)
        ax.set_title(f"({'ab'[index]}) {PHASE_BY_KEY[phase].label}", loc="left", fontsize=7.5, pad=12)
    axes[0].set_ylabel("Mean error (m)")
    handles = [_floor_handle(), Patch(facecolor=RANGE_COLOUR, edgecolor="none", label="Fixed 0.5 s … 5 s"),
               style.legend_handle(METHODS[LEARNERS[0]], linestyle="none", label=f"{METHODS[LEARNERS[0]].label}, "
                                   r"$\beta$ = 0.95")]
    style.figure_legend(fig, handles, ncol=3)
    style.save(fig, out, "error_floor")


# --- what each feature buys ------------------------------------------------------------------------------


def _feature_panel(ax, seeds: pd.DataFrame, quantity: str) -> None:
    for offset, algorithm in zip((-OFFSET, OFFSET), LEARNERS):
        colour = METHODS[algorithm].color
        means, lows, highs = [], [], []
        for step, key in enumerate(FEATURE_STEPS):
            gains = seeds[(seeds.variant == key) & (seeds.algorithm == algorithm) & (seeds.quantity == quantity)].gain
            jitter = np.linspace(-JITTER, JITTER, len(gains)) if len(gains) > 1 else np.zeros(len(gains))
            ax.scatter(step + offset + jitter, gains, s=4, color=colour, alpha=0.35, linewidths=0, zorder=2)
            interval = t_interval(gains)
            means.append(interval.mean), lows.append(interval.low), highs.append(interval.high)
        x = np.arange(len(FEATURE_STEPS)) + offset
        means = np.array(means)
        ax.errorbar(x, means, yerr=[means - np.array(lows), np.array(highs) - means], fmt="none", ecolor=colour,
                    elinewidth=0.9, capsize=1.5, zorder=3)
        ax.plot(x, means, **style.line_style(METHODS[algorithm], linewidth=0.7, zorder=4))
    ax.axhline(0, color=style.INK_MUTED, linewidth=0.7, zorder=1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks(range(len(FEATURE_STEPS)), [VARIANT_BY_KEY[key].label for key in FEATURE_STEPS], rotation=45,
                  ha="right", rotation_mode="anchor", fontsize=6)
    ax.set_xlim(-0.5, len(FEATURE_STEPS) - 0.5)


def feature_gains(seeds: pd.DataFrame, out: Path) -> None:
    """Gain over the fixed periods of each feature step (8 seeds, beta = 0.95): dots are seeds, bars 95% CIs."""
    fig, axes = style.figure(height=4.0, nrows=2, ncols=len(FEATURE_PANELS), sharex=True)
    for row, (prefix, estimate) in enumerate((("", "Whole error"), (CONNECTED, "Connected nodes"))):
        for column, (quantity, title) in enumerate(FEATURE_PANELS):
            ax = axes[row, column]
            _feature_panel(ax, seeds, prefix + quantity)
            ax.set_title(f"({'abcdefghij'[row * len(FEATURE_PANELS) + column]}) {title}", loc="left", fontsize=7,
                         pad=2)
        axes[row, 0].set_ylabel(f"{estimate}:\nerror reduction at equal rounds")
    style.figure_legend(fig, _learner_handles(heuristic=False), ncol=2)
    style.save(fig, out, "feature_gains")


def render(loaded: dict, seeds: pd.DataFrame, out: Path) -> None:
    style.apply()
    if FINAL in loaded:
        metrics, series = loaded[FINAL]
        final_dynamics(series, out)
        final_frontiers(metrics, out)
    if all(key in loaded for key in SCENARIO_STEPS):
        error_floor(loaded, out)
    if all(key in loaded for key in FEATURE_STEPS):
        feature_gains(seeds, out)
    plt.close("all")
