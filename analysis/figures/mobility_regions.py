"""Where the rounds go when only a region moves (``analysis.mobility_regions``), one set per scenario.

* ``<prefix>_dynamics``: round rate and error of the moving region and of the still nodes through the evaluation
  episodes, learned policies against two fixed periods;
* ``<prefix>_allocation``: the rate of the moving region over the rate of the still nodes in the two windows of the
  scenario (1 for any fixed period);
* ``<prefix>_routes``: rate and error of the moving nodes and of the still nodes whose route to the source crosses a
  moving node or not;
* ``<prefix>_snapshot``: one frame on the map: the routes to the source, where the error is with a fixed period, where
  the GNN spends its rounds and what error it leaves.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.ticker import FixedLocator, NullLocator

from .. import config, style
from ..mobility_pareto import FIXED_PERIODS, METHODS
from ..mobility_static import BOOTSTRAP_END_S
from ..mobility_variants import HEURISTIC, LEARNERS
from ..mobility_regions import (COMMUNICATION_RANGE_M, ROUTES, SNAPSHOT_LEVEL, SNAPSHOT_SCHEDULERS, SNAPSHOT_WINDOW_S,
                                TOPOLOGY_CARRIER, RegionScenario)
from ..stats import t_band
from .mobility_final import BRACKET, SHADE, _fixed_colour

GROUPS = ((True, "moving region"), (False, "still nodes"))
SMOOTHING_FRAMES = 2  # 4 s moving average of the round rate
MARK_EVERY = 10
RATE_HEADROOM = 0.25  # room above the rates for the event labels
OFFSET = 0.2
ROUTE_OFFSET = 0.14
ROUTE_LABELS = {"moving": "moving", "through": "still, route via\nmoving nodes", "clear": "still, route via\nstill nodes"}
FIXED_MARKER = "s"
RATE_TICKS = (0.2, 0.3, 0.5, 1.0, 1.5)
# Map: route classes in an ordinal grey ramp (validated light to dark), the moving nodes as triangles.
ROUTE_COLOURS = {"moving": "#1a1a1a", "through": "#6f6e69", "clear": "#a9a8a1"}
MAP_LABELS = {"moving": "moving", "through": "still, route via moving nodes", "clear": "still, route via still nodes",
              "cut off": "cut off from the source"}
MOVING_MARKER, STILL_MARKER, CUT_MARKER = "^", "o", "x"
MAP_MARKER_SIZE = 13
ERROR_MAP = mpl.colors.LinearSegmentedColormap.from_list("error", mpl.colormaps["Reds"](np.linspace(0.12, 1.0, 256)))
RATE_MAP = mpl.colors.LinearSegmentedColormap.from_list("rate", mpl.colormaps["Blues"](np.linspace(0.15, 1.0, 256)))
ERROR_GAMMA = 0.5  # square-root colour scale: the error spans 0 to hundreds of metres
ERROR_CEILING_QUANTILE = 0.95  # the colour scale saturates above it (a node that just reconnected holds ~500 m)  # log axis: the heuristic spends several times the rounds of the rest


def _mean(data: pd.DataFrame, algorithm: int, column: str) -> tuple[np.ndarray, np.ndarray]:
    rows = data[data.algorithm == algorithm].pivot_table(index="seed", columns="time", values=column)
    mean, _, _ = t_band(rows.to_numpy())
    if column == "rate":
        mean = pd.Series(mean).rolling(SMOOTHING_FRAMES, min_periods=1).mean().to_numpy()
    return rows.columns.to_numpy(), mean


def _events(ax, label: bool, event: str, motion_end: float | None = None) -> None:
    ax.axvspan(0, BOOTSTRAP_END_S, color=SHADE, linewidth=0, zorder=0)
    ax.axvline(config.SOURCE_SWITCH_S, color=style.INK_MUTED, linewidth=0.7, linestyle=":", zorder=1)
    if motion_end is not None:
        ax.axvline(motion_end, color=style.INK_MUTED, linewidth=0.7, linestyle=":", zorder=1)
        if label:
            ax.text(motion_end, 0.97, " stops", transform=ax.get_xaxis_transform(), fontsize=6,
                    color=style.INK_MUTED, ha="left", va="top")
    if label:
        ax.text(BOOTSTRAP_END_S / 2, 0.97, "new source", transform=ax.get_xaxis_transform(), fontsize=6,
                color=style.INK_MUTED, ha="center", va="top")
        ax.text(config.SOURCE_SWITCH_S, 0.97, " " + event.replace("\n", "\n "), transform=ax.get_xaxis_transform(),
                fontsize=6, color=style.INK_MUTED, ha="left", va="top")


def _handle(algorithm: int, labels: Mapping[int, str] | None = None, **overrides) -> Line2D:
    """Legend handle of a scheduler; ``labels`` replaces the method labels (e.g. without the training)."""
    if labels and algorithm in labels:
        overrides["label"] = labels[algorithm]
    return style.legend_handle(METHODS[algorithm], **overrides)


def region_dynamics(scenario: RegionScenario, series: pd.DataFrame, out: Path,
                    labels: Mapping[int, str] | None = None) -> None:
    """Rows: round rate, error; columns: the moving region and the still nodes at each mobile level."""
    levels = [level for level in scenario.levels if level > 0]
    panels = [(level, moving, name) for level in levels for moving, name in GROUPS]
    fig, axes = style.figure(height=3.6, nrows=2, ncols=len(panels), sharex=True, sharey="row")
    for column, (level, moving, name) in enumerate(panels):
        data = series[np.isclose(series.level, level) & (series.moving == moving)]
        rate_ax, error_ax = axes[:, column]
        for algorithm in BRACKET:
            line = dict(color=_fixed_colour(FIXED_PERIODS[algorithm]), linestyle="--", linewidth=0.9, zorder=2)
            rate_ax.axhline(1 / FIXED_PERIODS[algorithm], **line)
            style.direct_label(rate_ax, config.EPISODE_LENGTH_S, 1 / FIXED_PERIODS[algorithm],
                               f"{FIXED_PERIODS[algorithm]:g} s", dx=-1, dy=3, va="bottom", ha="right")
            error_ax.plot(*_mean(data, algorithm, "error"), **line)
        for algorithm in LEARNERS:
            kwargs = style.line_style(METHODS[algorithm], linewidth=1.0, markevery=MARK_EVERY)
            rate_ax.plot(*_mean(data, algorithm, "rate"), **kwargs)
            error_ax.plot(*_mean(data, algorithm, "error"), **kwargs)
        for ax in (rate_ax, error_ax):
            _events(ax, label=ax is rate_ax and column == 0, event=scenario.event, motion_end=scenario.motion_end_s)
        rate_ax.set_title(f"({'abcdefgh'[column]}) {level:g} m/s, {name}", loc="left", fontsize=7, pad=2)
        error_ax.set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Rounds per node per second\n(4 s moving average)")
    axes[1, 0].set_ylabel("Mean absolute error (m)")
    axes[1, 0].set_xlim(0, config.EPISODE_LENGTH_S)
    axes[1, 0].set_ylim(bottom=0)
    low, high = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(low, high + RATE_HEADROOM * (high - low))
    handles = [_handle(algorithm, labels) for algorithm in LEARNERS] + [
        Line2D([], [], color=_fixed_colour(FIXED_PERIODS[algorithm]), linestyle="--", linewidth=0.9,
               label=f"Fixed {FIXED_PERIODS[algorithm]:g} s") for algorithm in BRACKET]
    style.figure_legend(fig, handles, ncol=4)
    style.save(fig, out, f"{scenario.prefix}_dynamics")


def region_allocation(scenario: RegionScenario, ratios: pd.DataFrame, out: Path,
                      labels: Mapping[int, str] | None = None) -> None:
    """Rate of the moving region over the still nodes, in the main window (filled) and in the other one (open)."""
    levels = scenario.levels
    main, other = scenario.windows
    fig, ax = style.figure("single", height=2.2)
    for offset, algorithm in zip((-OFFSET, 0.0, OFFSET), (*LEARNERS, HEURISTIC)):
        method = METHODS[algorithm]
        for window, face in ((main, method.color), (other, "white")):
            rows = ratios[(ratios.algorithm == algorithm) & (ratios.window == window.key)].set_index("level")
            rows = rows.reindex(levels)
            x = np.arange(len(levels)) + offset
            means = rows["mean"].to_numpy(dtype=float)
            ax.errorbar(x, means, yerr=[means - rows["low"].to_numpy(dtype=float),
                                        rows["high"].to_numpy(dtype=float) - means],
                        fmt="none", ecolor=method.color, elinewidth=0.7, capsize=1.2)
            ax.plot(x, means, linestyle="none", **style.line_style(method, markerfacecolor=face,
                                                                   markeredgecolor=method.color))
    ax.axhline(1.0, color=style.INK_MUTED, linewidth=0.7, linestyle="--", zorder=1)
    ax.text(0.5, 1.0, "any fixed period", fontsize=6, color=style.INK_MUTED, ha="center", va="bottom")
    ax.set_xticks(range(len(levels)), [scenario.static if level == 0 else f"{level:g} m/s" for level in levels])
    ax.set_ylabel("Round rate, moving region /\nstill nodes")
    handles = [_handle(algorithm, labels, linestyle="none") for algorithm in (*LEARNERS, HEURISTIC)]
    handles += [Line2D([], [], marker="o", linestyle="none", color=style.INK_SECONDARY, label=main.label),
                Line2D([], [], marker="o", linestyle="none", color=style.INK_SECONDARY, markerfacecolor="white",
                       label=other.label)]
    style.figure_legend(fig, handles, ncol=3)
    style.save(fig, out, f"{scenario.prefix}_allocation")


def _route_points(ax, rows: pd.DataFrame, routes: list, offset: float, **kwargs) -> None:
    rows = rows.set_index("route").reindex(routes)
    x = np.arange(len(routes)) + offset
    means = rows["mean"].to_numpy(dtype=float)
    ax.errorbar(x, means, yerr=[means - rows["low"].to_numpy(dtype=float), rows["high"].to_numpy(dtype=float) - means],
                fmt="none", ecolor=kwargs["color"], elinewidth=0.7, capsize=1.2)
    ax.plot(x, means, linestyle="none", **kwargs)


def region_routes(scenario: RegionScenario, summary: pd.DataFrame, out: Path,
                  labels: Mapping[int, str] | None = None) -> None:
    """Rows: error, round rate; columns: mobile levels; x: the moving nodes and the still ones by route (main window)."""
    window = scenario.windows[0]
    levels = [level for level in scenario.levels if level > 0]
    routes = [route for route in ROUTES if route in ROUTE_LABELS]
    schedulers = [(algorithm, dict(color=_fixed_colour(FIXED_PERIODS[algorithm]), marker=FIXED_MARKER, markersize=3.5))
                  for algorithm in BRACKET]
    schedulers += [(algorithm, style.line_style(METHODS[algorithm])) for algorithm in (*LEARNERS, HEURISTIC)]
    fig, axes = style.figure(height=3.4, nrows=2, ncols=len(levels), sharex=True, sharey="row")
    rows = summary[summary.window == window.key]
    for column, level in enumerate(levels):
        at_level = rows[np.isclose(rows.level, level)]
        for row, quantity in enumerate(("error", "rate")):
            ax = axes[row, column]
            for index, (algorithm, kwargs) in enumerate(schedulers):
                points = at_level[(at_level.algorithm == algorithm) & (at_level.quantity == quantity)]
                _route_points(ax, points, routes, (index - (len(schedulers) - 1) / 2) * ROUTE_OFFSET, **kwargs)
            ax.set_xlim(-0.5, len(routes) - 0.5)
        shares = at_level[(at_level.algorithm == TOPOLOGY_CARRIER) & (at_level.quantity == "share")].set_index("route")
        ticks = [f"{ROUTE_LABELS[route]}\n({shares['mean'].get(route, np.nan):.0%} of node-time)" for route in routes]
        axes[1, column].set_xticks(range(len(routes)), ticks, fontsize=6)
        axes[0, column].set_title(f"({'ab'[column]}) {level:g} m/s, {window.label}", loc="left", fontsize=7, pad=2)
    axes[0, 0].set_ylabel("Mean absolute error (m)")
    axes[1, 0].set_ylabel("Rounds per node per second")
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_yscale("log")
    axes[1, 0].yaxis.set_major_locator(FixedLocator(RATE_TICKS))
    axes[1, 0].yaxis.set_minor_locator(NullLocator())
    axes[1, 0].yaxis.set_major_formatter(lambda value, _: f"{value:g}")
    handles = [Line2D([], [], linestyle="none", label=f"Fixed {FIXED_PERIODS[algorithm]:g} s", **kwargs)
               for algorithm, kwargs in schedulers[:len(BRACKET)]]
    handles += [_handle(algorithm, labels, linestyle="none") for algorithm in (*LEARNERS, HEURISTIC)]
    style.figure_legend(fig, handles, ncol=5)
    style.save(fig, out, f"{scenario.prefix}_routes")


def _map_axes(ax, frame: pd.DataFrame, title: str) -> None:
    """Shortest-path tree to the source, the source, the cut-off nodes; square and without ticks."""
    for row in frame.itertuples():
        ax.plot([row.x, row.parent_x], [row.y, row.parent_y], color=style.GRID, linewidth=0.6, zorder=1)
    cut = frame[frame.route == "cut off"]
    ax.scatter(cut.x, cut.y, marker=CUT_MARKER, s=MAP_MARKER_SIZE, color=style.INK_MUTED, linewidths=0.7, zorder=3)
    source = frame[frame.source]
    ax.scatter(source.x, source.y, marker="*", s=70, color="white", edgecolors=style.INK, linewidths=0.8, zorder=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=7, pad=2)


def _scatter_values(ax, frame: pd.DataFrame, column: str, cmap, norm):
    connected = frame[frame.route != "cut off"]
    image = None
    for moving, marker in ((True, MOVING_MARKER), (False, STILL_MARKER)):
        nodes = connected[connected.region == moving]
        image = ax.scatter(nodes.x, nodes.y, c=nodes[column], cmap=cmap, norm=norm, marker=marker, s=MAP_MARKER_SIZE,
                           edgecolors=style.INK_SECONDARY, linewidths=0.3, zorder=2)
    return image


def region_snapshot(scenario: RegionScenario, frame: pd.DataFrame, out: Path) -> None:
    """(a) routes to the source; (b) error with a fixed 3 s period; (c) GNN round rate; (d) GNN error."""
    fixed, learner = SNAPSHOT_SCHEDULERS
    start, end = scenario.snapshot_s - SNAPSHOT_WINDOW_S, scenario.snapshot_s
    fig, axes = style.figure(height=2.55, ncols=4, layout="constrained")
    _map_axes(axes[0], frame, f"(a) Routes to the source at {end:g} s")
    for route, colour in ROUTE_COLOURS.items():
        nodes = frame[frame.route == route]
        axes[0].scatter(nodes.x, nodes.y, color=colour, marker=MOVING_MARKER if route == "moving" else STILL_MARKER,
                        s=MAP_MARKER_SIZE, linewidths=0, zorder=2)
    connected = frame[frame.route != "cut off"]
    errors = connected[[f"error_{fixed}", f"error_{learner}"]].to_numpy().ravel()
    error_norm = PowerNorm(ERROR_GAMMA, vmin=0.0, vmax=np.quantile(errors, ERROR_CEILING_QUANTILE))
    rate_norm = Normalize(vmin=0.0, vmax=connected[f"rate_{learner}"].max())
    panels = ((axes[1], f"error_{fixed}", ERROR_MAP, error_norm, f"(b) Error, fixed {FIXED_PERIODS[fixed]:g} s"),
              (axes[2], f"rate_{learner}", RATE_MAP, rate_norm, f"(c) Rounds, {METHODS[learner].label.split()[0]}"),
              (axes[3], f"error_{learner}", ERROR_MAP, error_norm, f"(d) Error, {METHODS[learner].label.split()[0]}"))
    for ax, column, cmap, norm, title in panels:
        _map_axes(ax, frame, title)
        image = _scatter_values(ax, frame, column, cmap, norm)
        bar = fig.colorbar(image, ax=ax, orientation="horizontal", shrink=0.85, pad=0.02, aspect=22,
                           extend="max" if norm is error_norm else "neither")
        bar.ax.tick_params(labelsize=5.5)
        bar.set_label("rounds per second" if column.startswith("rate") else "mean absolute error (m)", fontsize=6)
    handles = [Line2D([], [], marker=MOVING_MARKER if route == "moving" else STILL_MARKER, linestyle="none",
                      color=colour, label=MAP_LABELS[route]) for route, colour in ROUTE_COLOURS.items()]
    handles += [Line2D([], [], marker=CUT_MARKER, linestyle="none", color=style.INK_MUTED, label=MAP_LABELS["cut off"]),
                Line2D([], [], marker="*", linestyle="none", markerfacecolor="white", markeredgecolor=style.INK,
                       markersize=7, label="source"),
                Line2D([], [], color=style.GRID, linewidth=0.8,
                       label=f"shortest route (links ≤ {COMMUNICATION_RANGE_M:g} m)")]
    style.figure_legend(fig, handles, ncol=3)
    fig.supxlabel(f"Seed 1, {SNAPSHOT_LEVEL:g} m/s, first evaluation episode; rate and error averaged over "
                  f"{start:g}–{end:g} s; triangles: moving nodes", fontsize=6)
    style.save(fig, out, f"{scenario.prefix}_snapshot")


def render(scenario: RegionScenario, series: pd.DataFrame, ratios: pd.DataFrame, routes: pd.DataFrame,
           snapshot: pd.DataFrame, out: Path) -> None:
    style.apply()
    region_dynamics(scenario, series, out)
    region_allocation(scenario, ratios, out)
    region_routes(scenario, routes, out)
    region_snapshot(scenario, snapshot, out)
    plt.close("all")
