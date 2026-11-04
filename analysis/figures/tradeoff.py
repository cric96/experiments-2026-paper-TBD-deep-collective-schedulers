"""Evaluation-phase trade-offs: metrics vs. beta, Pareto fronts, gains, summary at one beta."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter

from .. import config, style
from ..data import select
from ..dataset import Dataset
from ..methods import FIXED, TD3
from ..results import FRONTIER_GAINS, fixed_frontier, fixed_runs
from ..stats import t_interval
from .common import METRIC_LABELS, ci_of, errorbar, thousands

BETA_POSITIONS = np.arange(len(config.WEIGHTS))
DODGE = 0.09  # horizontal offset between methods on categorical beta axes
ANNOTATED_METHOD = TD3  # method whose gain at the focus beta is called out
PARETO_TICKS = (4000, 5000, 7000, 10000, 15000, 20000)


def _beta_axis(ax) -> None:
    ax.set_xticks(BETA_POSITIONS, [f"{w:g}" for w in config.WEIGHTS])
    ax.set_xlim(-0.4, len(config.WEIGHTS) - 0.6)
    ax.set_xlabel("Trade-off weight $\\beta$")


def tradeoff_vs_beta(data: Dataset, out: Path) -> None:
    """Rounds, unstable time and error of each learned method as beta grows."""
    metrics = ("ticks", "unstable", "error")
    fig, axes = style.figure("double", 2.0, 1, len(metrics))
    for ax, metric in zip(axes, metrics):
        for index, method in enumerate(data.methods):
            offset = (index - (len(data.methods) - 1) / 2) * DODGE
            intervals = [ci_of(select(data.evaluation, method.key, weight=w)[metric]) for w in config.WEIGHTS]
            means = np.array([ci.mean for ci in intervals])
            errors = np.array([[ci.mean - ci.low for ci in intervals], [ci.high - ci.mean for ci in intervals]])
            ax.errorbar(BETA_POSITIONS + offset, means, yerr=errors, elinewidth=0.6, **style.line_style(method))
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric == "ticks":
            thousands(ax, "y")
        _beta_axis(ax)
    style.figure_legend(fig, [style.legend_handle(m) for m in data.methods], ncol=len(data.methods))
    style.save(fig, out, "tradeoff_vs_beta")


def _period_axis(ax) -> None:
    """Secondary top axis: the same rounds expressed as the mean wake-up period."""

    def to_period(x):  # its own inverse; clipped so matplotlib may probe 0 safely
        return config.ticks_to_mean_period(np.maximum(np.asarray(x, dtype=float), 1.0))

    top = ax.secondary_xaxis("top", functions=(to_period, to_period))
    top.set_xticks([1, 1.5, 2, 3, 4, 5])
    top.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel("Mean wake-up period (s)", fontsize=7)
    top.tick_params(labelsize=6.5)


def _gain_callout(ax, ticks: float, value: float, frontier_value: float, gain: float) -> None:
    ax.annotate(
        "",
        xy=(ticks, value),
        xytext=(ticks, frontier_value),
        arrowprops={"arrowstyle": "->", "color": style.INK_SECONDARY, "linewidth": 0.7},
    )
    style.direct_label(ax, ticks, (value + frontier_value) / 2, f"$-${gain:.0%}", dx=3, color=style.INK, fontsize=6.5)


def pareto_frontiers(data: Dataset, out: Path) -> None:
    """Rounds vs. unstable time and vs. error: learned policies against fixed periods."""
    gains = data.gains.set_index(["method", "weight", "gain"])
    panels = (("unstable", "unstable_at_ticks"), ("error", "error_at_ticks"))
    fig, axes = style.figure("double", 2.75, 1, len(panels))
    for ax, (metric, gain_name) in zip(axes, panels):
        frontier = fixed_frontier(data.evaluation, metric)
        ax.plot(frontier.ticks, frontier[metric], **style.fixed_period_style())
        style.label_fixed_periods(ax, frontier.ticks, frontier[metric], frontier.period)
        for method in data.methods:
            points = []
            for weight in config.WEIGHTS:
                runs = select(data.evaluation, method.key, weight=weight)
                x, y = ci_of(runs.ticks), ci_of(runs[metric])
                errorbar(ax, x, y, method, label=None)
                points.append((x.mean, y.mean))
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=method.color, linewidth=0.7, alpha=0.8, zorder=1)
        # Call out the gain of one method at the focus beta.
        runs = select(data.evaluation, ANNOTATED_METHOD.key, weight=config.FOCUS_WEIGHT)
        ticks, value = runs.ticks.mean(), runs[metric].mean()
        gain = gains.loc[(ANNOTATED_METHOD.key, config.FOCUS_WEIGHT, gain_name), "mean"]
        _gain_callout(ax, ticks, value, value / (1 - gain), gain)
        ax.text(
            0.02,
            0.03,
            "learned: $\\beta$ = 0.5, 0.8, 0.95, 0.99, 1 from left to right",
            transform=ax.transAxes,
            fontsize=6,
            color=style.INK_MUTED,
        )
        ax.set_xscale("log")
        ax.set_xticks(PARETO_TICKS, [f"{t / 1000:g}k" for t in PARETO_TICKS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(f"{METRIC_LABELS['ticks']} (energy proxy, log scale)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.margins(y=0.1)
        _period_axis(ax)
    handles = [style.legend_handle(m, linestyle="-", linewidth=0.7) for m in data.methods]
    handles.append(Line2D([], [], **style.fixed_period_style()))
    style.figure_legend(fig, handles, ncol=len(handles))
    style.save(fig, out, "pareto_frontiers")


def _gain_cell_text(mean: float, low: float, high: float) -> str:
    if np.isnan(mean):
        return "lower error\nthan 1 s"
    return f"{mean:.0%}\n[{low:.0%}, {high:.0%}]".replace("-", "\u2212")  # typographic minus


def frontier_gains_heatmap(data: Dataset, out: Path) -> None:
    """Gains over the fixed-period frontier for every (method, beta), with 95 % bootstrap CIs."""
    gains = data.gains
    names = ("error_at_ticks", "unstable_at_ticks", "rounds_at_error")
    limit = 0.6
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    fig, axes = style.figure("double", 2.05, 1, len(names), sharey=True)
    for ax, name in zip(axes, names):
        table = gains[gains.gain == name].pivot(index="method", columns="weight")
        rows = [m.key for m in data.methods]
        mean, low, high = (table[column].reindex(rows).to_numpy() for column in ("mean", "low", "high"))
        image = ax.imshow(np.nan_to_num(mean, nan=0.0), cmap="RdBu", norm=norm, aspect="auto")
        for (r, c), value in np.ndenumerate(mean):
            if np.isnan(value):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="#ececec", zorder=2))
            light = np.isnan(value) or abs(value) < 0.35
            ax.text(
                c,
                r,
                _gain_cell_text(value, low[r, c], high[r, c]),
                ha="center",
                va="center",
                fontsize=5.5,
                color=style.INK if light else "white",
                zorder=3,
                linespacing=1.1,
            )
        ax.set_xticks(range(len(config.WEIGHTS)), [f"{w:g}" for w in config.WEIGHTS])
        ax.set_yticks(range(len(rows)), [m.short for m in data.methods])
        ax.set_xlabel("Trade-off weight $\\beta$")
        ax.set_title(FRONTIER_GAINS[name].capitalize())
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01)
    colorbar.set_label("Gain over fixed periods", fontsize=7)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    colorbar.outline.set_visible(False)
    style.save(fig, out, "frontier_gains_heatmap")


SUMMARY_PERIODS = (5.0, 3.0, 2.0)  # fixed periods shown next to the learned methods


def summary_beta095(data: Dataset, out: Path) -> None:
    """All four metrics at beta = 0.95: per-seed values (dots) and mean with 95 % CI."""
    weight = config.FOCUS_WEIGHT
    metrics = ("ticks", "unstable", "error", "reward")
    rows: list[tuple[str, str, str, object]] = []  # label, color, marker, runs
    baseline = fixed_runs(data.evaluation, weight)
    for period in SUMMARY_PERIODS:
        runs = baseline[np.isclose(baseline.period, period)]
        rows.append((f"Fixed {period:g} s", style.fixed_period_color(period), FIXED.marker, runs))
    for method in data.methods:
        rows.append((method.short, method.color, method.marker, select(data.evaluation, method.key, weight=weight)))
    positions = np.arange(len(rows))[::-1]
    rng = np.random.default_rng(config.RANDOM_SEED)
    fig, axes = style.figure("double", 1.9, 1, len(metrics), sharey=True)
    for ax, metric in zip(axes, metrics):
        for y, (_, color, marker, runs) in zip(positions, rows):
            values = runs[metric].to_numpy()
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(values, y + jitter, s=7, facecolors="none", edgecolors=color, linewidths=0.6, zorder=2)
            ci = t_interval(values)
            ax.errorbar(
                ci.mean,
                y,
                xerr=[[ci.mean - ci.low], [ci.high - ci.mean]],
                color=color,
                marker=marker,
                markersize=3.5,
                markeredgecolor="white",
                markeredgewidth=0.4,
                elinewidth=0.9,
                zorder=3,
            )
        ax.set_xlabel(METRIC_LABELS[metric])
        if metric == "ticks":
            thousands(ax, "x")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(positions, [row[0] for row in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    style.save(fig, out, "summary_beta095")


FIGURES = (tradeoff_vs_beta, pareto_frontiers, frontier_gains_heatmap, summary_beta095)
