"""Figures of the independent static / dynamic / zero-shot campaign, one folder per family under ``charts/``.

Every point is a mean over the (training seed, evaluation seed) pairs, after averaging the evaluation episodes of a
pair; whiskers are crossed-bootstrap 95% intervals (training and evaluation seeds resampled independently). The
fixed periods and the reactive baselines do not learn, so the same runs are paired with every training seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .. import methods, style
from ..campaign import PAIR, crossed_interval
from experiments.campaign import BASELINES, BETAS, FIXED, FOCUS_BETA, LEARNED

ARCHITECTURES = {0: methods.GNN, 1: methods.TD3}
ARCHITECTURE_LABELS = {0: "GNN", 1: "MLP"}
BASELINE_METHODS = dict(zip(BASELINES, (methods.TRICKLE, methods.GRADUAL)))
FIXED_PERIODS = dict(zip(FIXED, (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)))
VARIANT_LABELS = {
    "current": "Current",
    "legacy": "Legacy objective",
    "old-encoding": "Original encoding",
    "transferable": "Transferable features",
    "balanced": "+ balanced replay",
    "log-exploration": "+ log-uniform exploration",
    "long-training": "+ 30 training episodes",
    "matched-updates": "+ matched updates (10 ep.)",
    "agnostic": "Program-agnostic features",
    "error-cost": "Error cost",
    "wide-loss-scale": "500 m loss scale",
    "long-retention": "15 s retention",
    "paper-settings": "Original cost, scale, retention",
    "legacy-paper-steps": "Legacy, 110 steps per episode",
    "legacy-static-training": "Legacy, source switches only",
    "legacy-paper-training": "Legacy, paper training",
}
PROGRAMS = {0: "Gradient", 1: "Hop count", 2: "Broadcast", 3: "Collect", 4: "Election"}
PHASES = {"bootstrap": "Start-up\n0–40 s", "quiet": "Stable\n40–100 s", "event": "Event\n100–150 s",
          "after": "After\n150–200 s"}
# Once the network has converged every schedule holds an exact value: the error-based gain is defined only while the
# values are changing, the other phases are compared by their rounds.
CHANGING_PHASES = ("bootstrap", "event")
LONGEST_FIXED = 7
# Scenario families from the training one to the furthest from it.
FAMILY_ORDER = ("static", "dynamic", "generalization")
# The variant of a series: the final method filled, the current one hollow, the legacy control light and dotted.
VARIANT_FILL = {"final": "full", "current": "none", "legacy": "full"}
VARIANT_ALPHA = {"final": 1.0, "current": 1.0, "legacy": 0.45}
VARIANT_LINE = {"final": "-", "current": "-", "legacy": ":"}
# Room below the longest fixed period, as a share of the axis above it, that holds its label.
LabelRoom = 0.15
ERROR_LABEL = "Normalized error"
# Share of the colour range beyond which a heatmap cell is dark enough for white text.
HeatmapInkThreshold = 0.6
ROUNDS_LABEL = "Rounds per node per second"


def scenario(frame: pd.DataFrame, family: str, program: int, nodes: int, speed: float,
             phase: str = "whole") -> pd.DataFrame:
    return frame[(frame.family == family) & (frame.program == program) & (frame.nodes == nodes)
                 & (frame.speed == speed) & (frame.phase == phase)]


def log_axes(ax) -> None:
    """Log-log axes with plain decimal labels on 1-2-5 ticks (no crowded scientific notation)."""
    ax.set(xscale="log", yscale="log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_minor_formatter(NullFormatter())


def select(rows: pd.DataFrame, variant: str, algorithm: int, beta: float = FOCUS_BETA) -> pd.DataFrame:
    return rows[(rows.variant == variant) & (rows.algorithm == algorithm) & (rows.beta == beta)]


def estimate(rows: pd.DataFrame, value: str) -> dict:
    """Mean and crossed 95% interval of `value` over the seed pairs."""
    return crossed_interval(rows.groupby(PAIR, as_index=False)[value].mean(), value)


def fixed_frontier(rows: pd.DataFrame) -> pd.DataFrame:
    fixed = rows[rows.algorithm.isin(FIXED)].groupby("algorithm")[["roundsPerNodeSecond", "normalizedError"]].mean()
    fixed["period"] = [FIXED_PERIODS[algorithm] for algorithm in fixed.index]
    return fixed.sort_values("roundsPerNodeSecond")


def whiskers(point: dict) -> list[list[float]]:
    return [[point["mean"] - point["low"]], [point["high"] - point["mean"]]] if point["complete"] else None


def variant_style(method: methods.Method, role: str, **overrides) -> dict:
    return style.line_style(method, fillstyle=VARIANT_FILL[role], markeredgecolor=method.color,
                            alpha=VARIANT_ALPHA[role], **overrides)


def plot_point(ax, rows: pd.DataFrame, method: methods.Method, role: str, label: str | None) -> None:
    x, y = estimate(rows, "roundsPerNodeSecond"), estimate(rows, "normalizedError")
    ax.errorbar(x["mean"], y["mean"], xerr=whiskers(x), yerr=whiskers(y), linestyle="none", elinewidth=0.6,
                **variant_style(method, role, label=label))


def pareto_panel(ax, episodes: pd.DataFrame, where: tuple, finals: dict[int, str], phase: str = "whole") -> None:
    """Rounds vs error: fixed-period frontier, reactive baselines, and the variants compared for each architecture."""
    rows = scenario(episodes, *where, phase=phase)
    frontier = fixed_frontier(rows)
    ax.plot(frontier.roundsPerNodeSecond, frontier.normalizedError, **style.fixed_period_style(label="Fixed period"))
    style.label_fixed_periods(ax, frontier.roundsPerNodeSecond, frontier.normalizedError, frontier.period)
    for algorithm, method in BASELINE_METHODS.items():
        baseline = rows[rows.algorithm == algorithm]
        if not baseline.empty:
            plot_point(ax, baseline, method, "final", method.short)
    for algorithm in LEARNED:
        for role, variant in roles(finals[algorithm]).items():
            learned = select(rows, variant, algorithm)
            if not learned.empty:
                label = f"{ARCHITECTURE_LABELS[algorithm]}, {VARIANT_LABELS[variant].lower()}"
                plot_point(ax, learned, ARCHITECTURES[algorithm], role, label)
    log_axes(ax)
    ax.set(xlabel=ROUNDS_LABEL, ylabel=ERROR_LABEL)


def phase_gains_panel(ax, gains: pd.DataFrame, where: tuple, finals: dict[int, str]) -> None:
    """Error reduction w.r.t. the fixed-period frontier at matched rounds, per phase of the episode."""
    series = [(role, variant, algorithm) for algorithm in LEARNED for role, variant in roles(finals[algorithm]).items()]
    width = 0.8 / len(series)
    for index, (role, variant, algorithm) in enumerate(series):
        method = ARCHITECTURES[algorithm]
        points = [estimate(select(scenario(gains, *where, phase=phase), variant, algorithm), "gain")
                  for phase in CHANGING_PHASES]
        positions = np.arange(len(CHANGING_PHASES)) + (index - (len(series) - 1) / 2) * width
        means = [100 * point["mean"] for point in points]
        errors = np.array([[100 * (p["mean"] - p["low"]), 100 * (p["high"] - p["mean"])] if p["complete"]
                           else [np.nan, np.nan] for p in points]).T
        face = {"final": method.color, "current": "white", "legacy": to_rgba(method.color, VARIANT_ALPHA[role])}[role]
        ax.bar(positions, means, width, color=face, edgecolor=method.color,
               hatch="////" if role == "legacy" else None, linewidth=0.6,
               label=f"{ARCHITECTURE_LABELS[algorithm]}, {VARIANT_LABELS[variant].lower()}")
        ax.errorbar(positions, means, yerr=errors, linestyle="none", color=style.INK, elinewidth=0.6)
    ax.axhline(0, color=style.INK_MUTED, linewidth=0.6)
    ax.set_xticks(range(len(CHANGING_PHASES)), [PHASES[phase] for phase in CHANGING_PHASES])
    ax.set_ylabel("Error reduction at\nmatched rounds (%)")


def phase_rounds_panel(ax, episodes: pd.DataFrame, where: tuple, finals: dict[int, str]) -> None:
    """Rounds per node per second in every phase; the longest fixed period is the cheapest error-free schedule of a
    converged network."""
    positions = np.arange(len(PHASES))
    for algorithm in LEARNED:
        for role, variant in roles(finals[algorithm]).items():
            points = [estimate(select(scenario(episodes, *where, phase=phase), variant, algorithm),
                               "roundsPerNodeSecond") for phase in PHASES]
            if all(np.isfinite(point["mean"]) for point in points):
                ax.errorbar(positions, [point["mean"] for point in points],
                            yerr=np.array([whiskers(point) or [[np.nan], [np.nan]] for point in points])[:, :, 0].T,
                            linewidth=0.8, elinewidth=0.6, **variant_style(
                                ARCHITECTURES[algorithm], role, linestyle=VARIANT_LINE[role], label=None))
    longest_fixed_line(ax, scenario(episodes, *where))
    ax.set_xticks(positions, list(PHASES.values()))
    ax.set_ylabel(ROUNDS_LABEL)


def longest_fixed_line(ax, rows: pd.DataFrame) -> None:
    """The rounds of the longest fixed period, labelled below the line: no schedule can run fewer rounds."""
    longest = rows[rows.algorithm == LONGEST_FIXED].roundsPerNodeSecond.mean()
    ax.axhline(longest, **style.fixed_period_style(label=None, marker=None))
    low, high = ax.get_ylim()
    ax.set_ylim(min(low, longest - LabelRoom * (high - longest)), high)
    style.direct_label(ax, 0, longest, f"{FIXED_PERIODS[LONGEST_FIXED]:g} s fixed period", dx=0, dy=-2, va="top")


def roles(final: str) -> dict[str, str]:
    """The compared variants: the final method, the current control (when different) and the legacy control."""
    return {"final": final, **({"current": "current"} if final != "current" else {}), "legacy": "legacy"}


def family_figure(episodes, gains, out: Path, name: str, scenarios: list[tuple], titles: list[str],
                  finals: dict[int, str]) -> None:
    fig, axes = style.figure("double", 2.3 * len(scenarios), nrows=len(scenarios), ncols=3, squeeze=False,
                             layout="constrained", gridspec_kw={"width_ratios": (1.1, 0.8, 1.1)})
    for row, (where, title) in enumerate(zip(scenarios, titles)):
        pareto_panel(axes[row, 0], episodes, where, finals)
        phase_gains_panel(axes[row, 1], gains, where, finals)
        phase_rounds_panel(axes[row, 2], episodes, where, finals)
        axes[row, 0].set_title(title, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    style.figure_legend(fig, handles, labels, ncol=3)
    style.save(fig, out, name)


SIZES = (50, 100, 200)


def gain_grid(gains: pd.DataFrame, variant: str, algorithm: int) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """Gain (%) of a method for every program (rows) and population (columns), with the text of every cell."""
    table = np.full((len(PROGRAMS), len(SIZES)), np.nan)
    cells = []
    for (row, program), (column, nodes) in [(p, s) for p in enumerate(PROGRAMS) for s in enumerate(SIZES)]:
        family = "static" if (program, nodes) == (0, 100) else "generalization"
        rows = select(scenario(gains, family, program, nodes, 0.0), variant, algorithm)
        if not rows.empty:
            point = estimate(rows, "gain")
            table[row, column] = 100 * point["mean"]
            uncertain = not point["complete"] or point["low"] <= 0 <= point["high"]
            cells.append((row, column, f"{table[row, column]:.0f}" + ("*" if uncertain else "")))
    return table, cells


def generalization_figure(gains: pd.DataFrame, out: Path, finals: dict[int, str]) -> None:
    """Gain at matched rounds for every program and population, final method (static gradient at 100 nodes), on one
    colour scale shared by the architectures."""
    fig, axes = style.figure("double", 2.4, ncols=len(LEARNED), layout="constrained")
    grids = [gain_grid(gains, finals[algorithm], algorithm) for algorithm in LEARNED]
    finite = np.concatenate([np.abs(table[np.isfinite(table)]) for table, _ in grids])
    limit = finite.max() if finite.size else 1.0
    for ax, algorithm, (table, cells) in zip(axes, LEARNED, grids):
        final = finals[algorithm]
        for row, column, text in cells:
            strong = abs(table[row, column]) > limit * HeatmapInkThreshold
            ax.text(column, row, text, ha="center", va="center", fontsize=7, color="white" if strong else style.INK)
        ax.grid(False)
        image = ax.imshow(table, cmap=style.DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(SIZES)), [f"{size} nodes" for size in SIZES])
        ax.set_yticks(range(len(PROGRAMS)), list(PROGRAMS.values()))
        ax.set_title(f"{ARCHITECTURE_LABELS[algorithm]}, {VARIANT_LABELS[final].lower()}", loc="left")
    fig.colorbar(image, ax=axes, label="Error reduction at matched rounds (%)\n* 95% interval includes 0", shrink=.9)
    style.save(fig, out, "zero_shot_gains")


def improvement_figure(gains: pd.DataFrame, out: Path, finals: dict[int, str], control: str = "current") -> None:
    """Paired difference of the gain, final minus control, per scenario, for the architectures whose final method
    is not the control."""
    improved = [algorithm for algorithm in LEARNED if finals[algorithm] != control]
    if not improved:
        return
    whole = gains[gains.phase == "whole"]
    keys = sorted(whole[["family", "program", "nodes", "speed"]].drop_duplicates().itertuples(index=False),
                  key=lambda key: (FAMILY_ORDER.index(key.family), *key[1:]))
    fig, ax = style.figure("single", 0.22 * len(keys) + 0.8, layout="constrained")
    for offset, algorithm in zip(np.linspace(-0.15, 0.15, len(improved)), improved):
        final = finals[algorithm]
        for index, key in enumerate(keys):
            rows = scenario(whole, *key)
            rows = rows[(rows.algorithm == algorithm) & (rows.beta == FOCUS_BETA)]
            paired = rows.pivot_table(index=PAIR, columns="variant", values="gain")
            if {final, control} <= set(paired.columns):
                difference = (paired[final] - paired[control]).rename("difference").reset_index()
                point = crossed_interval(difference, "difference")
                ax.errorbar(100 * point["mean"], index + offset,
                            xerr=[[100 * (point["mean"] - point["low"])], [100 * (point["high"] - point["mean"])]]
                            if point["complete"] else None, linestyle="none", elinewidth=0.6,
                            **style.line_style(ARCHITECTURES[algorithm], label=ARCHITECTURE_LABELS[algorithm]
                                               if index == 0 else None))
    ax.axvline(0, color=style.INK_MUTED, linewidth=0.6)
    ax.set_yticks(range(len(keys)), [scenario_label(*key) for key in keys])
    ax.invert_yaxis()
    ax.set_xlabel(f"Gain of the final method − {VARIANT_LABELS[control].lower()} (points)")
    ax.legend(loc="upper right")
    style.save(fig, out, "improvement_over_current")


def scenario_label(family: str, program: int, nodes: int, speed: float) -> str:
    if family == "dynamic":
        return f"Gradient, motion {speed:g} m/s"
    return f"{PROGRAMS[program]}, {nodes} nodes"


def beta_figure(episodes: pd.DataFrame, out: Path, finals: dict[int, str], scenarios: list[tuple]) -> None:
    """Rounds vs error of the separately trained policies at every beta, with the reactive baselines (one point each:
    they ignore beta)."""
    columns = 3
    rows_count = int(np.ceil(len(scenarios) / columns))
    fig, axes = style.figure("double", 2.1 * rows_count, nrows=rows_count, ncols=columns, squeeze=False,
                             layout="constrained")
    for ax, where in zip(axes.flat, scenarios):
        rows = scenario(episodes, *where)
        frontier = fixed_frontier(rows)
        ax.plot(frontier.roundsPerNodeSecond, frontier.normalizedError, **style.fixed_period_style(label="Fixed period"))
        for algorithm, method in BASELINE_METHODS.items():
            point = rows[rows.algorithm == algorithm][["roundsPerNodeSecond", "normalizedError"]].mean()
            ax.plot(point.roundsPerNodeSecond, point.normalizedError, linestyle="none",
                    **style.line_style(method, label=method.short))
        for key, method, label in [((finals[algorithm], algorithm), ARCHITECTURES[algorithm],
                                    f"{ARCHITECTURE_LABELS[algorithm]}, {VARIANT_LABELS[finals[algorithm]].lower()}")
                                   for algorithm in LEARNED]:
            subset = rows[(rows.variant == key[0]) & (rows.algorithm == key[1])]
            means = subset.groupby("beta")[["roundsPerNodeSecond", "normalizedError"]].mean().reindex(BETAS)
            ax.plot(means.roundsPerNodeSecond, means.normalizedError, linewidth=0.8,
                    **style.line_style(method, label=label))
            ends = means.dropna().iloc[[0, -1]] if key[1] == LEARNED[0] and len(means.dropna()) > 1 else means[:0]
            for beta, point in ends.iterrows():
                style.direct_label(ax, point.roundsPerNodeSecond, point.normalizedError, f"β={beta:g}", fontsize=5)
        log_axes(ax)
        ax.set_title(scenario_label(*where[:4]), loc="left")
    for ax in axes[-1]:
        ax.set_xlabel(ROUNDS_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(ERROR_LABEL)
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    style.figure_legend(fig, handles, labels, ncol=4)
    style.save(fig, out, "beta_tradeoff")


def ablation_figure(gains: pd.DataFrame, out: Path, groups: dict[str, list[tuple]], name: str) -> None:
    """Validation: paired gain difference of every variant w.r.t. the current one, per scenario group."""
    whole = gains[(gains.phase == "whole") & gains.algorithm.isin(LEARNED) & (gains.variant != "fixed")]
    variants = [variant for variant in VARIANT_LABELS if variant in set(whole.variant) and variant != "current"]
    fig, axes = style.figure("double", 0.25 * len(variants) + 0.9, ncols=len(groups), sharey=True,
                             layout="constrained", squeeze=False)
    for ax, (title, members) in zip(axes[0], groups.items()):
        rows = pd.concat([scenario(whole, *member) for member in members])
        for offset, algorithm in zip((-0.15, 0.15), LEARNED):
            per_pair = rows[rows.algorithm == algorithm].groupby(["variant", *PAIR]).gain.mean().unstack("variant")
            for index, variant in enumerate(variants):
                difference = (per_pair[variant] - per_pair["current"]).rename("difference").reset_index()
                point = crossed_interval(difference, "difference")
                ax.errorbar(100 * point["mean"], index + offset, xerr=[[100 * (point["mean"] - point["low"])],
                            [100 * (point["high"] - point["mean"])]] if point["complete"] else None,
                            linestyle="none", elinewidth=0.6, **style.line_style(
                                ARCHITECTURES[algorithm], label=ARCHITECTURE_LABELS[algorithm] if index == 0 else None))
        ax.axvline(0, color=style.INK_MUTED, linewidth=0.6)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Gain − current (points)")
    axes[0, 0].set_yticks(range(len(variants)), [VARIANT_LABELS[variant] for variant in variants])
    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    style.figure_legend(fig, handles, labels, ncol=2)
    style.save(fig, out, name)
