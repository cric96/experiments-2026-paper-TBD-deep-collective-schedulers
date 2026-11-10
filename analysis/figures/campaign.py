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
from ..campaign import PAIR, at_focus, crossed_interval
from experiments.campaign import BASELINES, FIXED, LEARNED, focus_of, grid_of, tuned_control_of

ARCHITECTURES = {0: methods.GNN, 1: methods.TD3}
ARCHITECTURE_LABELS = {0: "GNN", 1: "MLP"}
BASELINE_METHODS = dict(zip(BASELINES, (methods.TRICKLE, methods.GRADUAL)))
# The variant of the untuned reactive baselines: shared controls, like the fixed periods.
UNTUNED = "fixed"
# The shared controls compared with the final method of each architecture, when different from it.
CONTROLS = ("current", "legacy")
# The reactive baselines tuned by beta join their betas with a dash-dotted line.
TUNED_LINE = "-."
# The beta labels at the ends of a curve: above the markers, on a light background that keeps them legible.
LABEL_ZORDER = 6
LABEL_BOX = {"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.75}
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
    "legacy-quiet": "Legacy, quiet steady state",
    "legacy-graded": "Legacy, graded reaction",
    "legacy-quiet-awaiting": "Legacy, quiet, awaits values",
    "legacy-graded-awaiting": "Legacy, graded, awaits values",
    "legacy-linear-awaiting": "Legacy, linear β, awaits values",
    "legacy-linear-patient": "Legacy, linear β, waits quietly",
    "legacy-full-range": "Full-range β",
    "legacy-full-range-reactive": "Full-range β, reactive",
    "error-full-range": "Full-range β, error objective",
    "neighbourhood-error-full-range": "Full-range β, neighbourhood error",
    "network-error-full-range": "Full-range β, network error",
    "network-error-discount-0.5": "Full-range β, network error (ρ 0.5)",
    "network-error-discount-0.75": "Full-range β, network error (ρ 0.75)",
    "network-error-long-training": "Full-range β, network error, 3× training",
    "network-error-wide-price": "Full-range β, network error, cost reference 0.1 s",
    "network-error-squared-loss": "Full-range β, network error, squared loss",
    "network-error-discount-0.5-unprivileged": "Full-range β, network error (ρ 0.5), plain critic",
    "network-error-discount-0.5-static-training": "Full-range β, network error (ρ 0.5), static training",
    "network-error-discount-0.5-rich-context": "Full-range β, network error (ρ 0.5), rich context",
    "network-error-discount-0.5-more-data": "Full-range β, network error (ρ 0.5), 20 training episodes",
    "network-error-discount-0.5-rich-context-more-data":
        "Full-range β, network error (ρ 0.5), rich context, 20 training episodes",
    "network-error-discount-0.5-gamma-0.95-two-switches": "Full-range β, network error (ρ 0.5), γ 0.95, two switches",
    "network-error-discount-0.5-unprivileged-gamma-0.95-two-switches":
        "Full-range β, network error (ρ 0.5), plain critic, γ 0.95, two switches",
    "network-error-hybrid-0.1": "Full-range β, hybrid network error, surrogate weight 0.1",
    "network-error-hybrid-0.3": "Full-range β, hybrid network error, surrogate weight 0.3",
    "network-error-hybrid-1": "Full-range β, hybrid network error, surrogate weight 1",
    "network-error-hybrid-0.5-more-data": "Full-range β, hybrid network error, surrogate weight 0.5, 20 training episodes",
    "network-error-hybrid-1-more-data": "Full-range β, hybrid network error, surrogate weight 1, 20 training episodes",
    "network-error-hybrid-2-more-data": "Full-range β, hybrid network error, surrogate weight 2, 20 training episodes",
    "network-error-hybrid-4-more-data": "Full-range β, hybrid network error, surrogate weight 4, 20 training episodes",
    "network-error-hybrid-scheduled": "Full-range β, hybrid network error, surrogate weight 0.2-2 with β",
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
GAIN_LABEL = "Error reduction at\nmatched rounds (%)"
# The rounds of the beta studies span less than a decade (about 0.25-1.4 per node and second): 1-2-3-5 ticks.
GAIN_ROUNDS_TICKS = (1.0, 2.0, 3.0, 5.0)


def scenario(frame: pd.DataFrame, family: str, program: int, nodes: int, speed: float,
             phase: str = "whole") -> pd.DataFrame:
    return frame[(frame.family == family) & (frame.program == program) & (frame.nodes == nodes)
                 & (frame.speed == speed) & (frame.phase == phase)]


def log_axis(axis, subs: tuple[float, ...] = (1.0, 2.0, 5.0)) -> None:
    """Plain decimal labels on the `subs` ticks of every decade of a log axis (no crowded scientific notation)."""
    axis.set_major_locator(LogLocator(subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


def log_axes(ax) -> None:
    """Log-log axes with plain decimal labels on 1-2-5 ticks."""
    ax.set(xscale="log", yscale="log")
    for axis in (ax.xaxis, ax.yaxis):
        log_axis(axis)


def select(rows: pd.DataFrame, variant: str, algorithm: int, beta: float | None = None) -> pd.DataFrame:
    """The rows of a method, at the focus beta of its variant unless `beta` is given."""
    beta = focus_of(variant) if beta is None else beta
    return rows[(rows.variant == variant) & (rows.algorithm == algorithm) & (rows.beta == beta)]


def reactive_variant(variants, finals: dict[int, str]) -> str:
    """The reactive baselines compared with the final methods: tuned by beta on their scale when the campaign has
    them, else the untuned ones."""
    tuned = tuned_control_of(finals[LEARNED[0]])
    return tuned if tuned in set(variants) else UNTUNED


def untuned_points(baselines: str) -> bool:
    """The untuned reactive baselines (one point each: they ignore beta) are drawn only when they are the compared
    baselines: the baselines tuned by beta replace them."""
    return baselines == UNTUNED


def baseline_label(method: methods.Method, baselines: str) -> str:
    return method.short if baselines == UNTUNED else f"{method.short}, tuned by β"


def baseline_title(method: methods.Method, baselines: str) -> str:
    """The full name of an untuned baseline; the short one of a tuned baseline, which says how it is tuned."""
    return method.label if baselines == UNTUNED else baseline_label(method, baselines)


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


def pareto_panel(ax, episodes: pd.DataFrame, where: tuple, finals: dict[int, str], phase: str = "whole",
                 baselines: str = UNTUNED, controls: tuple[str, ...] = CONTROLS) -> None:
    """Rounds vs error: fixed-period frontier, reactive baselines, and the variants compared for each architecture."""
    rows = scenario(episodes, *where, phase=phase)
    frontier = fixed_frontier(rows)
    ax.plot(frontier.roundsPerNodeSecond, frontier.normalizedError, **style.fixed_period_style(label="Fixed period"))
    style.label_fixed_periods(ax, frontier.roundsPerNodeSecond, frontier.normalizedError, frontier.period)
    for algorithm, method in BASELINE_METHODS.items():
        baseline = select(rows, baselines, algorithm)
        if not baseline.empty:
            plot_point(ax, baseline, method, "final", baseline_label(method, baselines))
    for algorithm in LEARNED:
        for role, variant in roles(finals[algorithm], controls).items():
            learned = select(rows, variant, algorithm)
            if not learned.empty:
                label = f"{ARCHITECTURE_LABELS[algorithm]}, {VARIANT_LABELS[variant].lower()}"
                plot_point(ax, learned, ARCHITECTURES[algorithm], role, label)
    log_axes(ax)
    ax.set(xlabel=ROUNDS_LABEL, ylabel=ERROR_LABEL)


def compared_series(finals: dict[int, str], baselines: str = UNTUNED,
                    controls: tuple[str, ...] = CONTROLS) -> list[tuple[str, str, int, methods.Method, str]]:
    """(role, variant, algorithm, method, label) of every compared series: the variants of each architecture, then the
    reactive baselines (drawn like a final method)."""
    learned = [(role, variant, algorithm, ARCHITECTURES[algorithm],
                f"{ARCHITECTURE_LABELS[algorithm]}, {VARIANT_LABELS[variant].lower()}")
               for algorithm in LEARNED for role, variant in roles(finals[algorithm], controls).items()]
    return learned + [("final", baselines, algorithm, method, baseline_label(method, baselines))
                      for algorithm, method in BASELINE_METHODS.items()]


def phase_gains_panel(ax, gains: pd.DataFrame, where: tuple, finals: dict[int, str],
                      baselines: str = UNTUNED, controls: tuple[str, ...] = CONTROLS) -> None:
    """Error reduction w.r.t. the fixed-period frontier at matched rounds, per phase of the episode."""
    series = compared_series(finals, baselines, controls)
    width = 0.8 / len(series)
    for index, (role, variant, algorithm, method, label) in enumerate(series):
        points = [estimate(select(scenario(gains, *where, phase=phase), variant, algorithm), "gain")
                  for phase in CHANGING_PHASES]
        positions = np.arange(len(CHANGING_PHASES)) + (index - (len(series) - 1) / 2) * width
        means = [100 * point["mean"] for point in points]
        errors = np.array([[100 * (p["mean"] - p["low"]), 100 * (p["high"] - p["mean"])] if p["complete"]
                           else [np.nan, np.nan] for p in points]).T
        face = {"final": method.color, "current": "white", "legacy": to_rgba(method.color, VARIANT_ALPHA[role])}[role]
        ax.bar(positions, means, width, color=face, edgecolor=method.color,
               hatch="////" if role == "legacy" else None, linewidth=0.6, label=label)
        ax.errorbar(positions, means, yerr=errors, linestyle="none", color=style.INK, elinewidth=0.6)
    ax.axhline(0, color=style.INK_MUTED, linewidth=0.6)
    ax.set_xticks(range(len(CHANGING_PHASES)), [PHASES[phase] for phase in CHANGING_PHASES])
    ax.set_ylabel(GAIN_LABEL)


def phase_rounds_panel(ax, episodes: pd.DataFrame, where: tuple, finals: dict[int, str],
                       baselines: str = UNTUNED, controls: tuple[str, ...] = CONTROLS) -> None:
    """Rounds per node per second in every phase; the longest fixed period is the cheapest error-free schedule of a
    converged network."""
    positions = np.arange(len(PHASES))
    for role, variant, algorithm, method, _ in compared_series(finals, baselines, controls):
        points = [estimate(select(scenario(episodes, *where, phase=phase), variant, algorithm),
                           "roundsPerNodeSecond") for phase in PHASES]
        if all(np.isfinite(point["mean"]) for point in points):
            ax.errorbar(positions, [point["mean"] for point in points],
                        yerr=np.array([whiskers(point) or [[np.nan], [np.nan]] for point in points])[:, :, 0].T,
                        linewidth=0.8, elinewidth=0.6,
                        **variant_style(method, role, linestyle=VARIANT_LINE[role], label=None))
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


def roles(final: str, controls: tuple[str, ...] = CONTROLS) -> dict[str, str]:
    """The compared variants: the final method and, when different from it, each of the `controls` (its own role)."""
    return {"final": final, **{control: control for control in controls if control != final}}


def family_figure(episodes, gains, out: Path, name: str, scenarios: list[tuple], titles: list[str],
                  finals: dict[int, str], baselines: str = UNTUNED, controls: tuple[str, ...] = CONTROLS) -> None:
    fig, axes = style.figure("double", 2.3 * len(scenarios), nrows=len(scenarios), ncols=3, squeeze=False,
                             layout="constrained", gridspec_kw={"width_ratios": (1.1, 0.8, 1.1)})
    for row, (where, title) in enumerate(zip(scenarios, titles)):
        pareto_panel(axes[row, 0], episodes, where, finals, baselines=baselines, controls=controls)
        phase_gains_panel(axes[row, 1], gains, where, finals, baselines, controls)
        phase_rounds_panel(axes[row, 2], episodes, where, finals, baselines, controls)
        axes[row, 0].set_title(title, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    style.figure_legend(fig, handles, labels, ncol=3)
    style.save(fig, out, name)


SIZES = (50, 100, 200)


def gain_grid(gains: pd.DataFrame, variant: str, algorithm: int,
              interval: bool = True) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """Gain (%) of a method for every program (rows) and population (columns), with the text of every cell; a star
    marks a 95% interval that includes 0 or cannot be computed, unless the method has no `interval` (a schedule run
    once per topology)."""
    table = np.full((len(PROGRAMS), len(SIZES)), np.nan)
    cells = []
    for (row, program), (column, nodes) in [(p, s) for p in enumerate(PROGRAMS) for s in enumerate(SIZES)]:
        family = "static" if (program, nodes) == (0, 100) else "generalization"
        rows = select(scenario(gains, family, program, nodes, 0.0), variant, algorithm)
        if not rows.empty:
            point = estimate(rows, "gain")
            table[row, column] = 100 * point["mean"]
            uncertain = interval and (not point["complete"] or point["low"] <= 0 <= point["high"])
            cells.append((row, column, f"{table[row, column]:.0f}" + ("*" if uncertain else "")))
    return table, cells


def zero_shot_panels(finals: dict[int, str], baselines: str = UNTUNED) -> list[tuple[str, int, str, bool]]:
    """(variant, algorithm, title, interval) of the final learned methods, then of the reactive baselines (run once
    per topology, under the first training seed: no interval over the training seeds)."""
    panels = [(final, algorithm, f"{ARCHITECTURE_LABELS[algorithm]}, {VARIANT_LABELS[final].lower()}", True)
              for algorithm, final in ((algorithm, finals[algorithm]) for algorithm in LEARNED)]
    return panels + [(baselines, algorithm, f"{baseline_title(method, baselines)}, one run per topology", False)
                     for algorithm, method in BASELINE_METHODS.items()]


def generalization_figure(gains: pd.DataFrame, out: Path, finals: dict[int, str], baselines: str = UNTUNED) -> None:
    """Gain at matched rounds for every program and population, final methods (static gradient at 100 nodes) and
    reactive baselines, on one colour scale."""
    panels = zero_shot_panels(finals, baselines)
    fig, axes = style.figure("double", 4.6, 2, len(panels) // 2, layout="constrained")
    grids = [gain_grid(gains, variant, algorithm, interval) for variant, algorithm, _, interval in panels]
    finite = np.concatenate([np.abs(table[np.isfinite(table)]) for table, _ in grids])
    limit = finite.max() if finite.size else 1.0
    for ax, (_, _, title, _), (table, cells) in zip(axes.ravel(), panels, grids):
        for row, column, text in cells:
            strong = abs(table[row, column]) > limit * HeatmapInkThreshold
            ax.text(column, row, text, ha="center", va="center", fontsize=7, color="white" if strong else style.INK)
        ax.grid(False)
        image = ax.imshow(table, cmap=style.DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(SIZES)), [f"{size} nodes" for size in SIZES])
        ax.set_yticks(range(len(PROGRAMS)), list(PROGRAMS.values()))
        ax.set_title(title, loc="left")
    fig.colorbar(image, ax=axes, label="Error reduction at matched rounds (%)\n* 95% interval includes 0 (learned)",
                 shrink=.9)
    style.save(fig, out, "zero_shot_gains")


def scenario_keys(whole: pd.DataFrame) -> list[tuple]:
    """The scenarios of a campaign: the training scenario, motion, then the zero-shot ones by program and size."""
    return sorted(whole[["family", "program", "nodes", "speed"]].drop_duplicates().itertuples(index=False),
                  key=lambda key: (FAMILY_ORDER.index(key.family), *key[1:]))


def forest(ax, keys: list[tuple], points: dict[int, list[dict | None]]) -> None:
    """One row per scenario and one marker per architecture: the mean (points) and its crossed 95% interval."""
    for offset, (algorithm, row) in zip(np.linspace(-0.15, 0.15, len(points)), points.items()):
        labelled = False
        for index, point in enumerate(row):
            if point is None:
                continue
            ax.errorbar(100 * point["mean"], index + offset,
                        xerr=[[100 * (point["mean"] - point["low"])], [100 * (point["high"] - point["mean"])]]
                        if point["complete"] else None, linestyle="none", elinewidth=0.6,
                        **style.line_style(ARCHITECTURES[algorithm],
                                           label=None if labelled else ARCHITECTURE_LABELS[algorithm]))
            labelled = True
    ax.axvline(0, color=style.INK_MUTED, linewidth=0.6)
    ax.set_yticks(range(len(keys)), [scenario_label(*key) for key in keys])
    ax.invert_yaxis()


def control_difference(rows: pd.DataFrame, final: str, control: str, algorithm: int) -> dict | None:
    """Paired difference of the gain of an architecture's final method from that of a control, at their focus betas."""
    rows = at_focus(rows[rows.algorithm == algorithm])
    paired = rows.pivot_table(index=PAIR, columns="variant", values="gain")
    if not {final, control} <= set(paired.columns):
        return None
    return crossed_interval((paired[final] - paired[control]).rename("difference").reset_index(), "difference")


def improvement_figure(gains: pd.DataFrame, out: Path, finals: dict[int, str], control: str = "current") -> None:
    """Paired difference of the gain, final minus control, per scenario, for the architectures whose final method
    is not the control."""
    improved = [algorithm for algorithm in LEARNED if finals[algorithm] != control]
    if not improved:
        return
    whole = gains[gains.phase == "whole"]
    keys = scenario_keys(whole)
    fig, ax = style.figure("single", 0.22 * len(keys) + 0.8, layout="constrained")
    forest(ax, keys, {algorithm: [control_difference(scenario(whole, *key), finals[algorithm], control, algorithm)
                                  for key in keys] for algorithm in improved})
    ax.set_xlabel(f"Gain of the final method − {VARIANT_LABELS[control].lower()} (points)")
    ax.legend(loc="upper right")
    style.save(fig, out, "improvement_over_current")


def better_baseline(rows: pd.DataFrame, baselines: str) -> int:
    """The reactive baseline with the larger mean gain in `rows` (one scenario), at the focus beta of `baselines`."""
    means = {algorithm: select(rows, baselines, algorithm).gain.mean() for algorithm in BASELINE_METHODS}
    return max((algorithm for algorithm in means if np.isfinite(means[algorithm])), key=means.get,
               default=BASELINES[0])


def paired_advantage(rows: pd.DataFrame, final: str, algorithm: int, baselines: str, reference: int) -> dict | None:
    """Paired difference (same training and evaluation seeds) of the gain of a final method from that of a reactive
    baseline, each at the focus beta of its scale."""
    learned = select(rows, final, algorithm).groupby(PAIR).gain.mean()
    baseline = select(rows, baselines, reference).groupby(PAIR).gain.mean()
    difference = (learned - baseline).dropna().rename("difference").reset_index()
    return None if difference.empty else crossed_interval(difference, "difference")


def zero_shot_separator(ax, keys: list[tuple]) -> None:
    """A rule between the scenarios of the training (static and motion) and the zero-shot ones."""
    first = next((index for index, key in enumerate(keys) if key[0] == "generalization"), None)
    if first is None:
        return
    ax.axhline(first - 0.5, color=style.INK_MUTED, linewidth=0.5, linestyle=":")
    ax.text(0.99, first - 0.4, "zero-shot", transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=6,
            color=style.INK_MUTED)


def advantage_figure(gains: pd.DataFrame, out: Path, finals: dict[int, str], baselines: str = UNTUNED) -> None:
    """Paired difference of the gain of each final method from that of the better reactive baseline (tuned by beta,
    when the campaign has them) in every scenario, each at the focus beta of its scale."""
    whole = gains[gains.phase == "whole"]
    keys = scenario_keys(whole)
    points = {algorithm: [] for algorithm in LEARNED}
    for key in keys:
        rows = scenario(whole, *key)
        reference = better_baseline(rows, baselines)
        for algorithm in LEARNED:
            points[algorithm].append(paired_advantage(rows, finals[algorithm], algorithm, baselines, reference))
    fig, ax = style.figure("single", 0.22 * len(keys) + 0.9, layout="constrained")
    forest(ax, keys, points)
    zero_shot_separator(ax, keys)
    ax.text(0.01, 1.01, "← baseline better", transform=ax.transAxes, ha="left", va="bottom", fontsize=6,
            color=style.INK_MUTED)
    ax.text(0.99, 1.01, "learned better →", transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
            color=style.INK_MUTED)
    tuning = "" if untuned_points(baselines) else "tuned "
    ax.set_xlabel(f"Gain over the better {tuning}baseline (points)")
    ax.legend(loc="lower right")
    style.save(fig, out, "advantage_over_baselines")


def gain_curve(ax, rows: pd.DataFrame, variant: str, algorithm: int, method: methods.Method, label: str,
               **overrides) -> pd.DataFrame:
    """The rounds and the gain (%, crossed 95% interval) of a method at every beta of its grid that the campaign has,
    joined in the order of the betas."""
    points = {beta: (estimate(subset, "roundsPerNodeSecond"), estimate(subset, "gain"))
              for beta in grid_of(variant)[0] if not (subset := select(rows, variant, algorithm, beta)).empty}
    means = pd.DataFrame({"roundsPerNodeSecond": [rounds["mean"] for rounds, _ in points.values()],
                          "gain": [100 * gain["mean"] for _, gain in points.values()]}, index=list(points))
    errors = np.array([[100 * (gain["mean"] - gain["low"]), 100 * (gain["high"] - gain["mean"])] if gain["complete"]
                       else [np.nan, np.nan] for _, gain in points.values()]).reshape(-1, 2).T
    ax.errorbar(means.roundsPerNodeSecond, means.gain, yerr=errors, linewidth=0.8, elinewidth=0.6,
                **style.line_style(method, label=label, **overrides))
    return means


def frontier_gains_figure(gains: pd.DataFrame, out: Path, finals: dict[int, str], scenarios: list[tuple],
                          baselines: str = UNTUNED) -> None:
    """The trade-off fronts of the beta study seen from the fixed-period frontier (the zero line): the gain at matched
    rounds of every method at every beta, on the training scenario, with motion and on the zero-shot programs."""
    columns = 3
    rows_count = int(np.ceil(len(scenarios) / columns))
    fig, axes = style.figure("double", 2.1 * rows_count, nrows=rows_count, ncols=columns, squeeze=False,
                             layout="constrained")
    for ax, where in zip(axes.flat, scenarios):
        rows = scenario(gains, *where)
        ax.axhline(0, **style.fixed_period_style(label="Fixed period", marker=None))
        for algorithm, method in BASELINE_METHODS.items():
            if untuned_points(baselines):
                point = estimate(select(rows, UNTUNED, algorithm), "gain")
                rounds = estimate(select(rows, UNTUNED, algorithm), "roundsPerNodeSecond")
                ax.plot(rounds["mean"], 100 * point["mean"], linestyle="none",
                        **style.line_style(method, label=method.short))
            else:
                gain_curve(ax, rows, baselines, algorithm, method, baseline_label(method, baselines),
                           linestyle=TUNED_LINE)
        for algorithm in LEARNED:
            means = gain_curve(ax, rows, finals[algorithm], algorithm, ARCHITECTURES[algorithm],
                               f"{ARCHITECTURE_LABELS[algorithm]}, {VARIANT_LABELS[finals[algorithm]].lower()}")
            if algorithm == LEARNED[0]:
                label_beta_ends(ax, means, "gain")
        ax.set_xscale("log")
        log_axis(ax.xaxis, GAIN_ROUNDS_TICKS)
        zero_shot = " (zero-shot)" if where[0] == "generalization" else ""
        ax.set_title(f"{scenario_label(*where[:4])}{zero_shot}", loc="left")
    for ax in axes[-1]:
        ax.set_xlabel(ROUNDS_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(GAIN_LABEL)
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    style.figure_legend(fig, handles, labels, ncol=legend_columns(len(handles)))
    style.save(fig, out, "frontier_gains_programs")


# A figure legend fits this many entries on one row; more go on two.
LEGEND_ROW = 5


def legend_columns(entries: int) -> int:
    return entries if entries <= LEGEND_ROW else (entries + 1) // 2


def scenario_label(family: str, program: int, nodes: int, speed: float) -> str:
    if family == "dynamic":
        return f"Gradient, motion {speed:g} m/s"
    return f"{PROGRAMS[program]}, {nodes} nodes"


def beta_curve(ax, rows: pd.DataFrame, variant: str, algorithm: int, method: methods.Method, label: str,
               **overrides) -> pd.DataFrame:
    """The mean rounds and error of a method at every beta of its grid, joined in the order of the betas."""
    subset = rows[(rows.variant == variant) & (rows.algorithm == algorithm)]
    means = subset.groupby("beta")[["roundsPerNodeSecond", "normalizedError"]].mean().reindex(grid_of(variant)[0])
    ax.plot(means.roundsPerNodeSecond, means.normalizedError, linewidth=0.8,
            **style.line_style(method, label=label, **overrides))
    return means


def label_beta_ends(ax, means: pd.DataFrame, value: str = "normalizedError") -> None:
    """The betas at the two ends of a curve of `value` against the rounds."""
    ends = means.dropna()
    for beta, point in (ends.iloc[[0, -1]] if len(ends) > 1 else ends[:0]).iterrows():
        style.direct_label(ax, point.roundsPerNodeSecond, point[value], f"β={beta:g}", fontsize=5,
                           zorder=LABEL_ZORDER, bbox=LABEL_BOX)


def beta_figure(episodes: pd.DataFrame, out: Path, finals: dict[int, str], scenarios: list[tuple],
                baselines: str = UNTUNED) -> None:
    """Rounds vs error of the separately trained policies at every beta, with the reactive baselines: tuned by beta
    (one point per beta) when the campaign has them, untuned otherwise (one point each: they ignore beta)."""
    columns = 3
    rows_count = int(np.ceil(len(scenarios) / columns))
    fig, axes = style.figure("double", 2.1 * rows_count, nrows=rows_count, ncols=columns, squeeze=False,
                             layout="constrained")
    for ax, where in zip(axes.flat, scenarios):
        rows = scenario(episodes, *where)
        frontier = fixed_frontier(rows)
        ax.plot(frontier.roundsPerNodeSecond, frontier.normalizedError,
                **style.fixed_period_style(label="Fixed period"))
        for algorithm, method in BASELINE_METHODS.items():
            if untuned_points(baselines):
                point = rows[(rows.variant == UNTUNED) & (rows.algorithm == algorithm)][
                    ["roundsPerNodeSecond", "normalizedError"]].mean()
                ax.plot(point.roundsPerNodeSecond, point.normalizedError, linestyle="none",
                        **style.line_style(method, label=method.short))
            else:
                beta_curve(ax, rows, baselines, algorithm, method, baseline_label(method, baselines),
                           linestyle=TUNED_LINE)
        for algorithm in LEARNED:
            means = beta_curve(ax, rows, finals[algorithm], algorithm, ARCHITECTURES[algorithm],
                               f"{ARCHITECTURE_LABELS[algorithm]}, {VARIANT_LABELS[finals[algorithm]].lower()}")
            if algorithm == LEARNED[0]:
                label_beta_ends(ax, means)
        log_axes(ax)
        ax.set_title(scenario_label(*where[:4]), loc="left")
    for ax in axes[-1]:
        ax.set_xlabel(ROUNDS_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(ERROR_LABEL)
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    style.figure_legend(fig, handles, labels, ncol=legend_columns(len(handles)))
    style.save(fig, out, "beta_tradeoff")


def ablation_figure(gains: pd.DataFrame, out: Path, groups: dict[str, list[tuple]], name: str,
                    reference: str = "current") -> None:
    """Validation: paired gain difference of every variant w.r.t. the `reference` one, per scenario group."""
    whole = at_focus(gains[(gains.phase == "whole") & gains.algorithm.isin(LEARNED) & (gains.variant != UNTUNED)])
    variants = [variant for variant in VARIANT_LABELS if variant in set(whole.variant) and variant != reference]
    fig, axes = style.figure("double", 0.25 * len(variants) + 0.9, ncols=len(groups), sharey=True,
                             layout="constrained", squeeze=False)
    for ax, (title, members) in zip(axes[0], groups.items()):
        rows = pd.concat([scenario(whole, *member) for member in members])
        for offset, algorithm in zip((-0.15, 0.15), LEARNED):
            per_pair = rows[rows.algorithm == algorithm].groupby(["variant", *PAIR]).gain.mean().unstack("variant")
            for index, variant in enumerate(variants):
                difference = (per_pair[variant] - per_pair[reference]).rename("difference").reset_index()
                point = crossed_interval(difference, "difference")
                ax.errorbar(100 * point["mean"], index + offset, xerr=[[100 * (point["mean"] - point["low"])],
                            [100 * (point["high"] - point["mean"])]] if point["complete"] else None,
                            linestyle="none", elinewidth=0.6, **style.line_style(
                                ARCHITECTURES[algorithm], label=ARCHITECTURE_LABELS[algorithm] if index == 0 else None))
        ax.axvline(0, color=style.INK_MUTED, linewidth=0.6)
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"Gain − {VARIANT_LABELS[reference].lower()} (points)")
    axes[0, 0].set_yticks(range(len(variants)), [VARIANT_LABELS[variant] for variant in variants])
    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    style.figure_legend(fig, handles, labels, ncol=2)
    style.save(fig, out, name)
