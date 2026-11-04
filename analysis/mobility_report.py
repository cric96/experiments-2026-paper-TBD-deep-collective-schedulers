"""Machine-generated summary.md of the static vs. mobile evaluation (``analysis.mobility_static``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .mobility_pareto import FIXED_PERIODS, METHODS
from .mobility_static import (BOOTSTRAP_END_S, MOBILE_DIFFUSION, PHASE_BY_KEY, STATIC_DIFFUSION, TOLERANCE_M,
                              bootstrap_seconds, fixed_reference, switch_reaction)
from .stats import t_interval

ORDER = (0, 1, 9, 8, 2)
FOCUS_BETA = config.FOCUS_WEIGHT
BOOTSTRAP_THRESHOLD_M = 25.0  # "the gradient exists": mean error below 5% of the 500 m scale (the mobile floor is ~10 m)
UNDEFINED = {"converged": "no error left (fixed periods exact)", "outside_range": "outside the fixed range"}


def _interval(values) -> str:
    interval = t_interval(values)
    return f"{interval.mean:.3g} [{interval.low:.3g}, {interval.high:.3g}]"


def _gain(gains: pd.DataFrame, algorithm: int, diffusion: float, phase: str, gain: str = "error",
          beta: float = FOCUS_BETA) -> str:
    rows = gains[(gains.algorithm == algorithm) & (gains.diffusion == diffusion) & (gains.phase == phase)
                 & (gains.gain == gain) & np.isclose(gains.beta, beta)]
    if rows.empty:
        return "—"
    cell = rows.iloc[0]
    if not np.isfinite(cell["mean"]):
        return UNDEFINED.get(cell["undefined"], "—")
    marker = "" if cell["low"] > 0 or cell["high"] < 0 else " (n.s.)"
    return f"{cell['mean']:+.0%} [{cell['low']:+.0%}, {cell['high']:+.0%}]{marker}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(row) + " |" for row in rows)]


def _protocol(metrics: pd.DataFrame) -> list[str]:
    runs = metrics[metrics.phase == "episode"]
    diffusions = ", ".join(f"{value:g}" for value in sorted(runs.diffusion.unique()))
    return [
        "## Data and protocol", "",
        f"- {len(runs)} runs (4 seeds each), evaluation diffusions D = {diffusions} m²/s, tolerance {TOLERANCE_M:g} m;"
        " 20 episodes of 200 s, policies frozen after episode 10; every number averages the 10 evaluation"
        " episodes of a seed, then the seeds (95% Student-t intervals, n = 4).",
        "- Training does not depend on the evaluation D (bit-identical training episodes): mobile-trained"
        " policies cycle D = 0, 0.5, 2 m²/s and tolerance 0, 5, 15 m; static-trained ones train at D = 0."
        " The runs at different D therefore evaluate *the same* policies.",
        f"- Phases: bootstrap 0–{BOOTSTRAP_END_S:g} s (every episode redeploys the nodes without a gradient: the"
        " error of a node without a value is capped at 500 m), tracking"
        f" {BOOTSTRAP_END_S:g}–{config.SOURCE_SWITCH_S:g} s, after the source switch"
        f" {config.SOURCE_SWITCH_S:g}–{config.EPISODE_LENGTH_S:g} s.",
        "- Gains: relative reduction w.r.t. the fixed-period frontier (0.5–5 s, linear interpolation, no"
        " extrapolation) of the same D, seed and phase, at the rounds the policy spent in that phase; paired by"
        " seed. (n.s.) = the 95% interval crosses zero.", "",
    ]


def _cross_table(metrics: pd.DataFrame, gains: pd.DataFrame) -> list[str]:
    focus = metrics[np.isclose(metrics.beta, FOCUS_BETA) & (metrics.phase == "episode")]
    rows = []
    for algorithm in ORDER:
        for diffusion in sorted(focus.diffusion.unique()):
            data = focus[(focus.algorithm == algorithm) & (focus.diffusion == diffusion)]
            if data.empty:
                continue
            rows.append([METHODS[algorithm].label, f"{diffusion:g}", f"{data.rounds.mean():.0f}",
                         f"{data.error.mean():.2f}", f"{data.outside.mean():.1f}",
                         _gain(gains, algorithm, diffusion, "episode"),
                         _gain(gains, algorithm, diffusion, "post_bootstrap"),
                         _gain(gains, algorithm, diffusion, "episode", "outside")])
    header = ["Method", "D", "Rounds", "Error (s)", "Outside tol. (s)", "Error gain, episode",
              f"Error gain, after {BOOTSTRAP_END_S:g} s", "Outside-tol. gain, episode"]
    return [f"## Same policies, static and mobile (beta = {FOCUS_BETA:g}, whole-episode means)", "",
            *_table(header, rows), ""]


def _phase_table(gains: pd.DataFrame) -> list[str]:
    phases = ("episode", "bootstrap", "tracking", "switch")
    lines = [f"## Where the gain comes from (integrated error, beta = {FOCUS_BETA:g})", ""]
    for diffusion in (STATIC_DIFFUSION, MOBILE_DIFFUSION):
        rows = [[METHODS[algorithm].label, *(_gain(gains, algorithm, diffusion, phase) for phase in phases)]
                for algorithm in ORDER]
        lines += [f"D = {diffusion:g} m²/s:", "", *_table(["Method", *(PHASE_BY_KEY[p].label for p in phases)], rows),
                  ""]
    return lines


def _beta_table(metrics: pd.DataFrame) -> list[str]:
    data = metrics[metrics.phase == "episode"]
    low, high = min(config.WEIGHTS), max(config.WEIGHTS)
    rows = []
    for algorithm in ORDER:
        for diffusion in (STATIC_DIFFUSION, MOBILE_DIFFUSION):
            scenario = data[(data.algorithm == algorithm) & (data.diffusion == diffusion)]
            means = scenario.groupby("beta").rounds.mean()
            if not {low, high} <= set(means.index):
                continue
            rows.append([METHODS[algorithm].label, f"{diffusion:g}", f"{means[low]:.0f}", f"{means[high]:.0f}",
                         f"x{means[high] / means[low]:.2f}"])
    header = ["Method", "D", f"Rounds at beta = {low:g}", f"Rounds at beta = {high:g}", "Ratio"]
    return ["## Does beta still steer the energy budget?", "", *_table(header, rows), ""]


def _dynamics_table(series: pd.DataFrame) -> list[str]:
    focus = series[np.isclose(series.beta, FOCUS_BETA)]
    reaction = switch_reaction(focus)
    bootstrap = bootstrap_seconds(focus, BOOTSTRAP_THRESHOLD_M)
    rows = []
    for algorithm in (*ORDER, *sorted(FIXED_PERIODS, key=FIXED_PERIODS.get)):
        label = METHODS[algorithm].label if algorithm in METHODS else f"Fixed {FIXED_PERIODS[algorithm]:g} s"
        for diffusion in (STATIC_DIFFUSION, MOBILE_DIFFUSION):
            ratio = reaction[(reaction.algorithm == algorithm) & (reaction.diffusion == diffusion)].ratio
            seconds = bootstrap[(bootstrap.algorithm == algorithm) & (bootstrap.diffusion == diffusion)].seconds
            if ratio.empty:
                continue
            reached = f"{seconds.notna().sum()}/{len(seconds)}"
            response = _interval(ratio) if algorithm in METHODS else "1 (by construction)"
            rows.append([label, f"{diffusion:g}", response, f"{seconds.mean():.1f} ({reached} seeds)"])
    header = ["Scheduler", "D", "Round rate 0–20 s after / before the switch",
              f"Bootstrap: first time mean error <= {BOOTSTRAP_THRESHOLD_M:g} m (s)"]
    return [f"## Reaction to events (beta = {FOCUS_BETA:g})", "", *_table(header, rows), ""]


def _fixed_table(metrics: pd.DataFrame) -> list[str]:
    """Fixed periods per phase: under mobility the tracking error barely depends on the period."""
    fixed = fixed_reference(metrics).pivot_table(index=["period", "diffusion"], columns="phase",
                                                 values=["rounds", "error"])
    phases = ("episode", "bootstrap", "tracking", "switch")
    rows = [[f"{period:g} s", f"{diffusion:g}", f"{row[('rounds', 'episode')]:.0f}",
             *(f"{row[('error', phase)]:.2f}" for phase in phases)]
            for (period, diffusion), row in fixed.iterrows()]
    header = ["Period", "D", "Rounds", *(f"Error (s), {PHASE_BY_KEY[phase].label}" for phase in phases)]
    return ["## Fixed-period reference (integrated error per phase)", "", *_table(header, rows), ""]


def write_report(metrics: pd.DataFrame, series: pd.DataFrame, gains: pd.DataFrame, path: Path) -> None:
    lines = ["# Static vs. mobile evaluation of the schedulers (generated by analysis.mobility_static)", "",
             *_protocol(metrics), *_cross_table(metrics, gains), *_phase_table(gains), *_beta_table(metrics),
             *_dynamics_table(series), *_fixed_table(metrics)]
    path.write_text("\n".join(lines))
