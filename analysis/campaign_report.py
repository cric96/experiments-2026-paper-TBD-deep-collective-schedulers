"""Markdown tables of a campaign: gains per scenario and variability, from ``summary.csv`` and ``variability.csv``.

python -m analysis.campaign_report build/learning-campaign/test --out charts/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.campaign import FOCUS_BETA, HEURISTIC, LEARNED
from .figures.campaign import ARCHITECTURE_LABELS, PROGRAMS, VARIANT_LABELS, scenario_label

PHASE_ORDER = ("whole", "bootstrap", "quiet", "event", "after")
# Phases where the values change: elsewhere every schedule holds exact values and the gain is undefined.
GAIN_PHASES = ("whole", "bootstrap", "event")


def markdown(frame: pd.DataFrame, index: bool = True, digits: int = 4) -> str:
    """A GitHub markdown table (no extra dependency)."""
    frame = frame.reset_index() if index else frame
    cells = [[f"{value:.{digits}f}" if isinstance(value, float) else str(value) for value in row]
             for row in frame.itertuples(index=False)]
    lines = [list(map(str, frame.columns)), ["---"] * len(frame.columns), *cells]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def interval(row: pd.Series, value: str, scale: float = 100.0, digits: int = 0) -> str:
    mean, low, high = (row[f"{value}_{part}"] * scale for part in ("mean", "low", "high"))
    if pd.isna(low):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def method_label(variant: str, algorithm: int) -> str:
    if algorithm == HEURISTIC:
        return "Heuristic"
    return f"{ARCHITECTURE_LABELS[algorithm]}, {VARIANT_LABELS.get(variant, variant).lower()}"


def gains_table(summary: pd.DataFrame, phase: str = "whole") -> str:
    """Gain (%) at the focus beta, one row per method and one column per scenario."""
    rows = summary[(summary.phase == phase) & (summary.beta == FOCUS_BETA)].copy()
    rows["method"] = [method_label(v, a) for v, a in zip(rows.variant, rows.algorithm)]
    rows["scenario"] = [scenario_label(f, p, n, s) for f, p, n, s in zip(rows.family, rows.program, rows.nodes, rows.speed)]
    rows["cell"] = [interval(row, "gain") for _, row in rows.iterrows()]
    table = rows.pivot_table(index="method", columns="scenario", values="cell", aggfunc="first")
    return markdown(table.fillna(''))


def phase_table(summary: pd.DataFrame, where: dict) -> str:
    """Gain (%) where it is defined, then the rounds per node per second of every phase."""
    rows = summary[(summary.beta == FOCUS_BETA) & summary.algorithm.isin(LEARNED)]
    for column, value in where.items():
        rows = rows[rows[column] == value]
    rows = rows.assign(method=[method_label(v, a) for v, a in zip(rows.variant, rows.algorithm)],
                       gain=[interval(row, "gain") for _, row in rows.iterrows()],
                       rounds=[interval(row, "roundsPerNodeSecond", scale=1.0, digits=2) for _, row in rows.iterrows()])
    gains = rows[rows.phase.isin(GAIN_PHASES)].pivot_table(index="method", columns="phase", values="gain",
                                                          aggfunc="first")
    rounds = rows.pivot_table(index="method", columns="phase", values="rounds", aggfunc="first")
    gains.columns = [f"gain {phase} (%)" for phase in gains.columns]
    rounds.columns = [f"rounds {phase}" for phase in rounds.columns]
    order = [f"gain {phase} (%)" for phase in GAIN_PHASES] + [f"rounds {phase}" for phase in PHASE_ORDER]
    table = gains.join(rounds)
    return markdown(table[[column for column in order if column in table.columns]].fillna(''))


def variability_table(variability: pd.DataFrame) -> str:
    rows = variability[(variability.beta == FOCUS_BETA) & variability.algorithm.isin(LEARNED)
                       & (variability.phase == "whole")]
    rows = rows.assign(method=[method_label(v, a) for v, a in zip(rows.variant, rows.algorithm)],
                       scenario=[scenario_label(f, p, n, s) for f, p, n, s in
                                 zip(rows.family, rows.program, rows.nodes, rows.speed)])
    columns = ["scenario", "method", "mean", "withinRun", "acrossTopologies", "acrossPolicies", "policyCV"]
    return markdown(rows[columns].sort_values(["scenario", "method"]), index=False)


def report(home: Path) -> str:
    summary = pd.read_csv(home / "summary.csv")
    variability = pd.read_csv(home / "variability.csv")
    decision = next((home / name for name in ("selection.json", "confirmation.json") if (home / name).exists()))
    sections = [
        f"# Campaign `{home.name}`",
        "Gains are error reductions (%) w.r.t. the fixed-period frontier at the same rounds, mean "
        "[crossed 95% interval] over (training seed, evaluation seed) pairs, β = 0.95.",
        "## Whole episode", gains_table(summary),
        "## Phases of the static episode (source switch at 100 s)",
        phase_table(summary, dict(family="static")),
        f"## Decision (`{decision.name}`)", "```json\n" + json.dumps(json.loads(decision.read_text()), indent=2)
        + "\n```",
        "## Variability of the normalized error (descriptive)",
        "`withinRun`: across the episodes of a pair; `acrossTopologies`: across evaluation seeds for one policy; "
        "`acrossPolicies`: across training seeds on the same topologies; `policyCV` = acrossPolicies / mean.",
        variability_table(variability),
    ]
    dynamic = summary[summary.family == "dynamic"]
    for speed in sorted(dynamic.speed.unique()):
        sections += [f"## Phases with motion at {speed:g} m/s (100–150 s)",
                     phase_table(summary, dict(family="dynamic", speed=speed))]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(report(args.home))


if __name__ == "__main__":
    main()
