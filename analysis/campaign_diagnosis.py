"""Diagnosis of the GNN transfer: `current` with one setting of the original experiment restored at a time.

python -m analysis.campaign_diagnosis build/learning-campaign/validation+build/learning-campaign/validation-diagnosis \
    build/learning-campaign/validation-retention --out charts/generalization

Every argument is a group of campaign folders analysed together ('+'-separated): each group holds the fixed-period
controls of its own network (the retention changes the network, so its variants come with their controls). Every gain
is measured on the frontier of its group; the change w.r.t. `current` pairs the same (training seed, evaluation seed)
of the validation split, so it compares gains over the respective frontiers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.campaign import FOCUS_BETA, PROGRAMS, STATIC
from . import style
from .campaign import PAIR, crossed_interval, gains, load
from .campaign_report import markdown
from .figures.campaign import HeatmapInkThreshold, VARIANT_LABELS

GNN = 0
CONTROL = "current"
ORDER = ("current", "legacy", "error-cost", "wide-loss-scale", "long-retention", "paper-settings")
COLUMNS = [(STATIC, "Gradient"), *zip(PROGRAMS, ("Hop count", "Broadcast", "Collect", "Election"))]


def gnn_gains(groups: list[list[Path]]) -> pd.DataFrame:
    """Whole-episode gains of the GNN at the focus beta, at 100 nodes, of every group."""
    frames = [gains(load(*homes)) for homes in groups]
    data = pd.concat(frames, ignore_index=True)
    return data[(data.algorithm == GNN) & (data.beta == FOCUS_BETA) & (data.phase == "whole") & (data.nodes == 100)]


def scenario_rows(data: pd.DataFrame, scenario: tuple) -> pd.DataFrame:
    family, program, nodes, speed = scenario
    return data[(data.family == family) & (data.program == program) & (data.speed == speed)]


def summary(data: pd.DataFrame) -> pd.DataFrame:
    """Per variant and scenario: gain, change of the gain w.r.t. the control on the same pairs, rounds."""
    records = []
    for scenario, title in COLUMNS:
        rows = scenario_rows(data, scenario)
        control = rows[rows.variant == CONTROL].set_index(PAIR).gain
        for variant in [variant for variant in ORDER if variant in set(rows.variant)]:
            values = rows[rows.variant == variant]
            paired = (values.set_index(PAIR).gain - control).rename("change").dropna().reset_index()
            change = crossed_interval(paired, "change") if variant != CONTROL else dict(mean=0., low=0., high=0.)
            gain = crossed_interval(values, "gain")
            records.append(dict(variant=variant, scenario=title, gain=gain["mean"], gain_low=gain["low"],
                                gain_high=gain["high"], change=change["mean"], change_low=change["low"],
                                change_high=change["high"], rounds=values.roundsPerNodeSecond.mean()))
    return pd.DataFrame(records)


def cell(mean: float, low: float, high: float) -> str:
    return f"{100 * mean:.0f} [{100 * low:.0f}, {100 * high:.0f}]"


def report(table: pd.DataFrame) -> str:
    table = table.assign(label=[VARIANT_LABELS.get(variant, variant) for variant in table.variant])
    sections = []
    for name, columns in (("Gain over the fixed-period frontier (%)", ("gain", "gain_low", "gain_high")),
                          ("Change of the gain w.r.t. current (points)", ("change", "change_low", "change_high"))):
        frame = table.assign(value=[cell(*row) for row in table[list(columns)].itertuples(index=False)])
        sections += [f"## {name}", markdown(frame.pivot_table(index="label", columns="scenario", values="value",
                                                              aggfunc="first", sort=False))]
    rounds = table.pivot_table(index="label", columns="scenario", values="rounds", aggfunc="first", sort=False)
    sections += ["## Rounds per node per second", markdown(rounds, digits=2)]
    return "# GNN transfer diagnosis (validation seeds, β = 0.95, 100 nodes)\n\n" + "\n\n".join(sections) + "\n"


def figure(table: pd.DataFrame, out: Path) -> None:
    """Gains of every variant on every scenario, one shared colour scale."""
    variants = [variant for variant in ORDER if variant in set(table.variant)]
    scenarios = [title for _, title in COLUMNS]
    grid = table.pivot_table(index="variant", columns="scenario", values="gain").reindex(index=variants,
                                                                                         columns=scenarios) * 100
    limit = np.nanmax(np.abs(grid.to_numpy()))
    fig, ax = style.figure("double", 2.4, layout="constrained")
    image = ax.imshow(grid.to_numpy(), cmap=style.DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    lookup = table.set_index(["variant", "scenario"])
    for (row, variant), (column, scenario) in [(r, c) for r in enumerate(variants) for c in enumerate(scenarios)]:
        if (variant, scenario) in lookup.index:
            point = lookup.loc[(variant, scenario)]
            uncertain = point.gain_low <= 0 <= point.gain_high
            value = grid.loc[variant, scenario]
            ax.text(column, row, f"{value:.0f}{'*' if uncertain else ''}\n{point.rounds:.2f}/s", ha="center",
                    va="center", fontsize=6.5, color="white" if abs(value) > limit * HeatmapInkThreshold else style.INK)
    ax.set_xticks(range(len(scenarios)), scenarios)
    ax.set_yticks(range(len(variants)), [VARIANT_LABELS.get(variant, variant) for variant in variants])
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Error reduction at matched rounds (%)\n* 95% interval includes 0; "
                                     "below: rounds per node per second", shrink=0.9)
    ax.set_title("GNN, validation seeds, β = 0.95: settings of the original experiment restored in current", loc="left")
    style.save(fig, out, "transfer_diagnosis")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="+", help="'+'-separated campaign folders analysed together")
    parser.add_argument("--out", type=Path, default=Path("charts/generalization"))
    args = parser.parse_args()
    style.apply()
    table = summary(gnn_gains([[Path(home) for home in group.split("+")] for group in args.groups]))
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "transfer_diagnosis.csv", index=False)
    (args.out / "transfer_diagnosis.md").write_text(report(table))
    figure(table, args.out)


if __name__ == "__main__":
    main()
