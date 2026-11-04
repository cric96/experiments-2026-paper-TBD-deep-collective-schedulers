"""Illustrative snapshots of the inspected test run: GNN of the final method, first test training seed, evaluation
seed 202 (chosen before the campaign), beta 0.95: attention over time and the use of the action space; the
filmstrips are drawn as in the paper (``campaign_paper``). Descriptive only: attention weights are not a causal
attribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import style
from experiments.campaign import FOCUS_BETA, TRAIN_SEEDS

INSPECTED_EVALUATION_SEED = 202
EMBEDDINGS = "gnn_spatiotemporal_embeddings.csv"
PROGRAMS = {0: "Gradient", 1: "Hop count", 2: "Broadcast", 3: "Collect", 4: "Election"}
# Shortest and longest period of the action space (s): the share of decisions at a bound says whether it binds.
PERIOD_BOUNDS = (0.5, 5.0)


def inspected_runs(home: Path, final: str) -> list[tuple[dict, pd.DataFrame]]:
    manifest = json.loads((home / "manifest.json").read_text())
    selected = [job for job in manifest if job["evaluation_seed"] == INSPECTED_EVALUATION_SEED
                and job["training_seed"] == TRAIN_SEEDS["test"][0] and job["variant"] == final
                and job["algorithm"] == 0 and job["beta"] == FOCUS_BETA and job["nodes"] == 100]
    runs = []
    for job in selected:
        paths = list((home / "exports" / job["key"]).rglob(EMBEDDINGS))
        if len(paths) != 1:
            raise ValueError(f"Missing or ambiguous snapshots for {job['key']}")
        runs.append((job, pd.read_csv(paths[0])))
    return runs


def title(job: dict) -> str:
    if job["family"] == "dynamic":
        return f"Gradient, half of the nodes move at {job['speed']:g} m/s (100–150 s)"
    return f"{PROGRAMS[job['program']]}, {'source switch' if job['program'] != 4 else 'leader removal'} at 100 s"


def attention_figure(runs: list[tuple[dict, pd.DataFrame]], out: Path) -> None:
    fig, axes = style.figure("double", 2.0, ncols=2, layout="constrained")
    for job, frame in runs:
        means = frame.groupby("tEp")[["attEntropy", "maxAlpha"]].mean()
        label = f"Gradient, motion {job['speed']:g} m/s" if job["family"] == "dynamic" else PROGRAMS[job["program"]]
        for ax, column in zip(axes, means.columns):
            ax.plot(means.index, means[column], linewidth=0.9, label=label)
    for ax, ylabel in zip(axes, ("Mean attention entropy", "Mean largest attention weight")):
        ax.axvline(100, color=style.INK_MUTED, linestyle=":", linewidth=0.7)
        ax.set(xlabel="Time in the episode (s)", ylabel=ylabel)
    style.figure_legend(fig, *axes[0].get_legend_handles_labels(), ncol=4)
    style.save(fig, out, "attention_over_time")


def period_saturation(runs: list[tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
    """Share of the decisions at the bounds of the action space, per inspected run."""
    rows = []
    for job, frame in runs:
        periods = frame.nextWakeUp
        rows.append(dict(scenario=title(job), decisions=len(periods),
                         atShortest=float((periods <= PERIOD_BOUNDS[0] + 1e-3).mean()),
                         atLongest=float((periods >= PERIOD_BOUNDS[1] - 1e-3).mean()),
                         median=float(periods.median())))
    return pd.DataFrame(rows)


def draw(home: Path, output: Path, final: str) -> None:
    """Attention over time and the use of the action space; the filmstrips are the paper's (``campaign_paper``)."""
    runs = inspected_runs(home, final)
    if not runs:
        return
    attention_figure(runs, output / "attention")
    period_saturation(runs).to_csv(output / "attention" / "period_saturation.csv", index=False)
