"""The paper's summary figures drawn from the campaign: metrics against beta, the four metrics at beta 0.95, the gains
over the fixed-period frontier for every beta, and the overview of the zero-shot programs.

The first three call the paper's own functions on the campaign laid out as the paper's evaluation table: the learned
methods per training seed, the fixed periods per evaluation topology. Two of the paper's metrics have a campaign
counterpart under another name, so their labels are replaced while drawing: the unstable time by the recovery time
after the event, the integrated error by the mean absolute error.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.campaign import BETAS, FIXED, FOCUS_BETA, HEURISTIC, LEARNED, PROGRAMS, STATIC
from . import methods, results, style
from .campaign_paper import (LABELS, architecture, episode_table, of_scenario, reward_at)
from .figures import tradeoff
from .figures.campaign import FIXED_PERIODS, VARIANT_LABELS
from .figures.common import METRIC_LABELS

KEYS = {0: methods.GNN.key, 1: methods.TD3.key}
COLUMNS = ["rounds", "absoluteError", "recoverySeconds", "rewardRate", "normalizedError", "roundsPerNodeSecond"]
PAPER_NAMES = {"rounds": "ticks", "absoluteError": "error", "recoverySeconds": "unstable", "rewardRate": "reward"}
OVERVIEW_SCENARIOS = [(STATIC, "Gradient"), *zip(PROGRAMS, ("Hop count", "Broadcast", "Collect", "Election"))]
OVERVIEW_REFERENCE = 5.0
OVERVIEW_PERIODS = (5.0, 3.0)
# Distinct identifiers of the evaluation topologies: an evaluation seed holds one per evaluation episode.
TOPOLOGIES_PER_SEED = 100
# The legacy controls are drawn lighter than the final variants.
LEGACY_ALPHA = 0.45


def paper_methods(finals: dict[int, str]) -> tuple[methods.Method, ...]:
    return tuple(dataclasses.replace(architecture(algorithm, finals), key=KEYS[algorithm],
                                     short=f"{('GNN', 'MLP')[algorithm]}, {VARIANT_LABELS[finals[algorithm]].lower()}")
                 for algorithm in LEARNED)


def paper_evaluation(table: pd.DataFrame, training: int) -> pd.DataFrame:
    """One row per (method, beta, seed), as the paper's evaluation table: the learned methods per training seed
    (mean over the evaluation episodes), the fixed periods per evaluation episode (every one has a topology of its
    own; they do not learn, so only the topology varies), repeated at every beta with the reward of that beta."""
    evaluation = table[table.episode > training]
    frames = []
    for algorithm in LEARNED:
        means = evaluation[evaluation.algorithm == algorithm].groupby(["beta", "training_seed"])[COLUMNS].mean()
        frames.append(means.reset_index().rename(columns={"beta": "weight", "training_seed": "seed"})
                      .assign(method=KEYS[algorithm], period=np.nan))
    fixed = evaluation[evaluation.algorithm.isin(FIXED)].groupby(["algorithm", "evaluation_seed", "episode"])[COLUMNS]
    fixed = fixed.mean().reset_index()
    fixed = fixed.assign(seed=fixed.evaluation_seed * TOPOLOGIES_PER_SEED + fixed.episode)
    fixed = fixed.drop(columns=["evaluation_seed", "episode"])
    for beta in BETAS:
        frames.append(fixed.assign(method=methods.FIXED.key, weight=beta, period=fixed.algorithm.map(FIXED_PERIODS),
                                   rewardRate=reward_at(fixed, beta)))
    return pd.concat(frames, ignore_index=True).drop(columns="algorithm").rename(columns=PAPER_NAMES)


@contextmanager
def campaign_labels():
    """The campaign's names of the paper's metrics, while the paper's functions draw."""
    saved = dict(METRIC_LABELS), dict(results.FRONTIER_GAINS)
    METRIC_LABELS.update(unstable=LABELS["recoverySeconds"], error=LABELS["absoluteError"],
                         reward=LABELS["rewardRate"])
    results.FRONTIER_GAINS.update(unstable_at_ticks="recovery-time reduction at equal rounds")
    try:
        yield
    finally:
        METRIC_LABELS.clear(), METRIC_LABELS.update(saved[0])
        results.FRONTIER_GAINS.clear(), results.FRONTIER_GAINS.update(saved[1])


def summary_figures(table: pd.DataFrame, finals: dict[int, str], training: int, out: Path) -> None:
    learned = paper_methods(finals)
    evaluation = paper_evaluation(table, training)
    data = SimpleNamespace(evaluation=evaluation, methods=learned, gains=results.frontier_gains(evaluation, learned))
    with campaign_labels():
        tradeoff.tradeoff_vs_beta(data, out)
        tradeoff.summary_beta095(data, out)
        tradeoff.frontier_gains_heatmap(data, out)


def overview_entries(finals: dict[int, str]) -> list[tuple[str, str, str, float, dict]]:
    """(label, colour, marker, alpha, selection) of every schedule in the overview, as in the paper."""
    entries = [(f"Fixed {period:g} s", methods.FIXED_PERIOD_GREYS[period], methods.FIXED.marker, 1.0,
                dict(algorithm=next(code for code, value in FIXED_PERIODS.items() if value == period)))
               for period in OVERVIEW_PERIODS]
    entries.append((methods.HEURISTIC.label, methods.HEURISTIC.color, methods.HEURISTIC.marker, 1.0,
                    dict(algorithm=HEURISTIC)))
    for algorithm in LEARNED:
        method = architecture(algorithm, finals)
        entries.append((method.label, method.color, method.marker, 1.0,
                        dict(algorithm=algorithm, variant=finals[algorithm])))
        entries.append((f"{method.label.split(',')[0]}, legacy objective", method.color, method.marker,
                        LEGACY_ALPHA, dict(algorithm=algorithm, variant="legacy")))
    return entries


def pair_values(table: pd.DataFrame, selection: dict, metric: str, training: int) -> np.ndarray:
    """The evaluation mean of `metric` of every (training seed, topology) pair of a schedule at the focus beta."""
    rows = table[(table.episode > training) & (table.beta == FOCUS_BETA)]
    for column, value in selection.items():
        rows = rows[rows[column] == value]
    return rows.groupby(["training_seed", "evaluation_seed"])[metric].mean().to_numpy()


def generalization_overview(home: Path, jobs: list[dict], finals: dict[int, str], training: int, out: Path) -> None:
    """Recovery time, error relative to the 5 s period and rounds on each program (zero-shot but the gradient)."""
    wanted = [job for job in jobs if job["beta"] == FOCUS_BETA and (job["algorithm"] in (*FIXED, HEURISTIC)
                                                                      or job["variant"] in (*finals.values(), "legacy"))]
    tables = [episode_table(home, of_scenario(wanted, scenario)) for scenario, _ in OVERVIEW_SCENARIOS]
    entries = overview_entries(finals)
    metrics = ("recoverySeconds", "relative", "rounds")
    labels = {**LABELS, "relative": f"Error relative to fixed {OVERVIEW_REFERENCE:g} s (%)"}
    width = 0.8 / len(entries)
    reference = dict(algorithm=next(code for code, value in FIXED_PERIODS.items() if value == OVERVIEW_REFERENCE))
    fig, axes = style.figure("double", 2.3, 1, len(metrics))
    for ax, metric in zip(axes, metrics):
        for index, (label, colour, marker, alpha, selection) in enumerate(entries):
            for position, table in enumerate(tables):
                if metric == "relative":
                    scale = pair_values(table, reference, "normalizedError", training).mean()
                    values = pair_values(table, selection, "normalizedError", training) / scale * 100.0
                else:
                    values = pair_values(table, selection, metric, training)
                x = position + (index - (len(entries) - 1) / 2) * width
                ax.plot([x, x], [values.min(), values.max()], color=colour, linewidth=0.6, alpha=alpha)
                ax.plot(x, values.mean(), marker=marker, color=colour, alpha=alpha, linestyle="none",
                        markersize=3.6 * (1.3 if marker == "*" else 1), markeredgecolor="white",
                        markeredgewidth=0.3, label=label if position == 0 else None)
        ax.set_xticks(range(len(tables)), [title for _, title in OVERVIEW_SCENARIOS], rotation=25, ha="right")
        ax.set_ylabel(labels[metric])
        ax.grid(axis="x", visible=False)
        if metric == "rounds":
            ax.yaxis.set_major_formatter(lambda value, _: f"{value / 1000:g}k")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncol=4)
    style.save(fig, out, "generalization_overview")
