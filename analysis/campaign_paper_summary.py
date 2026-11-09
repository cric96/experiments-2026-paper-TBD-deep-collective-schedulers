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

from experiments.campaign import BETAS, FIXED, LEARNED, PROGRAMS, STATIC, focus_of
from . import methods, results, style
from .campaign_paper import (LABELS, architecture, betas_of, episode_table, of_scenario, reward_at, tuned)
from .figures import tradeoff
from .figures.campaign import BASELINE_METHODS, CONTROLS, FIXED_PERIODS, UNTUNED, VARIANT_LABELS, untuned_points
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


def summary_methods(finals: dict[int, str], baselines: str = UNTUNED) -> tuple[methods.Method, ...]:
    """The learned methods and the reactive `baselines`, in the order of the summary figures."""
    return (*paper_methods(finals), *(tuned(method, baselines) for method in BASELINE_METHODS.values()))


def per_topology(evaluation: pd.DataFrame, algorithms) -> pd.DataFrame:
    """A schedule that does not learn, one row per evaluation episode: every one has a topology of its own, so only
    the topology varies."""
    rows = evaluation[evaluation.algorithm.isin(algorithms)].groupby(["algorithm", "evaluation_seed", "episode"])
    rows = rows[COLUMNS].mean().reset_index()
    return rows.assign(seed=rows.evaluation_seed * TOPOLOGIES_PER_SEED + rows.episode) \
        .drop(columns=["evaluation_seed", "episode"])


def at_every_beta(rows: pd.DataFrame, betas=BETAS, **columns) -> list[pd.DataFrame]:
    """A schedule that ignores beta, repeated at every beta with the reward of that beta."""
    return [rows.assign(weight=beta, rewardRate=reward_at(rows, beta), **columns) for beta in betas]


def baseline_frames(evaluation: pd.DataFrame, algorithm: int, baselines: str, betas, **columns) -> list[pd.DataFrame]:
    """A reactive baseline at every beta: untuned (the same runs, with the reward of each beta), or tuned at each."""
    rows = evaluation[evaluation.variant == baselines]
    if baselines == UNTUNED:
        return at_every_beta(per_topology(rows, [algorithm]), betas, **columns)
    return [per_topology(rows[rows.beta == beta], [algorithm]).assign(weight=beta, **columns) for beta in betas]


def paper_evaluation(table: pd.DataFrame, training: int, baselines: str = UNTUNED, betas=BETAS) -> pd.DataFrame:
    """One row per (method, beta, seed), as the paper's evaluation table: the learned methods per training seed
    (mean over the evaluation episodes); the fixed periods and the untuned reactive baselines per evaluation episode,
    repeated at every beta with the reward of that beta (the tuned ones at each beta)."""
    evaluation = table[table.episode > training]
    frames = []
    for algorithm in LEARNED:
        means = evaluation[evaluation.algorithm == algorithm].groupby(["beta", "training_seed"])[COLUMNS].mean()
        frames.append(means.reset_index().rename(columns={"beta": "weight", "training_seed": "seed"})
                      .assign(method=KEYS[algorithm], period=np.nan))
    fixed = per_topology(evaluation, FIXED)
    frames += at_every_beta(fixed, betas, method=methods.FIXED.key, period=fixed.algorithm.map(FIXED_PERIODS))
    for algorithm, method in BASELINE_METHODS.items():
        frames += baseline_frames(evaluation, algorithm, baselines, betas, method=method.key, period=np.nan)
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


def summary_figures(table: pd.DataFrame, finals: dict[int, str], training: int, out: Path,
                    baselines: str = UNTUNED) -> None:
    """Metrics against beta, the four metrics at the focus beta of the final GNN and the gains over the fixed-period
    frontier, on the betas of the final methods."""
    learned = summary_methods(finals, baselines)
    betas = betas_of(finals)
    evaluation = paper_evaluation(table, training, baselines, betas)
    data = SimpleNamespace(evaluation=evaluation, methods=learned,
                           gains=results.frontier_gains(evaluation, learned, betas))
    with campaign_labels():
        tradeoff.tradeoff_vs_beta(data, out, betas)
        tradeoff.summary_beta095(data, out, focus_of(finals[LEARNED[0]]))
        tradeoff.frontier_gains_heatmap(data, out, betas)


def overview_entries(finals: dict[int, str], baselines: str = UNTUNED,
                     controls: tuple[str, ...] = CONTROLS) -> list[tuple[str, str, str, float, dict]]:
    """(label, colour, marker, alpha, selection) of every schedule in the overview, as in the paper; the legacy
    objective when it is one of the `controls`."""
    entries = [(f"Fixed {period:g} s", methods.FIXED_PERIOD_GREYS[period], methods.FIXED.marker, 1.0,
                dict(algorithm=next(code for code, value in FIXED_PERIODS.items() if value == period)))
               for period in OVERVIEW_PERIODS]
    entries += [(tuned(method, baselines).label, method.color, method.marker, 1.0,
                 dict(algorithm=algorithm, variant=baselines)) for algorithm, method in BASELINE_METHODS.items()]
    for algorithm in LEARNED:
        method = architecture(algorithm, finals)
        entries.append((method.label, method.color, method.marker, 1.0,
                        dict(algorithm=algorithm, variant=finals[algorithm])))
        if "legacy" in controls and finals[algorithm] != "legacy":
            entries.append((f"{method.label.split(',')[0]}, legacy objective", method.color, method.marker,
                            LEGACY_ALPHA, dict(algorithm=algorithm, variant="legacy")))
    return entries


def pair_values(table: pd.DataFrame, selection: dict, metric: str, training: int) -> np.ndarray:
    """The evaluation mean of `metric` of every (training seed, topology) pair of a schedule at its focus beta, or at
    the `beta` of the selection."""
    rows = table[table.episode > training]
    if "beta" not in selection:
        rows = rows[rows.beta == rows.variant.map(focus_of)]
    for column, value in selection.items():
        rows = rows[rows[column] == value]
    return rows.groupby(["training_seed", "evaluation_seed"])[metric].mean().to_numpy()


def matched_beta(table: pd.DataFrame, algorithm: int, baselines: str, rounds: float, training: int) -> float:
    """The beta of a tuned baseline whose mean rounds (evaluation episodes) are the closest to `rounds`, in ratio."""
    rows = table[(table.episode > training) & (table.algorithm == algorithm) & (table.variant == baselines)]
    return float((np.log(rows.groupby("beta").rounds.mean()) - np.log(rounds)).abs().idxmin())


def matched_betas(tables: list[pd.DataFrame], finals: dict[int, str], baselines: str,
                  training: int) -> dict[int, list[float]]:
    """For each tuned baseline, in each table, the beta at which it runs about as many rounds as the final GNN."""
    gnn = dict(algorithm=LEARNED[0], variant=finals[LEARNED[0]])
    references = [pair_values(table, gnn, "rounds", training).mean() for table in tables]
    return {algorithm: [matched_beta(table, algorithm, baselines, rounds, training)
                        for table, rounds in zip(tables, references)] for algorithm in BASELINE_METHODS}


def matched_label(method: methods.Method, betas: list[float]) -> str:
    low, high = min(betas), max(betas)
    return f"{method.short} at the GNN's rounds (β = {low:g}{'' if low == high else f'–{high:g}'})"


def overview_selection(selection: dict, position: int, betas: dict[int, list[float]], baselines: str) -> dict:
    """The selection of a schedule in the table at `position`: a tuned baseline at its matched beta there."""
    algorithm = selection.get("algorithm")
    matched = selection.get("variant") == baselines and algorithm in betas
    return selection | {"beta": betas[algorithm][position]} if matched else selection


def generalization_overview(home: Path, jobs: list[dict], finals: dict[int, str], training: int, out: Path,
                            baselines: str = UNTUNED, controls: tuple[str, ...] = CONTROLS) -> None:
    """Recovery time, error relative to the 5 s period and rounds on each program (zero-shot but the gradient).
    The baselines tuned by beta are compared at the same energy: in each program, at the beta of their grid whose
    rounds are the closest to those of the final GNN (at the same beta they run many more rounds)."""
    matched = not untuned_points(baselines)
    wanted = [job for job in jobs if (job["beta"] == focus_of(job["variant"]) or
                                      (matched and job["variant"] == baselines))
              and (job["algorithm"] in FIXED or job["variant"] in (*finals.values(), "legacy", baselines))]
    tables = [episode_table(home, of_scenario(wanted, scenario)) for scenario, _ in OVERVIEW_SCENARIOS]
    betas = matched_betas(tables, finals, baselines, training) if matched else {}
    entries = [(matched_label(BASELINE_METHODS[selection["algorithm"]], betas[selection["algorithm"]])
                if overview_selection(selection, 0, betas, baselines) != selection else label, *rest, selection)
               for label, *rest, selection in overview_entries(finals, baselines, controls)]
    metrics = ("recoverySeconds", "relative", "rounds")
    labels = {**LABELS, "relative": f"Error relative to fixed {OVERVIEW_REFERENCE:g} s (%)"}
    width = 0.8 / len(entries)
    reference = dict(algorithm=next(code for code, value in FIXED_PERIODS.items() if value == OVERVIEW_REFERENCE))
    fig, axes = style.figure("double", 2.3, 1, len(metrics))
    for ax, metric in zip(axes, metrics):
        for index, (label, colour, marker, alpha, selection) in enumerate(entries):
            for position, table in enumerate(tables):
                selected = overview_selection(selection, position, betas, baselines)
                if metric == "relative":
                    scale = pair_values(table, reference, "normalizedError", training).mean()
                    values = pair_values(table, selected, "normalizedError", training) / scale * 100.0
                else:
                    values = pair_values(table, selected, metric, training)
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
