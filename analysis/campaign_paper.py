"""The figures of the paper (``plotter.py``) drawn from a campaign, with the same look: learning curves of the reward
and of the metrics, Pareto fronts swept over beta, the effect of beta within the episode, and the GNN attention
filmstrips and latent dynamics of the inspected run.

python -m analysis.campaign_paper build/learning-campaign/test --out charts

The learned methods are the final variant of each architecture (the one of its beta study). Bands and whiskers are 95%
t-intervals over the training seeds, after averaging the evaluation topologies of each seed and the evaluation
episodes of each run. The fixed periods and the reactive baselines do not learn: they are means over the topologies.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from experiments.campaign import (BASELINES, DYNAMIC, FIXED, LEARNED, PROGRAMS, STATIC, TRAIN_SEEDS, focus_of,
                                  grid_of)
from . import config, methods, style
from .campaign import EPISODE_LENGTH
from .campaign_snapshots import EMBEDDINGS, INSPECTED_EVALUATION_SEED, SNAPSHOT_BETA
from .figures import gnn, gnn_programs
from .figures.campaign import (BASELINE_METHODS, CONTROLS, FIXED_PERIODS, UNTUNED, VARIANT_LABELS, baseline_label,
                               reactive_variant, untuned_points)
from .figures.common import mean_band, method_band, thousands
from .figures.tradeoff import _period_axis
from .pareto import frontier_value_at
from .stats import t_interval

NODES = 100
EPISODES = np.arange(1, 21)
MARK_EVERY = 3
# Rounds per episode of a node population running one round per node per second.
ROUNDS_PER_RATE = NODES * EPISODE_LENGTH
# The betas of the reward and of the metric curves, when the grid of the final methods has them (else its own).
REWARD_BETAS = (0.5, 0.8, 0.95, 1.0)
METRIC_BETAS = (0.95, 0.8)
REFERENCE_PERIODS = (1.0, 3.0, 5.0)
PARETO_TICKS = (4000, 5000, 7000, 10000, 15000, 20000, 40000)
# Legend entries of the Pareto figures that fit in one row; more go on two.
PARETO_LEGEND_ROW = 5
# Rounds per episode of the longest and of the shortest period: no schedule leaves this range.
ROUNDS_RANGE = (ROUNDS_PER_RATE / 5.0, ROUNDS_PER_RATE / 0.5)
# Margin of the rounds axis around that range (log scale).
ROUNDS_MARGIN = 1.12
# Windows of the effective period: nodes wake up nearly in step, so a 2 s window aliases periods of 3-5 s; the rate is
# measured over this many consecutive windows (10 s, twice the longest period).
PERIOD_WINDOWS = 5
# The first window holds the start-up transient (no value yet anywhere), as in the paper's figure.
FIRST_WINDOW_S = 4.0
EVENT_S = 100.0
MOTION_END_S = 150.0
# Lightest and darkest shade of the GNN's ramp of betas.
GNN_RAMP = (0.35, 0.9)
# The reactive baselines do not learn: thin dash-dotted curves in their own colours.
BASELINE_LINE = dict(linestyle="-.", linewidth=0.9)
# The paper's fixed periods start at 1 s; the campaign's 0.5 s period is drawn lighter than all of them.
HALF_SECOND_GREY = "#cfcfcf"
LABELS = {
    "rewardRate": "Reward per node and second",
    "rounds": "Rounds per episode",
    "absoluteError": "Mean absolute error (m)",
    "recoverySeconds": "Recovery after the event (s)",
    "normalizedError": "Normalized error",
}
EPISODE_COLUMNS = ["roundsPerNodeSecond", "normalizedError", "absoluteError", "recoverySeconds", "rewardRate"]
PROGRAM_PANELS = [(STATIC, "Gradient, source switch"), (DYNAMIC[0], "Gradient, half of the nodes move at 1 m/s"),
                  *[(scenario, title) for scenario, title in
                    zip(PROGRAMS, ("Hop count (zero-shot)", "Broadcast (zero-shot)", "Collect (zero-shot)",
                                   "Election (zero-shot)"))]]


def architecture(algorithm: int, finals: dict[int, str]) -> methods.Method:
    """The paper's style of an architecture, labelled with its final variant."""
    base = {0: methods.GNN, 1: methods.TD3}[algorithm]
    return dataclasses.replace(base, label=f"{base.label}, {VARIANT_LABELS[finals[algorithm]].lower()}")


def of_scenario(jobs: list[dict], scenario: tuple) -> list[dict]:
    family, program, nodes, speed = scenario
    return [job for job in jobs if (job["family"], job["program"], job["nodes"], job["speed"])
            == (family, program, nodes, speed)]


def beta_colors(algorithm: int, betas) -> dict[float, tuple]:
    """Ordinal ramp of the betas, light to dark: the GNN in its own hue, the MLP in the paper's orange ramp."""
    if algorithm == LEARNED[0]:
        return dict(zip(betas, mpl.colormaps["RdPu"](np.linspace(*GNN_RAMP, len(betas)))))
    return dict(zip(betas, style.BETA_COLORS.values()))


def betas_of(finals: dict[int, str]) -> list[float]:
    """The betas of the final methods (those of the final GNN)."""
    return grid_of(finals[LEARNED[0]])[0]


def relevant(job: dict, finals: dict[int, str], baselines: str = UNTUNED) -> bool:
    """The final variant of each architecture, the fixed periods, the untuned reactive baselines and the `baselines`
    compared with the final methods."""
    return (job["algorithm"] in LEARNED and job["variant"] == finals[job["algorithm"]]) or \
        job["algorithm"] in FIXED or (job["algorithm"] in BASELINES and job["variant"] in (UNTUNED, baselines))


def baseline_rows(table: pd.DataFrame, algorithm: int, baselines: str, beta: float) -> pd.DataFrame:
    """A reactive baseline: untuned (it ignores beta), or tuned at `beta`."""
    rows = table[(table.algorithm == algorithm) & (table.variant == baselines)]
    return rows if baselines == UNTUNED else rows[rows.beta == beta]


def censored_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    """An episode that does not recover before its end (the 5 s period, sometimes) recovers at least in the time left
    after the event: a lower bound, so the gains over the fixed periods stay conservative."""
    censored = frame.recoveryCensored > 0
    return frame.assign(recoverySeconds=frame.recoverySeconds.mask(censored, EPISODE_LENGTH - EVENT_S))


def episode_shift(job: dict) -> int:
    """How many training episodes a run has beyond the paper's (config.TRAINING_EPISODES): its episodes are drawn that
    many places earlier, so that the evaluation of every run starts at the same episode."""
    return job["settings"]["trainingEpisodes"] - config.TRAINING_EPISODES


def episode_table(home: Path, jobs: list[dict]) -> pd.DataFrame:
    """One row per (run, episode), episodes counted from 1 and aligned on the start of the evaluation (a run that
    trains longer than the paper's protocol shows only its last training episodes, [[episode_shift]]), the censored
    recovery times bounded."""
    rows = []
    for job in jobs:
        frame = censored_recovery(pd.read_csv(next((home / "exports" / job["key"]).glob("*_episodes.csv"))))
        description = {key: value for key, value in job.items() if key != "settings"}
        rows.append(frame[["episode", *EPISODE_COLUMNS]].assign(**description)
                    .assign(episode=frame.episode - episode_shift(job)))
    table = pd.concat(rows, ignore_index=True)
    return table.assign(episode=table.episode + 1, rounds=table.roundsPerNodeSecond * ROUNDS_PER_RATE)


def motion_episodes(job: dict) -> list[int]:
    """The training episodes (from 1, aligned as in [[episode_table]]) where a region moves, from the cyclic schedule
    of training events."""
    events = job["settings"]["trainingEvents"].split(",")
    return [episode + 1 - episode_shift(job) for episode in range(job["settings"]["trainingEpisodes"])
            if events[episode % len(events)] == "RegionStartsMoving" and episode + 1 > episode_shift(job)]


def learned(table: pd.DataFrame, algorithm: int, beta: float) -> pd.DataFrame:
    return table[(table.algorithm == algorithm) & (table.beta == beta)]


def fixed(table: pd.DataFrame, period: float) -> pd.DataFrame:
    algorithm = next(code for code, value in FIXED_PERIODS.items() if value == period)
    return table[table.algorithm == algorithm]


def reward_at(rows: pd.DataFrame, beta: float) -> pd.Series:
    """The exported reward rate of a schedule at any beta: it does not depend on beta for a fixed period."""
    return -beta * rows.normalizedError - (1.0 - beta) * rows.roundsPerNodeSecond


def seed_matrix(rows: pd.DataFrame, metric: str) -> np.ndarray:
    """Training seeds x episodes, after averaging the evaluation topologies."""
    means = rows.groupby(["training_seed", "episode"])[metric].mean().unstack("episode")
    return means.reindex(columns=EPISODES).to_numpy()


def period_grey(period: float) -> str:
    return methods.FIXED_PERIOD_GREYS.get(period, HALF_SECOND_GREY)


def fixed_line(ax, rows: pd.DataFrame, metric: str, period: float) -> None:
    """A fixed period's mean curve (over the topologies) in grey, labelled at the right edge."""
    mean = rows.groupby("episode")[metric].mean().reindex(EPISODES).to_numpy()
    ax.plot(EPISODES, mean, color=period_grey(period), linestyle="--", linewidth=0.8, zorder=1)
    style.direct_label(ax, EPISODES[-1], mean[-1], f"{period:g} s", dx=2, color=style.INK_SECONDARY, fontsize=5.5)


def mark_motion(ax, episodes: list[int]) -> None:
    for episode in episodes:
        ax.axvline(episode, color=style.INK_MUTED, linestyle=":", linewidth=0.6, zorder=0)


def episode_axis(ax) -> None:
    ax.set_xlim(0.5, EPISODES[-1] + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20])


def baseline_line(ax, rows: pd.DataFrame, metric: str, method: methods.Method) -> None:
    """A reactive baseline's mean curve over the topologies, in its colour."""
    mean = rows.groupby("episode")[metric].mean().reindex(EPISODES).to_numpy()
    ax.plot(EPISODES, mean, zorder=2, **style.line_style(method, label=None, markevery=MARK_EVERY, **BASELINE_LINE))


def tuned(method: methods.Method, baselines: str) -> methods.Method:
    """A reactive baseline labelled (in full and in short) as tuned by beta, when it is."""
    if baselines == UNTUNED:
        return method
    label = baseline_label(method, baselines)
    return dataclasses.replace(method, label=label, short=label)


def baseline_handles(baselines: str = UNTUNED) -> list[Line2D]:
    return [style.legend_handle(tuned(method, baselines), **BASELINE_LINE) for method in BASELINE_METHODS.values()]


def curve_handles(finals: dict[int, str], fixed_label: str, baselines: str = UNTUNED) -> list[Line2D]:
    handles = [style.legend_handle(architecture(algorithm, finals)) for algorithm in LEARNED]
    handles += baseline_handles(baselines)
    handles.append(Line2D([], [], color=style.fixed_period_color(3.0), linestyle="--", linewidth=0.8,
                          label=fixed_label))
    handles.append(Line2D([], [], color=style.INK_MUTED, linestyle=":", linewidth=0.6,
                          label="training episode with motion"))
    return handles


def curve_legend(fig, finals: dict[int, str], fixed_label: str, baselines: str = UNTUNED) -> None:
    handles = curve_handles(finals, fixed_label, baselines)
    style.figure_legend(fig, handles, ncol=(len(handles) + 1) // 2)


def reward_betas(betas) -> tuple[float, ...]:
    return REWARD_BETAS if set(REWARD_BETAS) <= set(betas) else tuple(betas)


def metric_betas(betas, focus: float) -> tuple[float, ...]:
    return tuple(beta for beta in METRIC_BETAS if beta in betas) or (focus,)


def learning_curves_reward(table: pd.DataFrame, finals: dict[int, str], motion: list[int], training: int,
                           out: Path, baselines: str = UNTUNED) -> None:
    """Reward per episode for four betas (or for every beta of another grid), against the best fixed period of each
    beta."""
    shown = reward_betas(betas_of(finals))
    fig, axes = style.figure("double", 1.95, 1, len(shown))
    for index, (ax, beta) in enumerate(zip(axes, shown)):
        style.shade_evaluation(ax, label=index == 0)
        mark_motion(ax, motion)
        for algorithm in LEARNED:
            method_band(ax, EPISODES, seed_matrix(learned(table, algorithm, beta), "rewardRate"),
                        architecture(algorithm, finals), markevery=MARK_EVERY)
        rewards = {period: fixed(table, period).assign(reward=lambda rows: reward_at(rows, beta))
                   for period in FIXED_PERIODS.values()}
        best = max(rewards, key=lambda period: rewards[period][rewards[period].episode > training].reward.mean())
        fixed_line(ax, rewards[best], "reward", best)
        for algorithm, method in BASELINE_METHODS.items():
            rows = baseline_rows(table, algorithm, baselines, beta)
            baseline_line(ax, rows.assign(reward=reward_at(rows, beta)), "reward", method)
        ax.set_title(f"$\\beta$ = {beta:g}")
        ax.set_xlabel("Episode")
        episode_axis(ax)
    axes[0].set_ylabel(LABELS["rewardRate"])
    curve_legend(fig, finals, "Best fixed period", baselines)
    style.save(fig, out, "learning_curves_reward")


def learning_curves_metrics(table: pd.DataFrame, finals: dict[int, str], motion: list[int], out: Path,
                            baselines: str = UNTUNED) -> None:
    """Rounds, error and recovery time per episode (two betas), against fixed periods."""
    metrics = ("rounds", "absoluteError", "recoverySeconds")
    shown = metric_betas(betas_of(finals), focus_of(finals[LEARNED[0]]))
    fig, axes = style.figure("double", 3.5, len(shown), len(metrics), sharex=True, squeeze=False)
    for row, beta in enumerate(shown):
        for column, metric in enumerate(metrics):
            ax = axes[row, column]
            style.shade_evaluation(ax, label=row == 0 and column == 0)
            mark_motion(ax, motion)
            for algorithm in LEARNED:
                method_band(ax, EPISODES, seed_matrix(learned(table, algorithm, beta), metric),
                            architecture(algorithm, finals), markevery=MARK_EVERY)
            for period in REFERENCE_PERIODS:
                fixed_line(ax, fixed(table, period), metric, period)
            for algorithm, method in BASELINE_METHODS.items():
                baseline_line(ax, baseline_rows(table, algorithm, baselines, beta), metric, method)
            ax.set_title(f"$\\beta$ = {beta:g}")
            ax.set_ylabel(LABELS[metric])
            if metric == "rounds":
                thousands(ax, "y")
            episode_axis(ax)
            if row == len(shown) - 1:
                ax.set_xlabel("Episode")
    curve_legend(fig, finals, "Fixed period (1, 3, 5 s)", baselines)
    style.save(fig, out, "learning_curves_metrics")


def evaluation_point(rows: pd.DataFrame, metric: str, training: int):
    """Mean and 95% t-interval over the training seeds of the evaluation mean of `metric`."""
    return t_interval(rows[rows.episode > training].groupby("training_seed")[metric].mean().to_numpy())


def gain_callout(ax, rounds: float, value: float, reference: float) -> None:
    """Arrow from the fixed-period frontier to a method at the same rounds, labelled with the relative change."""
    change = value / reference - 1.0
    ax.annotate("", xy=(rounds, value), xytext=(rounds, reference),
                arrowprops={"arrowstyle": "->", "color": style.INK_SECONDARY, "linewidth": 0.7})
    sign = "+" if change > 0 else "\N{MINUS SIGN}"
    style.direct_label(ax, rounds, (value + reference) / 2, f"{sign}{abs(change):.0%}", dx=3, color=style.INK,
                       fontsize=6.5)


def rounds_whiskers(interval) -> list[list[float]]:
    """The interval of the rounds, clipped to the range that a schedule can reach; a mean just beyond the range (a node
    that sleeps 5 s rounds a little less often in an episode) has no whisker on that side."""
    low, high = max(interval.low, ROUNDS_RANGE[0]), min(interval.high, ROUNDS_RANGE[1])
    return [[max(interval.mean - low, 0.0)], [max(high - interval.mean, 0.0)]]


def pareto_axes(ax, metric: str) -> None:
    ax.set_xscale("log")
    ax.set_xlim(ROUNDS_RANGE[0] / ROUNDS_MARGIN, ROUNDS_RANGE[1] * ROUNDS_MARGIN)
    ax.set_xticks(PARETO_TICKS, [f"{tick / 1000:g}k" for tick in PARETO_TICKS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Rounds per episode (energy proxy, log scale)")
    ax.set_ylabel(LABELS[metric])
    ax.margins(y=0.1)
    _period_axis(ax)


def tuned_curve(ax, evaluation: pd.DataFrame, algorithm: int, baselines: str, metric: str,
                method: methods.Method) -> None:
    """A reactive baseline tuned by beta: its mean at every beta of its grid, joined by a dash-dotted line."""
    rows = evaluation[(evaluation.algorithm == algorithm) & (evaluation.variant == baselines)]
    means = rows.groupby("beta")[["rounds", metric]].mean().reindex(grid_of(baselines)[0])
    ax.plot(means.rounds, means[metric], zorder=1, **style.line_style(method, label=None, **BASELINE_LINE))


def pareto_panel(ax, table: pd.DataFrame, metric: str, finals: dict[int, str], training: int,
                 baselines: str = UNTUNED) -> None:
    """Fixed-period frontier, reactive baselines (tuned by beta when the campaign has them, untuned otherwise: one
    point each, they ignore beta) and the learned beta sweeps; the gain of the final GNN at the focus beta."""
    evaluation = table[table.episode > training]
    frontier = evaluation[evaluation.algorithm.isin(FIXED)].groupby("algorithm")[["rounds", metric]].mean()
    frontier = frontier.assign(period=[FIXED_PERIODS[code] for code in frontier.index]).sort_values("rounds")
    ax.plot(frontier.rounds, frontier[metric], **style.fixed_period_style())
    style.label_fixed_periods(ax, frontier.rounds, frontier[metric], frontier.period)
    for algorithm, method in BASELINE_METHODS.items():
        if untuned_points(baselines):
            baseline = baseline_rows(evaluation, algorithm, UNTUNED, focus_of(UNTUNED))[["rounds", metric]].mean()
            ax.plot(baseline.rounds, baseline[metric], linestyle="none", **style.line_style(method, label=None))
        else:
            tuned_curve(ax, evaluation, algorithm, baselines, metric, method)
    for algorithm in LEARNED:
        method = architecture(algorithm, finals)
        points = []
        for beta in grid_of(finals[algorithm])[0]:
            rows = learned(table, algorithm, beta)
            x, y = evaluation_point(rows, "rounds", training), evaluation_point(rows, metric, training)
            ax.errorbar(x.mean, y.mean, xerr=rounds_whiskers(x),
                        yerr=[[y.mean - y.low], [y.high - y.mean]],
                        **style.line_style(method, linestyle="none", elinewidth=0.6, label=None))
            points.append((x.mean, y.mean))
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=method.color, linewidth=0.7, alpha=0.8, zorder=1)
    focus = learned(table, LEARNED[0], focus_of(finals[LEARNED[0]]))
    rounds, value = (evaluation_point(focus, column, training).mean for column in ("rounds", metric))
    reference = frontier_value_at(frontier.rounds, frontier[metric], rounds)
    gain_callout(ax, rounds, value, reference)
    swept = "learned" if baselines == UNTUNED else "learned and tuned"
    betas = ", ".join(f"{beta:g}" for beta in betas_of(finals))
    # In the top right corner, which a trade-off front leaves empty (few rounds go with a large error).
    ax.text(0.98, 0.97, f"{swept}, from left to right:\n$\\beta$ = {betas}", transform=ax.transAxes,
            fontsize=6, color=style.INK_MUTED, ha="right", va="top")
    pareto_axes(ax, metric)


def pareto_legend(fig, finals: dict[int, str], baselines: str = UNTUNED) -> None:
    handles = [style.legend_handle(architecture(algorithm, finals), linestyle="-", linewidth=0.7)
               for algorithm in LEARNED]
    handles += [style.legend_handle(method, linestyle="none") for method in BASELINE_METHODS.values()] \
        if untuned_points(baselines) else baseline_handles(baselines)
    handles.append(Line2D([], [], **style.fixed_period_style(label="Fixed period (0.5–5 s)")))
    style.figure_legend(fig, handles, ncol=len(handles) if len(handles) <= PARETO_LEGEND_ROW
                        else (len(handles) + 1) // 2)


def pareto_frontiers(table: pd.DataFrame, finals: dict[int, str], training: int, out: Path,
                     baselines: str = UNTUNED) -> None:
    """Rounds vs. error and vs. recovery time on the static gradient, like the paper's figure."""
    fig, axes = style.figure("double", 2.75, 1, 2)
    for ax, metric in zip(axes, ("absoluteError", "recoverySeconds")):
        pareto_panel(ax, table, metric, finals, training, baselines)
    pareto_legend(fig, finals, baselines)
    style.save(fig, out, "pareto_frontiers")


def pareto_frontiers_programs(tables: dict[tuple, pd.DataFrame], finals: dict[int, str], training: int,
                              out: Path, baselines: str = UNTUNED) -> None:
    """The same fronts (normalized error) on the training scenario, with motion, and on the zero-shot programs."""
    fig, axes = style.figure("double", 5.4, 2, 3)
    for ax, (scenario, title) in zip(axes.ravel(), PROGRAM_PANELS):
        pareto_panel(ax, tables[scenario], "normalizedError", finals, training, baselines)
        ax.set_title(title, loc="left")
    pareto_legend(fig, finals, baselines)
    style.save(fig, out, "pareto_frontiers_programs")


def windowed(home: Path, job: dict) -> pd.DataFrame:
    """Rate and mean absolute error of every export window, averaged over the evaluation episodes of a run; the
    exports hold running means since the episode start, so a window's mean is the difference of the integrals."""
    full = pd.read_csv(next((home / "exports" / job["key"]).glob("*_full.csv")))
    full = full[full.episode >= job["settings"]["trainingEpisodes"]]
    elapsed = full.time - full.episode * EPISODE_LENGTH
    integrals = pd.DataFrame(dict(episode=full.episode, elapsed=elapsed, rounds=full.roundsPerNodeSecond * elapsed,
                                  error=full.absoluteError * elapsed))
    episodes = integrals.groupby("episode")[["elapsed", "rounds", "error"]]
    steps, spans = episodes.diff(), episodes.diff(PERIOD_WINDOWS)
    windows = integrals.assign(rate=spans.rounds / spans.elapsed, deviation=steps.error / steps.elapsed)
    return windows.dropna(subset=["deviation"]).groupby("elapsed")[["rate", "deviation"]].mean()


def beta_traces(home: Path, jobs: list[dict], algorithm: int, beta: float,
                variant: str | None = None) -> dict[str, np.ndarray]:
    """Training seeds x windows of the effective period (1 / rounds per node per second over the last 10 s, at the
    window end) and of the error, of the runs of `variant` (any, if not given)."""
    runs = [(job["training_seed"], windowed(home, job)) for job in jobs
            if job["algorithm"] == algorithm and job["beta"] == beta and variant in (None, job["variant"])]
    frame = pd.concat([series.assign(training_seed=seed) for seed, series in runs]).reset_index()
    frame = frame[frame.elapsed >= FIRST_WINDOW_S].assign(period=lambda rows: 1.0 / rows.rate)
    means = frame.groupby(["training_seed", "elapsed"])[["period", "deviation"]].mean()
    return {column: means[column].unstack("elapsed").to_numpy() for column in ("period", "deviation")} | \
        {"time": np.sort(frame.elapsed.unique())}


def mark_event(ax, scenario: tuple, label: bool) -> None:
    if scenario[0] == "dynamic":
        ax.axvspan(EVENT_S, MOTION_END_S, color=style.PHASE_SHADE, zorder=0, linewidth=0)
    style.mark_source_switch(ax, label=False)
    if label:
        text = " half of the nodes move" if scenario[0] == "dynamic" else " source moves"
        ax.text(EVENT_S, 0.97, text, transform=ax.get_xaxis_transform(), fontsize=6, color=style.INK_MUTED,
                va="top")


def baseline_traces(period_ax, deviation_ax, home: Path, jobs: list[dict], baselines: str = UNTUNED) -> None:
    """The period and the error of the reactive `baselines` at their focus beta (the untuned ones ignore it), for
    reference."""
    for algorithm, method in BASELINE_METHODS.items():
        traces = beta_traces(home, jobs, algorithm, focus_of(baselines), baselines)
        for ax, column in ((period_ax, "period"), (deviation_ax, "deviation")):
            ax.plot(traces["time"], traces[column].mean(axis=0), color=method.color, zorder=1,
                    **BASELINE_LINE)
    deviation_ax.legend(handles=[Line2D([], [], color=method.color, label=method.short, **BASELINE_LINE)
                                 for method in BASELINE_METHODS.values()],
                        loc="upper right", fontsize=6, title="ignore $\\beta$" if baselines == UNTUNED else
                        f"tuned, $\\beta$ = {focus_of(baselines):g}", title_fontsize=6)


def beta_effect_intra_episode(home: Path, jobs: list[dict], finals: dict[int, str], scenario: tuple, name: str,
                              out: Path, baselines: str = UNTUNED) -> None:
    """How beta shapes the wake-up schedule and the error of each architecture within the evaluation episodes,
    against the reactive baselines."""
    fig, axes = style.figure("double", 3.6, 2, len(LEARNED), sharex=True)
    for column, algorithm in enumerate(LEARNED):
        period_ax, deviation_ax = axes[:, column]
        betas = grid_of(finals[algorithm])[0]
        colours = beta_colors(algorithm, betas)
        for beta in betas:
            traces = beta_traces(home, jobs, algorithm, beta, finals[algorithm])
            mean_band(period_ax, traces["time"], traces["period"], colours[beta], f"{beta:g}", linewidth=1.0)
            mean_band(deviation_ax, traces["time"], traces["deviation"], colours[beta], linewidth=1.0,
                      nonnegative=True)
        baseline_traces(period_ax, deviation_ax, home, jobs, baselines)
        period_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=len(betas), fontsize=6,
                         title=f"{architecture(algorithm, finals).label}, $\\beta$ =", title_fontsize=6.5)
        period_ax.set_ylabel("Effective wake-up period (s)")
        period_ax.set_ylim(0.4, 5.2)
        deviation_ax.set_ylabel(LABELS["absoluteError"])
        deviation_ax.set_xlabel("Time within the evaluation episodes (s)")
        deviation_ax.set_xlim(0, EPISODE_LENGTH)
        for row, ax in enumerate((period_ax, deviation_ax)):
            mark_event(ax, scenario, label=row == 1)
            style.panel_label(ax, "abcd"[row * len(LEARNED) + column])
    style.save(fig, out, name)


def inspected(jobs: list[dict], final: str) -> list[dict]:
    """The inspected GNN runs (chosen before the campaign) of the static and zero-shot scenarios."""
    return [job for job in jobs if job["evaluation_seed"] == INSPECTED_EVALUATION_SEED
            and job["training_seed"] == TRAIN_SEEDS["test"][0] and job["variant"] == final
            and job["algorithm"] == LEARNED[0] and job["beta"] == SNAPSHOT_BETA and job["nodes"] == NODES
            and job["family"] != "dynamic"]


def snapshot_figures(home: Path, jobs: list[dict], final: str, out: Path) -> None:
    """The paper's GNN filmstrips and latent dynamics on the inspected runs, laid out as the paper's exports."""
    run = f"{VARIANT_LABELS[final].lower()}, training seed {TRAIN_SEEDS['test'][0]}, test topology " \
          f"{INSPECTED_EVALUATION_SEED}, final episode"
    with tempfile.TemporaryDirectory() as folder:
        spatial = Path(folder)
        for job in inspected(jobs, final):
            source = next((home / "exports" / job["key"]).rglob(EMBEDDINGS))
            target = spatial / "generalization" / source.parent.name / EMBEDDINGS
            target.parent.mkdir(parents=True)
            target.symlink_to(source.resolve())
            if job["family"] == "static":
                (spatial / EMBEDDINGS).symlink_to(source.resolve())
        data = SimpleNamespace(spatial_dir=spatial)
        gnn.gnn_attention_filmstrip(data, out / "filmstrip")
        gnn.gnn_latent_dynamics(data, out / "attention")
        gnn_programs.gnn_program_attention_filmstrip(data, out / "filmstrip", run=run)
        gnn_programs.gnn_program_response_filmstrips(data, out / "filmstrip", run=run)


def draw(home: Path, out: Path, finals: dict[int, str], controls: tuple[str, ...] = CONTROLS) -> None:
    # The spatial and summary figures use the helpers above: imported here, so that they may import this module.
    from .campaign_paper_spatial import source_switch_dynamics, spatial_figures
    from .campaign_paper_summary import generalization_overview, summary_figures

    manifest = json.loads((home / "manifest.json").read_text())
    baselines = reactive_variant([job["variant"] for job in manifest], finals)
    jobs = [job for job in manifest if relevant(job, finals, baselines)]
    static = of_scenario(jobs, STATIC)
    # The episodes of every run are aligned on the start of the evaluation (episode_table).
    training = config.TRAINING_EPISODES
    motion = motion_episodes(next((job for job in static if job["variant"] == finals[LEARNED[0]]), static[0]))
    table = episode_table(home, static)
    learning_curves_reward(table, finals, motion, training, out / "learning", baselines)
    learning_curves_metrics(table, finals, motion, out / "learning", baselines)
    pareto_frontiers(table, finals, training, out / "static", baselines)
    tables = {scenario: episode_table(home, of_scenario(jobs, scenario)) for scenario, _ in PROGRAM_PANELS}
    pareto_frontiers_programs(tables, finals, training, out / "generalization", baselines)
    beta_effect_intra_episode(home, static, finals, STATIC, "beta_effect_intra_episode", out / "beta-study",
                              baselines)
    beta_effect_intra_episode(home, of_scenario(jobs, DYNAMIC[0]), finals, DYNAMIC[0],
                              "beta_effect_intra_episode_motion", out / "beta-study", baselines)
    snapshot_figures(home, jobs, finals[LEARNED[0]], out)
    summary_figures(table, finals, training, out / "static", baselines)
    generalization_overview(home, manifest, finals, training, out / "generalization", baselines, controls)
    spatial_figures(home, jobs, finals, out)
    source_switch_dynamics(home, jobs, finals, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home", type=Path, help="Test campaign folder (with confirmation.json)")
    parser.add_argument("--out", type=Path, default=Path("charts"))
    args = parser.parse_args()
    style.apply()
    finals = {int(key): value for key, value in
              json.loads((args.home / "confirmation.json").read_text())["selected"].items()}
    for family in ("learning", "static", "generalization", "beta-study", "filmstrip", "attention"):
        (args.out / family).mkdir(parents=True, exist_ok=True)
    draw(args.home, args.out, finals)


if __name__ == "__main__":
    main()
