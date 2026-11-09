"""Derived results shared by the figures, the LaTeX tables and the summary.

All functions take the tidy tables of :mod:`analysis.data` and return
DataFrames, so that every number appears once and is reused everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import config
from .data import METRICS, select
from .methods import BETA_CONDITIONED, FIXED, LEARNED, Method
from .pareto import matched_error_saving, matched_ticks_gain
from .stats import Interval, bootstrap_two_sample, compare, holm, relative_change, summarize

# Windows of the final episode used to characterise the reaction to the source switch.
STEADY_WINDOW_S = (60.0, 100.0)  # converged, before the source moves
TRANSIENT_WINDOW_S = (100.0, 140.0)  # right after the source moves
RECOVERY_TOLERANCE_M = 1.0  # |mean g - mean g*| below which the network has re-converged
# A seed spending >= 85 % of the rounds of the 1 s schedule behaves like the fastest baseline.
COLLAPSE_FRACTION = 0.85
COMPARED_METRICS = ("reward", "error", "ticks", "unstable")


# --- fixed-period baselines ---------------------------------------------------


def fixed_runs(evaluation: pd.DataFrame, weight: float = config.FOCUS_WEIGHT) -> pd.DataFrame:
    """Per-seed evaluation rows of the fixed baselines.

    Only the reward depends on beta: rounds, error and unstable time of a
    fixed schedule are identical for every weight (checked in
    :func:`fixed_metrics_independent_of_weight`).
    """
    return select(evaluation, FIXED.key, weight=weight).sort_values(["period", "seed"])


def fixed_metrics_independent_of_weight(evaluation: pd.DataFrame) -> bool:
    fixed = evaluation[evaluation.method == FIXED.key]
    spread = fixed.groupby(["period", "seed"])[["ticks", "error", "unstable"]].agg(lambda s: s.max() - s.min())
    return bool(np.allclose(spread.to_numpy(), 0.0))


def fixed_frontier(evaluation: pd.DataFrame, metric: str, weight: float = config.FOCUS_WEIGHT) -> pd.DataFrame:
    """Mean rounds and mean ``metric`` of every fixed period (sorted by period)."""
    runs = fixed_runs(evaluation, weight)
    return runs.groupby("period")[["ticks", metric]].mean().reset_index()


def best_fixed_period(evaluation: pd.DataFrame, weight: float) -> float:
    """Fixed period with the highest mean reward for ``weight`` (the RL objective)."""
    runs = select(evaluation, FIXED.key, weight=weight)
    return float(runs.groupby("period").reward.mean().idxmax())


# --- learned methods ------------------------------------------------------------


def metric_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Mean, t-interval, sd and n of every metric per (method, weight, period)."""
    tables = []
    for metric in METRICS:
        table = summarize(evaluation, ["method", "weight", "period"], metric)
        tables.append(table.assign(metric=metric))
    return pd.concat(tables, ignore_index=True)


def significance(evaluation: pd.DataFrame, metrics: Sequence[str] = COMPARED_METRICS) -> pd.DataFrame:
    """Each learned method vs. the best fixed period of the same beta.

    Holm-Bonferroni correction is applied per metric over the
    (method x beta) family of comparisons.
    """
    rows = []
    for weight in config.WEIGHTS:
        period = best_fixed_period(evaluation, weight)
        reference = select(evaluation, FIXED.key, weight=weight, period=period)
        for method in LEARNED:
            runs = select(evaluation, method.key, weight=weight)
            for metric in metrics:
                result = compare(runs[metric], reference[metric])
                reference_mean = reference[metric].mean()
                rows.append(
                    {
                        "weight": weight,
                        "method": method.key,
                        "metric": metric,
                        "best_period": period,
                        "mean": runs[metric].mean(),
                        "reference_mean": reference_mean,
                        "difference": result.difference,
                        "relative": result.difference / abs(reference_mean),
                        "hedges_g": result.hedges_g,
                        "cliffs_delta": result.cliffs_delta,
                        "welch_p": result.welch_p,
                        "mwu_p": result.mwu_p,
                        "n": result.n_a,
                        "n_reference": result.n_b,
                    }
                )
    table = pd.DataFrame(rows)
    for column in ("welch_p", "mwu_p"):
        table[f"{column}_holm"] = table.groupby("metric")[column].transform(lambda p: holm(p.to_numpy()))
    return table


def _bootstrap_gain(method_runs: pd.DataFrame, baseline: pd.DataFrame, metric: str, saving: bool) -> Interval:
    """Gain of one configuration against the fixed frontier, bootstrapping seeds of both."""
    ticks = method_runs.ticks.to_numpy()
    values = method_runs[metric].to_numpy()
    table = baseline.pivot(index="seed", columns="period", values=metric)
    ref_values = table.to_numpy()
    ref_ticks = baseline.groupby("period").ticks.mean().reindex(table.columns).to_numpy()

    def statistic(method_index: np.ndarray, seed_index: np.ndarray) -> float:
        i, j = method_index.astype(int), seed_index.astype(int)
        frontier = ref_values[j].mean(axis=0)
        gain = matched_error_saving if saving else matched_ticks_gain
        return gain(ticks[i].mean(), values[i].mean(), ref_ticks, frontier)

    ci = bootstrap_two_sample(np.arange(len(ticks)), np.arange(len(ref_values)), statistic)
    if np.isnan(ci.mean):  # the point lies outside the frontier's range
        return Interval(ci.mean, float("nan"), float("nan"), ci.n)
    return ci


FRONTIER_GAINS = {
    "error_at_ticks": "error reduction at equal rounds",
    "unstable_at_ticks": "unstable-time reduction at equal rounds",
    "rounds_at_error": "rounds saved at equal error",
    "rounds_vs_fastest": "rounds saved vs. 1 s period",
    "error_vs_fastest": "error reduction vs. 1 s period",
}


def frontier_gains(evaluation: pd.DataFrame, methods: tuple[Method, ...] = LEARNED,
                   weights=config.WEIGHTS) -> pd.DataFrame:
    """Gains of every learned (method, beta of `weights`) over the fixed-period schedules.

    * ``error_at_ticks`` / ``unstable_at_ticks``: relative reduction w.r.t. the
      fixed-period frontier at the same rounds;
    * ``rounds_at_error``: fraction of rounds saved w.r.t. a fixed period with
      the same error (NaN when the error is below the 1 s schedule's: no fixed
      period reaches it -- see ``rounds_vs_fastest`` then);
    * ``rounds_vs_fastest`` / ``error_vs_fastest``: rounds saved and error
      reduction w.r.t. the fastest (1 s) schedule.

    Intervals: 95 % percentile bootstrap over the seeds of both groups.
    """
    baseline = fixed_runs(evaluation)
    fastest = baseline[np.isclose(baseline.period, config.FIXED_PERIODS_S[0])]
    rows = []
    for method in methods:
        for weight in weights:
            runs = select(evaluation, method.key, weight=weight)
            intervals = {
                "error_at_ticks": _bootstrap_gain(runs, baseline, "error", saving=False),
                "unstable_at_ticks": _bootstrap_gain(runs, baseline, "unstable", saving=False),
                "rounds_at_error": _bootstrap_gain(runs, baseline, "error", saving=True),
                "rounds_vs_fastest": bootstrap_two_sample(runs.ticks, fastest.ticks, relative_change),
                "error_vs_fastest": bootstrap_two_sample(runs.error, fastest.error, relative_change),
            }
            for name, ci in intervals.items():
                rows.append(
                    {
                        "method": method.key,
                        "weight": weight,
                        "gain": name,
                        "mean": ci.mean,
                        "low": ci.low,
                        "high": ci.high,
                        "n": ci.n,
                    }
                )
    return pd.DataFrame(rows)


def collapsed_seeds(ticks: np.ndarray, fastest_ticks: float) -> np.ndarray:
    """Seeds whose policy degenerated to (almost) always the fastest period."""
    return np.asarray(ticks) >= COLLAPSE_FRACTION * fastest_ticks


def stability(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Across-seed variability of the rounds per learned (method, beta)."""
    fastest = fixed_runs(evaluation).query("period == @config.FIXED_PERIODS_S[0]").ticks.mean()
    rows = []
    for method in LEARNED:
        for weight in config.WEIGHTS:
            ticks = select(evaluation, method.key, weight=weight).ticks.to_numpy()
            rows.append(
                {
                    "method": method.key,
                    "weight": weight,
                    "ticks_cv": float(np.std(ticks, ddof=1) / np.mean(ticks)),
                    "n_collapsed": int(collapsed_seeds(ticks, fastest).sum()),
                    "n": len(ticks),
                }
            )
    return pd.DataFrame(rows)


# --- reaction to the source switch (final episode) ----------------------------


def _window(trace: pd.DataFrame, window: tuple[float, float]) -> pd.DataFrame:
    return trace[(trace.t >= window[0]) & (trace.t < window[1])]


def _rate(trace: pd.DataFrame, window: tuple[float, float]) -> float:
    """Network rounds per second within ``window`` (from the cumulative tick count)."""
    start, end = np.interp(window, trace.t, trace.ticks)
    return (end - start) / (window[1] - window[0])


def switch_response(trace: pd.DataFrame) -> dict[str, float]:
    """How one run reacts to the source switch in the final episode."""
    after = trace[trace.t >= config.SOURCE_SWITCH_S]
    above = np.flatnonzero(after.deviation.to_numpy() >= RECOVERY_TOLERANCE_M)
    if len(above) == 0:
        recovery = 0.0
    elif above[-1] == len(after) - 1:
        recovery = float("nan")  # never re-converged within the episode
    else:
        recovery = float(after.t.iloc[above[-1] + 1] - config.SOURCE_SWITCH_S)
    return {
        "steady_period": float(_window(trace, STEADY_WINDOW_S).wakeup.mean()),
        "min_period": float(_window(trace, TRANSIENT_WINDOW_S).wakeup.min()),
        "rate_ratio": _rate(trace, TRANSIENT_WINDOW_S) / _rate(trace, STEADY_WINDOW_S),
        "recovery_time": recovery,
    }


def switch_responses(traces: pd.DataFrame, weight: float = config.FOCUS_WEIGHT) -> pd.DataFrame:
    """Per-seed switch-response metrics of the learned methods (at ``weight``) and fixed periods."""
    rows = []
    configurations = [(m.key, None) for m in LEARNED] + [(FIXED.key, p) for p in config.FIXED_PERIODS_S]
    for method, period in configurations:
        runs = select(traces, method, weight=weight, period=period)
        for seed, trace in runs.groupby("seed"):
            rows.append({"method": method, "period": period, "seed": seed, **switch_response(trace)})
    return pd.DataFrame(rows)


# --- generalization -------------------------------------------------------------


def generalization_points(generalization: pd.DataFrame) -> pd.DataFrame:
    """Evaluation means per (program, method, period, beta_in).

    beta-conditioned policies give one point per beta_in (two episodes each);
    Q-learning and the fixed periods ignore beta_in and use all evaluation episodes.
    """
    evaluation = generalization[generalization.episode.isin(config.EVAL_EPISODES)].copy()
    evaluation.loc[~evaluation.method.isin(BETA_CONDITIONED), "beta_in"] = np.nan
    keys = ["program", "method", "period", "beta_in"]
    points = evaluation.groupby(keys, dropna=False)[list(METRICS)].mean()
    points["n_episodes"] = evaluation.groupby(keys, dropna=False).size()
    points["n_seeds"] = evaluation.groupby(keys, dropna=False).seed.nunique()
    return points.reset_index()


def generalization_gains(points: pd.DataFrame) -> pd.DataFrame:
    """Error / unstable-time gains at matched rounds against each program's fixed frontier."""
    rows = []
    for program, group in points.groupby("program"):
        fixed = group[group.method == FIXED.key].sort_values("ticks")
        for _, point in group[group.method != FIXED.key].iterrows():
            rows.append(
                {
                    "program": program,
                    "method": point.method,
                    "beta_in": point.beta_in,
                    "ticks": point.ticks,
                    "error_at_ticks": matched_ticks_gain(point.ticks, point.error, fixed.ticks, fixed.error),
                    "unstable_at_ticks": matched_ticks_gain(point.ticks, point.unstable, fixed.ticks, fixed.unstable),
                    "rounds_at_error": matched_error_saving(point.ticks, point.error, fixed.ticks, fixed.error),
                }
            )
    return pd.DataFrame(rows)
