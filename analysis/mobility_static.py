"""Static vs. mobile evaluation of the same scheduling policies: python -m analysis.mobility_static.

Every mobile run trains on the same schedule whatever its evaluation diffusion D (mobile-trained
policies cycle D = 0, 0.5, 2 m²/s, static-trained ones always use D = 0), so the runs that differ only
in D evaluate *the same trained policy* in a static (D = 0) or mobile (D > 0) network.

The episode is split into phases, because the integrated error of a whole episode mixes three different
regimes: the bootstrap of the gradient from scratch (every node starts without a value, an error capped
at the normalization scale), the tracking of a fixed source while the nodes move, and the re-convergence
after the source switch. Gains are paired by seed against the fixed-period frontier of the same D and
phase (no extrapolation outside the 0.5-5 s periods), and summarized with Student-t intervals over seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .mobility_pareto import ADAPTIVE, FIXED_PERIODS
from .pareto import frontier_value_at, matched_error_saving, matched_ticks_gain
from .stats import t_interval

DEFAULT_EXPORTS = (
    Path("build/exports/mobility-campaign"),
    Path("build/exports/mobility-beta-campaign"),
    Path("build/exports/mobility-diffusion-campaign"),
)
DEFAULT_OUTPUT = Path("charts/archive/mobility-static")
# The mobility level is the diffusion D (m²/s, Brownian runs) or the speed V (m/s, waypoint runs); scenario variants
# append a suffix (_waypoint, _warm). The column keeps the name "diffusion" for every level.
RUN = re.compile(
    r"seed(?P<seed>\d+)_algo(?P<algorithm>\d+)_[DV](?P<diffusion>[\d.]+)_tol(?P<tolerance>[\d.]+)"
    r"_beta(?P<beta>[\d.]+)_dt0\.25_train10_length200\.0_sourcetrue(?:_[a-z]+[\d.]*)*_(?P<kind>episodes|full)\.csv"
)
TOLERANCE_M = 5.0  # evaluation tolerance of every campaign analysed here
SEEDS = (1, 2, 3, 4)
STATIC_DIFFUSION = 0.0
MOBILE_DIFFUSION = 1.0  # the standard mobile scenario (and the one of the beta sweep)
FRAME_S = config.FULL_EXPORT_INTERVAL_S
EVALUATION = range(config.TRAINING_EPISODES, config.N_EPISODES)  # 0-based episode indices of the exports
# By 40 s every fixed period, 5 s included, has rebuilt the gradient from scratch in every campaign.
BOOTSTRAP_END_S = 40.0
EVENT_END_S = 150.0  # end of the motion of the region in the final experiment (it moves for 50 s from halfway)


@dataclass(frozen=True)
class Phase:
    key: str
    label: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


PHASES = (
    Phase("episode", "Whole episode", 0.0, config.EPISODE_LENGTH_S),
    Phase("bootstrap", f"Bootstrap (0–{BOOTSTRAP_END_S:g} s)", 0.0, BOOTSTRAP_END_S),
    Phase("post_bootstrap", f"After the bootstrap ({BOOTSTRAP_END_S:g}–{config.EPISODE_LENGTH_S:g} s)",
          BOOTSTRAP_END_S, config.EPISODE_LENGTH_S),
    Phase("tracking", f"Tracking ({BOOTSTRAP_END_S:g}–{config.SOURCE_SWITCH_S:g} s)", BOOTSTRAP_END_S,
          config.SOURCE_SWITCH_S),
    Phase("switch", f"After the switch ({config.SOURCE_SWITCH_S:g}–{config.EPISODE_LENGTH_S:g} s)",
          config.SOURCE_SWITCH_S, config.EPISODE_LENGTH_S),
    # A region that moves for a while: while it moves, and once it has stopped.
    Phase("event", f"During the event ({config.SOURCE_SWITCH_S:g}–{EVENT_END_S:g} s)", config.SOURCE_SWITCH_S,
          EVENT_END_S),
    Phase("settled", f"After the event ({EVENT_END_S:g}–{config.EPISODE_LENGTH_S:g} s)", EVENT_END_S,
          config.EPISODE_LENGTH_S),
)
PHASE_BY_KEY = {phase.key: phase for phase in PHASES}
# Exported cumulative per-node rates (per second since the episode start) -> integrated quantities.
CUMULATIVE = {"roundsPerNodeSecond": "rounds", "normalizedError": "error", "withinTolerance": "within",
              "absoluteError": "absolute", "reachableFraction": "reachable"}
GAIN_METRICS = {"error": "Integrated error", "outside": "Outside-tolerance time"}
# Per-node phase totals (s) below which the frontier has no error left: relative gains would divide ~0 by ~0.
NEGLIGIBLE = {"error": 0.01, "outside": 1.0}


# --- loading -----------------------------------------------------------------


def _run_files(directories) -> dict[tuple, dict[str, Path]]:
    """One entry per distinct run; symlinked or copied duplicates across campaigns must be identical."""
    runs: dict[tuple, dict[str, Path]] = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*.csv")):
            match = RUN.fullmatch(path.name)
            if not match or float(match["tolerance"]) != TOLERANCE_M:
                continue
            key = (int(match["algorithm"]), float(match["beta"]), float(match["diffusion"]), int(match["seed"]))
            known = runs.setdefault(key, {}).get(match["kind"])
            if known is not None and _digest(known) != _digest(path):
                raise ValueError(f"Two different exports for the same run: {known} and {path}")
            runs[key].setdefault(match["kind"], path)
    return runs


def _digest(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _evaluation_frames(episodes: Path, full: Path) -> pd.DataFrame:
    """Cumulative integrals at every 2 s frame of the evaluation episodes, closed by the 200 s episode row.

    The full export has no frame at the episode end (the next episode starts there), so the episode
    summary provides the last point.
    """
    frames = pd.read_csv(full)
    ends = pd.read_csv(episodes).assign(time=lambda rows: (rows.episode + 1) * config.EPISODE_LENGTH_S)
    rows = pd.concat([frames, ends], ignore_index=True)
    rows = rows[rows.episode.isin(EVALUATION)].copy()
    rows["t"] = np.round(rows.time - rows.episode * config.EPISODE_LENGTH_S, 6)
    for column, name in CUMULATIVE.items():
        rows[name] = rows[column] * rows.t
    rows = rows.drop_duplicates(["episode", "t"]).sort_values(["episode", "t"])
    expected = len(EVALUATION) * (int(config.EPISODE_LENGTH_S / FRAME_S) + 1)
    if len(rows) != expected:
        raise ValueError(f"Incomplete evaluation frames in {full.name}: {len(rows)} of {expected}")
    return rows[["episode", "t", *CUMULATIVE.values()]]


def phase_metrics(frames: pd.DataFrame) -> pd.DataFrame:
    """Per-node integrals of every phase, averaged over the evaluation episodes, in paper units.

    rounds: network rounds in the phase; error: integrated normalized error per node (s);
    outside: time per node outside the tolerance (s); absolute: integrated absolute error per node (m s);
    unreachable: time per node cut off from the source (s), the same for every scheduler of a seed.
    """
    indexed = frames.set_index(["episode", "t"])
    records = []
    for phase in PHASES:
        delta = (indexed.xs(phase.end, level="t") - indexed.xs(phase.start, level="t")).mean()
        records.append(dict(phase=phase.key, rounds=delta.rounds * config.N_NODES, error=delta.error,
                            outside=phase.duration - delta.within, absolute=delta.absolute,
                            unreachable=phase.duration - delta.reachable))
    return pd.DataFrame(records)


def interval_series(frames: pd.DataFrame) -> pd.DataFrame:
    """Per-frame rates (rounds per node per second, mean absolute error in m, share of nodes cut off from the
    source), averaged over episodes."""
    ordered = frames.sort_values(["episode", "t"])
    steps = ordered.groupby("episode")[["t", "rounds", "absolute", "error", "within", "reachable"]].diff()
    series = pd.DataFrame({
        "t": ordered.t,
        "rate": steps.rounds / steps.t,
        "absoluteError": steps.absolute / steps.t,
        "withinTolerance": steps.within / steps.t,
        "unreachable": 1 - steps.reachable / steps.t,
    }).dropna()
    return series.groupby("t").mean().reset_index()


def load(directories=DEFAULT_EXPORTS, partial: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-seed phase metrics, per-seed 2 s series) of every complete configuration."""
    return load_runs(_run_files(directories), partial)


def load_runs(runs: dict[tuple, dict[str, Path]], partial: bool = False,
              seeds=SEEDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``load`` over runs keyed by (algorithm, beta, level, seed), e.g. a variant laid over a baseline."""
    metrics, series = [], []
    for (algorithm, beta, diffusion, seed), files in sorted(runs.items()):
        if set(files) != {"episodes", "full"}:
            continue
        try:
            frames = _evaluation_frames(files["episodes"], files["full"])
        except ValueError:
            if partial:
                continue
            raise
        keys = dict(algorithm=algorithm, beta=beta, diffusion=diffusion, seed=seed)
        metrics.append(phase_metrics(frames).assign(**keys))
        series.append(interval_series(frames).assign(**keys))
    if not metrics:
        raise ValueError(f"No complete mobile runs among {len(runs)}")
    complete = _complete(pd.concat(metrics, ignore_index=True), partial, seeds)
    series = pd.concat(series, ignore_index=True).merge(
        complete[["algorithm", "beta", "diffusion", "seed"]].drop_duplicates())
    return complete, series


def _complete(metrics: pd.DataFrame, partial: bool, expected=SEEDS) -> pd.DataFrame:
    """Keep configurations with every seed and a fixed-period reference for their D (or fail)."""
    metrics = metrics[metrics.seed.isin(expected)]
    seeds = metrics.groupby(["algorithm", "beta", "diffusion"]).seed.agg(lambda values: set(values) == set(expected))
    fixed = [(algorithm, diffusion) for (algorithm, _, diffusion), ok in seeds.items()
             if ok and algorithm in FIXED_PERIODS]
    references = {diffusion for diffusion in metrics.diffusion.unique()
                  if all((algorithm, diffusion) in fixed for algorithm in FIXED_PERIODS)}
    complete = [key for key, ok in seeds.items() if ok and key[2] in references]
    missing = sorted(set(seeds.index) - set(complete))
    if missing and not partial:
        raise ValueError(f"Incomplete configurations (algorithm, beta, D): {missing}")
    keep = metrics.set_index(["algorithm", "beta", "diffusion"]).index.isin(complete)
    return metrics[keep].reset_index(drop=True)


# --- comparisons with the fixed-period frontier ------------------------------


def fixed_reference(metrics: pd.DataFrame) -> pd.DataFrame:
    """Fixed-period points (beta-independent), with their nominal period."""
    fixed = metrics[metrics.algorithm.isin(FIXED_PERIODS)].copy()
    fixed["period"] = fixed.algorithm.map(FIXED_PERIODS)
    return fixed


def frontier_gains(metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired (per-seed) gains of each adaptive configuration over the frontier of its D, seed and phase."""
    fixed = fixed_reference(metrics).set_index(["diffusion", "seed", "phase"]).sort_index()
    adaptive = metrics[metrics.algorithm.isin(ADAPTIVE)]
    records = []
    for (algorithm, beta, diffusion, phase), rows in adaptive.groupby(["algorithm", "beta", "diffusion", "phase"]):
        samples = {name: [] for name in [*GAIN_METRICS, "rounds_at_error"]}
        for row in rows.itertuples():
            reference = fixed.loc[(diffusion, row.seed, phase)]
            for metric in GAIN_METRICS:
                samples[metric].append(_relative_gain(row.rounds, getattr(row, metric), reference, metric))
            samples["rounds_at_error"].append(matched_error_saving(row.rounds, row.error,
                                                                   reference.rounds, reference.error))
        for gain, sample in samples.items():
            interval = t_interval(sample)
            metric = "error" if gain == "rounds_at_error" else gain
            records.append(dict(algorithm=algorithm, beta=beta, diffusion=diffusion, phase=phase, gain=gain,
                                mean=interval.mean, low=interval.low, high=interval.high, seeds=interval.n,
                                undefined=_undefined(rows, fixed.loc[(diffusion, slice(None), phase)], metric)))
    return pd.DataFrame(records)


def _undefined(rows: pd.DataFrame, reference: pd.DataFrame, metric: str) -> str:
    """Why a gain has no value: the fixed periods are exact in the phase, or the rounds are outside their range."""
    if reference[metric].max() <= NEGLIGIBLE[metric]:
        return "converged"
    if rows.rounds.min() < reference.rounds.min() or rows.rounds.max() > reference.rounds.max():
        return "outside_range"
    return ""


def paired_gains(metrics: pd.DataFrame, metric: str = "error") -> pd.DataFrame:
    """Per-seed gains (the samples of ``frontier_gains``) of each adaptive configuration and phase."""
    fixed = fixed_reference(metrics).set_index(["diffusion", "seed", "phase"]).sort_index()
    adaptive = metrics[metrics.algorithm.isin(ADAPTIVE)]
    gains = [_relative_gain(row.rounds, getattr(row, metric), fixed.loc[(row.diffusion, row.seed, row.phase)], metric)
             for row in adaptive.itertuples()]
    return adaptive[["algorithm", "beta", "diffusion", "phase", "seed"]].assign(gain=gains)


def connected(metrics: pd.DataFrame) -> pd.DataFrame:
    """The metrics without the nodes cut off from the source, which no schedule can correct.

    A component without the source counts to infinity: its finite values are at the maximum normalized error for
    every scheduler (the motion does not depend on the schedule), so the paired comparisons subtract the same
    amount. An isolated node that holds infinity is exact, hence this is an upper bound of the subtraction.
    """
    return metrics.assign(error=metrics.error - metrics.unreachable)


def _relative_gain(rounds: float, value: float, reference: pd.DataFrame, metric: str) -> float:
    """Gain at equal rounds; undefined (NaN) when the frontier has nothing left to reduce in the phase."""
    frontier = frontier_value_at(reference.rounds, reference[metric], rounds)
    return float("nan") if not frontier > NEGLIGIBLE[metric] else matched_ticks_gain(
        rounds, value, reference.rounds, reference[metric])


def configuration_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% t interval over seeds of every metric, per configuration and phase."""
    records = []
    for (algorithm, beta, diffusion, phase), rows in metrics.groupby(["algorithm", "beta", "diffusion", "phase"]):
        record = dict(algorithm=algorithm, beta=beta, diffusion=diffusion, phase=phase, seeds=len(rows))
        for metric in ["rounds", "error", "outside", "absolute"]:
            interval = t_interval(rows[metric])
            record.update({metric: interval.mean, f"{metric}_low": interval.low, f"{metric}_high": interval.high})
        records.append(record)
    return pd.DataFrame(records)


def bootstrap_seconds(series: pd.DataFrame, threshold_m: float) -> pd.DataFrame:
    """First time at which the mean absolute error falls below ``threshold_m`` (per run, NaN if never)."""
    early = series[series.t <= BOOTSTRAP_END_S]
    below = early[early.absoluteError <= threshold_m]
    first = below.groupby(["algorithm", "beta", "diffusion", "seed"]).t.min()
    runs = early.groupby(["algorithm", "beta", "diffusion", "seed"]).size().index
    return first.reindex(runs).rename("seconds").reset_index()


def switch_reaction(series: pd.DataFrame, window_s: float = 20.0) -> pd.DataFrame:
    """Per run: round rate in the ``window_s`` after the source switch over the rate in the ``window_s`` before."""
    switch = config.SOURCE_SWITCH_S
    before = series[(series.t > switch - window_s) & (series.t <= switch)]
    after = series[(series.t > switch) & (series.t <= switch + window_s)]
    keys = ["algorithm", "beta", "diffusion", "seed"]
    ratio = after.groupby(keys).rate.mean() / before.groupby(keys).rate.mean()
    return ratio.rename("ratio").reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exports", type=Path, nargs="+", default=list(DEFAULT_EXPORTS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partial", action="store_true", help="skip incomplete configurations (previews only)")
    arguments = parser.parse_args()
    from .figures import mobility_static as figures
    from .mobility_report import write_report

    metrics, series = load(arguments.exports, arguments.partial)
    gains = frontier_gains(metrics)
    summary = configuration_summary(metrics)
    arguments.out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(arguments.out / "per_seed_phases.csv", index=False)
    summary.to_csv(arguments.out / "summary_phases.csv", index=False)
    gains.to_csv(arguments.out / "frontier_gains_phases.csv", index=False)
    series.to_csv(arguments.out / "series_2s.csv", index=False)
    figures.render(metrics, series, gains, arguments.out)
    write_report(metrics, series, gains, arguments.out / "summary.md")
    print(f"Saved {len(summary)} configuration-phases to {arguments.out}")


if __name__ == "__main__":
    main()
