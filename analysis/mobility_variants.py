"""Incremental variants of the mobile experiment: python -m analysis.mobility_variants.

Every variant adds one change to the previous one (see ``build/mobility-variants/run_variants.py``): the complete
replay, a 4x training budget, the staleness cost, a 50 m loss scale, a 6 s message retention, waypoint mobility and
a warm start, then three optional local features (own speed, change not explained by the motion, dependants);
finally, the same configuration under other scenarios: trained on two events only (a switching source in a static
network, a region that starts moving) and evaluated on the switch (static: seen; every node moving: not seen) and on
the moving region, and trained and evaluated with only half of the nodes (a region) moving.
Training variants reuse the fixed periods and heuristic of the scenario they build on; scenario variants (the
retention, the waypoint mobility and the warm start) run their own. Every quantity is a mean over the seeds of the
variant (4, or 8 from the warm start on) with a 95% t interval. Gains are also reported on the error of the nodes
connected to the source (``connected_`` quantities): waypoint mobility and the warm start cut 7-11% of the nodes off
the source, and those count to infinity at the maximum error whatever the schedule.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .mobility_static import (DEFAULT_EXPORTS, SEEDS, _run_files, connected, fixed_reference, frontier_gains,
                              load_runs, paired_gains, switch_reaction)
from .stats import t_interval

VARIANT_EXPORTS = Path("build/exports/mobility-variants")
DEFAULT_OUTPUT = Path("charts/archive/mobility-variants")
LEARNERS = (0, 1)
HEURISTIC = 2
FOCUS_BETA = config.FOCUS_WEIGHT
BETA_RANGE = (0.5, 1.0)
FASTEST, SLOWEST = 10, 7  # fixed 0.5 s and 5 s
MORE_SEEDS = tuple(range(1, 9))
# Gains reported per variant: (level, phase), the level being the index in Variant.levels (static, focus, fastest).
GAINS = {"gain_episode": (1, "episode"), "gain_bootstrap": (1, "bootstrap"), "gain_tracking": (1, "tracking"),
         "gain_switch": (1, "switch"), "gain_post_static": (0, "post_bootstrap"),
         "gain_post_focus": (1, "post_bootstrap"), "gain_post_fastest": (2, "post_bootstrap")}
CONNECTED = "connected_"  # prefix of the gains on the error of the nodes connected to the source
LADDER_END = "dependants"  # last step of the cumulative ladder: the final configuration
# The final configuration under other training and evaluation scenarios, compared per level and phase.
SCENARIO_VARIANTS = ("dependants", "events-switch", "events-region", "region-moves")
SCENARIO_PHASES = ("bootstrap", "tracking", "switch")


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    change: str
    levels: tuple[float, float, float]  # static, focus and fastest mobility level (D in m²/s or speed in m/s)
    scenario: str = "baseline"  # the variant whose fixed periods and heuristic it uses: itself for a new scenario
    seeds: tuple[int, ...] = SEEDS

    @property
    def folder(self) -> Path | None:
        return None if self.key == "baseline" else VARIANT_EXPORTS / self.key


BROWNIAN = (0.0, 1.0, 4.0)
WAYPOINT = (0.0, 1.0, 2.0)
VARIANTS = (
    Variant("baseline", "Baseline", "capped replay, 110 steps, held error / 500 m, 15 s retention, Brownian",
            BROWNIAN),
    Variant("complete-replay", "+Replay", "complete replay (every episode, every node)", BROWNIAN),
    Variant("more-training", "+Training", "4x training steps (440 per episode)", BROWNIAN),
    Variant("staleness-cost", "+Staleness", "staleness cost instead of the held error", BROWNIAN),
    Variant("local-loss-scale", "+50 m scale", "loss scale 50 m instead of 500 m", BROWNIAN),
    Variant("short-retention", "+Retention", "message retention 6 s instead of 15 s", BROWNIAN, "short-retention"),
    Variant("waypoint-mobility", "+Waypoint", "waypoint mobility with pauses (levels: speed in m/s)", WAYPOINT,
            "waypoint-mobility"),
    Variant("warm-start", "+Warm start", "warm start (no redeployment between episodes); 8 seeds", WAYPOINT,
            "warm-start", MORE_SEEDS),
    Variant("own-speed", "+Speed", "own speed feature (odometry)", WAYPOINT, "warm-start", MORE_SEEDS),
    Variant("unexplained-change", "+Unexplained", "feature: change of the own value not explained by the motion",
            WAYPOINT, "warm-start", MORE_SEEDS),
    Variant("dependants", "+Dependants", "feature: share of neighbours that use the node as parent", WAYPOINT,
            "warm-start", MORE_SEEDS),
    Variant("events-switch", "+Event training", "trained on a switching source in a static network and on a region "
            "that starts moving halfway (1, 2 m/s); evaluated on the source switch", WAYPOINT, "events-switch",
            MORE_SEEDS),
    Variant("events-region", "Region event", "the same training, evaluated on a region that starts moving halfway "
            "(static level: no event at all)", WAYPOINT, "events-region", MORE_SEEDS),
    Variant("final-static", "Final, static", "trained on the static switch and on a region that moves for 50 s from "
            "halfway, with the neighbours' speeds; evaluated on the static switch", WAYPOINT, "final-static", MORE_SEEDS),
    Variant("final-region", "Final, region", "the same policies, evaluated on a region (half of the nodes) that moves "
            "for 50 s from halfway", WAYPOINT, "final-region", MORE_SEEDS),
    Variant("region-moves", "Half moving", "the final configuration where only half of the nodes (a region chosen at "
            "the episode start) move, in training and in evaluation; the source switches halfway", WAYPOINT,
            "region-moves", MORE_SEEDS),
)
VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}


def variant_runs(variant: Variant) -> dict[tuple, dict[str, Path]]:
    """The runs of a variant; training variants borrow the schedulers that do not learn from their scenario."""
    if variant.folder is None:
        return _run_files(DEFAULT_EXPORTS)
    own = _run_files([variant.folder])
    if variant.scenario == variant.key:
        return own
    reference = variant_runs(VARIANT_BY_KEY[variant.scenario])
    return {key: files for key, files in reference.items() if key[0] not in LEARNERS} | own


def _record(variant: Variant, algorithm: int, quantity: str, values) -> dict:
    interval = t_interval(list(values))
    return dict(variant=variant.key, algorithm=algorithm, quantity=quantity, mean=interval.mean, low=interval.low,
                high=interval.high, seeds=interval.n)


def _per_seed(metrics: pd.DataFrame, algorithm: int, level: float, beta: float, phase: str = "episode") -> pd.Series:
    rows = metrics[(metrics.algorithm == algorithm) & np.isclose(metrics.diffusion, level)
                   & np.isclose(metrics.beta, beta) & (metrics.phase == phase)]
    return rows.set_index("seed").rounds


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator).dropna()


def _gains(variant: Variant, gains: pd.DataFrame, algorithm: int, prefix: str = "") -> list[dict]:
    records = []
    for quantity, (level, phase) in GAINS.items():
        rows = gains[(gains.algorithm == algorithm) & np.isclose(gains.diffusion, variant.levels[level])
                     & (gains.phase == phase) & (gains.gain == "error") & np.isclose(gains.beta, FOCUS_BETA)]
        if not rows.empty:
            cell = rows.iloc[0]
            records.append(dict(variant=variant.key, algorithm=algorithm, quantity=prefix + quantity,
                                mean=cell["mean"], low=cell["low"], high=cell["high"], seeds=cell["seeds"]))
    return records


def per_seed_gains(variant: Variant, metrics: pd.DataFrame) -> pd.DataFrame:
    """The paired gains of every seed at beta = 0.95, on the whole error and on the connected nodes only."""
    frames = []
    for prefix, data in (("", metrics), (CONNECTED, connected(metrics))):
        gains = paired_gains(data)
        gains = gains[np.isclose(gains.beta, FOCUS_BETA)]
        for quantity, (level, phase) in GAINS.items():
            rows = gains[np.isclose(gains.diffusion, variant.levels[level]) & (gains.phase == phase)]
            frames.append(rows[["algorithm", "seed", "gain"]].assign(variant=variant.key, quantity=prefix + quantity))
    return pd.concat(frames, ignore_index=True)


def _behaviour(variant: Variant, metrics: pd.DataFrame, reaction: pd.DataFrame, algorithm: int) -> list[dict]:
    low, focus, high = variant.levels
    records = [_record(variant, algorithm, "rounds_focus", _per_seed(metrics, algorithm, focus, FOCUS_BETA)),
               _record(variant, algorithm, "level_response", _ratio(_per_seed(metrics, algorithm, high, FOCUS_BETA),
                                                                    _per_seed(metrics, algorithm, low, FOCUS_BETA)))]
    beta_ratio = _ratio(_per_seed(metrics, algorithm, focus, BETA_RANGE[1]),
                        _per_seed(metrics, algorithm, focus, BETA_RANGE[0]))
    if not beta_ratio.empty:
        records.append(_record(variant, algorithm, "beta_range", beta_ratio))
    for quantity, level in (("reaction_static", low), ("reaction_focus", focus)):
        rows = reaction[(reaction.algorithm == algorithm) & np.isclose(reaction.diffusion, level)
                        & np.isclose(reaction.beta, FOCUS_BETA)]
        records.append(_record(variant, algorithm, quantity, rows.ratio))
    return records


def _frontier(variant: Variant, metrics: pd.DataFrame) -> list[dict]:
    """How much the schedule can matter: error of the 5 s period over the 0.5 s one, per phase, at the focus level."""
    fixed = fixed_reference(metrics)
    fixed = fixed[np.isclose(fixed.diffusion, variant.levels[1])].set_index(["algorithm", "phase", "seed"]).error
    if not {SLOWEST, FASTEST} <= set(fixed.index.get_level_values("algorithm")):
        return []
    return [_record(variant, -1, f"fixed_ratio_{phase}", _ratio(fixed[SLOWEST][phase], fixed[FASTEST][phase]))
            for phase in ("bootstrap", "tracking", "switch")]


def load_variant(variant: Variant, partial: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-seed phase metrics, per-seed 2 s series) of the variant at its three levels."""
    metrics, series = load_runs(variant_runs(variant), partial, variant.seeds)
    metrics = metrics[metrics.diffusion.isin(variant.levels)]
    if not metrics.algorithm.isin(LEARNERS).any():
        raise ValueError(f"No complete learner configuration in {variant.key}")
    return metrics, series[series.diffusion.isin(variant.levels)]


def summarize(variant: Variant, metrics: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    gains, reaction = frontier_gains(metrics), switch_reaction(series)
    connected_gains = frontier_gains(connected(metrics))
    records = _frontier(variant, metrics)
    for algorithm in (*LEARNERS, HEURISTIC):
        records += (_gains(variant, gains, algorithm) + _gains(variant, connected_gains, algorithm, CONNECTED)
                    + _behaviour(variant, metrics, reaction, algorithm))
    return pd.DataFrame(records)


def scenario_gains(variant: Variant, metrics: pd.DataFrame, phases=SCENARIO_PHASES) -> pd.DataFrame:
    """Gain over the fixed periods at beta = 0.95 per scheduler, level and phase, whole error and connected nodes."""
    records = []
    for prefix, data in (("", metrics), (CONNECTED, connected(metrics))):
        gains = paired_gains(data)
        gains = gains[np.isclose(gains.beta, FOCUS_BETA) & gains.phase.isin(phases)]
        for (algorithm, level, phase), rows in gains.groupby(["algorithm", "diffusion", "phase"]):
            interval = t_interval(rows.gain.to_numpy())
            records.append(dict(variant=variant.key, algorithm=algorithm, estimate=prefix or "whole", level=level,
                                phase=phase, mean=interval.mean, low=interval.low, high=interval.high,
                                seeds=interval.n))
    return pd.DataFrame(records)


def available(variants=VARIANTS) -> list[Variant]:
    return [variant for variant in variants if variant.folder is None or variant.folder.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partial", action="store_true", help="skip incomplete configurations (previews only)")
    arguments = parser.parse_args()
    from .figures import mobility_final as final_figures
    from .figures import mobility_variants as figures
    from .mobility_variants_report import write_report

    loaded = {}
    for variant in available():
        try:
            loaded[variant.key] = load_variant(variant, arguments.partial)
        except ValueError as error:
            if not arguments.partial:
                raise
            print(f"Skipping {variant.key}: {error}")
    summary = pd.concat([summarize(VARIANT_BY_KEY[key], *data) for key, data in loaded.items()], ignore_index=True)
    seeds = pd.concat([per_seed_gains(VARIANT_BY_KEY[key], data[0]) for key, data in loaded.items()],
                      ignore_index=True)
    scenarios = pd.concat([scenario_gains(VARIANT_BY_KEY[key], loaded[key][0])
                           for key in SCENARIO_VARIANTS if key in loaded], ignore_index=True)
    arguments.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(arguments.out / "variants_summary.csv", index=False)
    seeds.to_csv(arguments.out / "variants_per_seed_gains.csv", index=False)
    scenarios.to_csv(arguments.out / "scenario_gains.csv", index=False)
    figures.render(summary, scenarios, arguments.out)
    final_figures.render(loaded, seeds, arguments.out)
    write_report(summary, arguments.out / "variants.md")
    print(f"Saved {summary.variant.nunique()} variants to {arguments.out}")


if __name__ == "__main__":
    main()
