"""Figures of the mobile experiment: python -m analysis.mobility_paper.

One policy per seed and beta (0.5, 0.8, 0.95, 0.99, 1), trained on two kinds of episodes: a static network whose source
switches halfway, and a still network where the half of the nodes nearest a random point moves from halfway for 50 s
(waypoints at 1 or 2 m/s). Every episode also starts with a new source (warm start). Besides its own speed, a node sees
the speed of each neighbour (shared with the messages). Evaluation, 8 seeds:

* **static**: the source switches at 100 s (``final-static``);
* **dynamic**: half of the nodes move from 100 s to 150 s (``final-region``, 1 m/s with every beta, 2 m/s at 0.95).

``charts/mobility/static`` and ``charts/mobility/dynamic`` hold the figures of the static paper (``charts/paper``) for
each scenario, drawn by the paper's own functions (whole-episode metrics, 1-5 s frontier, bootstrap over seeds), with
the attention and the maps of the inspected run (seed 2, last episode); ``dynamic`` adds where the rounds and the error
go when the region moves. ``charts/mobility/recap`` compares the two with gains paired by seed, per phase.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from . import config
from .dataset import Dataset
from .methods import FIXED, GNN, TD3
from .mobility_regions import COMMUNICATION_RANGE_M, FINAL_REGION, allocation, route_summary, route_table, snapshot
from .mobility_regions import load as load_region_series
from .mobility_static import PHASE_BY_KEY
from .mobility_variants import VARIANT_BY_KEY, VARIANT_EXPORTS, load_variant, scenario_gains, variant_runs

OUTPUT = Path("charts/mobility")
STATIC_VARIANT = "final-static"
REGION = FINAL_REGION
STATIC_LEVEL = 0.0
PHASES = ("bootstrap", "tracking", "event", "settled")  # new source, quiet, the event, after it
SWEEP_LEVEL = 1.0  # the region level of the beta sweep
REFERENCE_FIXED = 5  # the fixed 3 s period: errors vs. beta are relative to it, seed by seed
# The figures of the static paper, drawn with its own functions: its two learners that the mobile experiment trains
# (TD3+BC is the MLP), against its frontier of fixed periods (1-5 s, no 0.5 s).
PAPER_METHODS = (TD3, GNN)
PAPER_ALGORITHMS = {0: GNN.key, 1: TD3.key}
PAPER_FIXED = {3: 1.0, 4: 2.0, 5: 3.0, 6: 4.0, 7: 5.0}  # algorithm -> period (s)
PAPER_SCENARIOS = {"static": (STATIC_VARIANT, STATIC_LEVEL), "dynamic": (REGION.variant, SWEEP_LEVEL)}
RECAP = "recap"
INSPECTED_SEED = 2  # the run whose last episode holds the snapshots (``FinalEpisodeSnapshots.InspectedRun``)
INSPECTED_ALGORITHMS = (0, 1)  # the GNN records its attention, the MLP the state of the nodes
SNAPSHOTS = "spatial"  # sub-folder of the exports with one folder of snapshots per inspected run
# One folder per scenario gathering the snapshots of its inspected runs, as the paper's ``Dataset`` expects them.
SNAPSHOT_VIEWS = Path("build/mobility-snapshot-views")
MOTION_EPSILON_M = 1e-3  # a node moved if its recorded position changed by more than this in one second


@dataclass(frozen=True)
class PaperData:
    static_metrics: pd.DataFrame  # per-seed phase metrics of the static network, every beta
    static_series: pd.DataFrame  # per-seed 2 s series of the static network
    region_metrics: pd.DataFrame  # per-seed phase metrics of the region scenario, every level and beta
    region_network: pd.DataFrame  # per-seed 2 s series of the whole network in the region scenario, every beta
    region_series: pd.DataFrame  # per-seed 2 s series of the moving region and of the still nodes, beta = 0.95
    routes: pd.DataFrame  # rate and error by route to the source (``route_summary``)
    frame: pd.DataFrame  # one frame on the map (``snapshot``)
    gains: pd.DataFrame  # gain per scenario, level and phase (``scenario_gains``)


def beta_response(metrics: pd.DataFrame, phase: str, quantity: str) -> pd.DataFrame:
    """Per scheduler, beta and seed: rounds per node per second in ``phase`` (``quantity`` "rate"), or the error over
    the error of the fixed 3 s period of the same seed (``quantity`` "error"), which removes the spread between the
    seeds' topologies."""
    rows = metrics[metrics.phase == phase]
    if quantity == "rate":
        values = rows.rounds / (config.N_NODES * PHASE_BY_KEY[phase].duration)
    else:
        reference = rows[rows.algorithm == REFERENCE_FIXED].set_index("seed").error
        values = rows.error / rows.seed.map(reference)
    return rows[["algorithm", "beta", "seed"]].assign(value=values)


def paper_metrics(episodes: pd.DataFrame) -> dict:
    """The per-episode means of the mobile exports (rates per node per second) as the paper's episode totals:
    rounds per episode, unstable time (outside the tolerance), integrated normalized error and return per node."""
    length = config.EPISODE_LENGTH_S
    return dict(ticks=episodes.roundsPerNodeSecond.mean() * config.N_NODES * length,
                unstable=(1.0 - episodes.withinTolerance.mean()) * length,
                error=episodes.normalizedError.mean() * length,
                reward=episodes.rewardRate.mean() * length)


def _paper_configuration(algorithm: int) -> tuple[str, float] | None:
    if algorithm in PAPER_ALGORITHMS:
        return PAPER_ALGORITHMS[algorithm], np.nan
    if algorithm in PAPER_FIXED:
        return FIXED.key, PAPER_FIXED[algorithm]
    return None


def paper_dataset(variant_key: str, level: float) -> Dataset:
    """The runs of one scenario as the paper's ``Dataset``: one evaluation row per run (method, weight, period,
    seed, means over the evaluation episodes), compared on ``PAPER_METHODS``."""
    variant = VARIANT_BY_KEY[variant_key]
    rows = []
    for (algorithm, beta, run_level, seed), files in sorted(variant_runs(variant).items()):
        configuration = _paper_configuration(algorithm)
        if configuration and seed in variant.seeds and np.isclose(run_level, level) and "episodes" in files:
            episodes = pd.read_csv(files["episodes"])
            method, period = configuration
            rows.append(dict(method=method, weight=beta, period=period, seed=seed,
                             **paper_metrics(episodes[~episodes.training])))
    empty = pd.DataFrame()
    return Dataset(snapshot_view(variant_key, level), empty, pd.DataFrame(rows), empty, empty, PAPER_METHODS)


def snapshot_view(variant_key: str, level: float) -> Path:
    """A folder whose ``spatial`` sub-folder links the snapshot files of the inspected runs of a scenario."""
    view = SNAPSHOT_VIEWS / variant_key / f"V{level}"
    target = view / SNAPSHOTS
    target.mkdir(parents=True, exist_ok=True)
    for link in target.iterdir():
        link.unlink()
    for algorithm in INSPECTED_ALGORITHMS:
        pattern = f"seed{INSPECTED_SEED}_algo{algorithm}_V{level}_*/*.csv"
        for path in sorted((VARIANT_EXPORTS / variant_key / SNAPSHOTS).glob(pattern)):
            (target / path.name).symlink_to(path.resolve())
    return view


def _moving(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of every second but the first, with whether the node moved since the previous second."""
    ordered = frame.sort_values(["nodeId", "tEp"])
    step = ordered.groupby("nodeId")[["x", "y"]].diff()
    return ordered.assign(moving=np.hypot(step.x, step.y) > MOTION_EPSILON_M)[step.x.notna()]


def _attention_at(nodes: pd.DataFrame) -> dict | None:
    positions = nodes[["x", "y"]].to_numpy()
    moving = nodes.moving.to_numpy()
    distances = cdist(positions, positions)
    neighbours = (distances <= COMMUNICATION_RANGE_M) & ~np.eye(len(nodes), dtype=bool)
    attended = cdist(nodes[["bestNbrX", "bestNbrY"]].to_numpy(), positions).argmin(axis=1)
    exposed = ~moving & (neighbours & moving[None, :]).any(axis=1)
    if not exposed.any():
        return None
    chance = (neighbours & moving[None, :]).sum(axis=1)[exposed] / neighbours.sum(axis=1)[exposed]
    return dict(attended=moving[attended[exposed]].mean(), chance=chance.mean(), exposed=int(exposed.sum()),
                pc1_moving=nodes.pc1[moving].mean(), pc1_still=nodes.pc1[~moving].mean())


def attention_on_motion(frame: pd.DataFrame) -> pd.DataFrame:
    """Per second of the GNN attention snapshots: among the still nodes with a moving neighbour, the share whose most
    attended neighbour moves (``attended``) and the share of moving neighbours, which a random choice would match
    (``chance``); and the mean embedding PC1 of the moving and of the still nodes."""
    rows = []
    for time, nodes in _moving(frame).groupby("tEp"):
        stats = _attention_at(nodes.sort_values("nodeId").reset_index(drop=True))
        if stats:
            rows.append(dict(tEp=time, **stats))
    return pd.DataFrame(rows)


def load(partial: bool = False) -> PaperData:
    static_metrics, static_series = load_variant(VARIANT_BY_KEY[STATIC_VARIANT], partial)
    static_metrics = static_metrics[np.isclose(static_metrics.diffusion, STATIC_LEVEL)]
    static_series = static_series[np.isclose(static_series.diffusion, STATIC_LEVEL)]
    region_metrics, region_network = load_variant(VARIANT_BY_KEY[REGION.variant], partial)
    region_metrics = region_metrics[region_metrics.diffusion > STATIC_LEVEL]
    gains = pd.concat([scenario_gains(VARIANT_BY_KEY[STATIC_VARIANT], static_metrics, PHASES),
                       scenario_gains(VARIANT_BY_KEY[REGION.variant], region_metrics, PHASES)], ignore_index=True)
    return PaperData(static_metrics, static_series, region_metrics, region_network, load_region_series(REGION),
                     route_summary(route_table(REGION)), snapshot(REGION), gains)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--partial", action="store_true", help="skip incomplete configurations (previews only)")
    arguments = parser.parse_args()
    from .figures import mobility_paper as figures

    from .figures import mobility_paper_style as paper_figures

    data = load(arguments.partial)
    tables = arguments.out / RECAP
    tables.mkdir(parents=True, exist_ok=True)
    data.gains.to_csv(tables / "gains.csv", index=False)
    data.routes.to_csv(tables / "region_routes.csv", index=False)
    allocation(data.region_series, REGION.windows).to_csv(tables / "region_allocation.csv", index=False)
    for name, (variant, level) in PAPER_SCENARIOS.items():
        paper_figures.render(paper_dataset(variant, level), data, name, arguments.out / name)
    figures.render(data, arguments.out)
    print(f"Saved the mobile figures to {arguments.out}")


if __name__ == "__main__":
    main()
