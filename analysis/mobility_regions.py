"""Where the rounds go when only a region moves: python -m analysis.mobility_regions.

The runs of the region scenarios export, per node and 2 s frame, the position, the rounds and the integrated absolute
error since the episode start, and whether the node belongs to the moving region: a region that starts moving halfway
through the episode (``events-region``), or half of the nodes moving all the time while the source switches halfway
(``region-moves``). This module splits the round rate and the error of the evaluation episodes between the moving
region and the still nodes, and the still nodes by their route to the source: a fixed period gives every node the same
rate, a scheduler that sees where the motion is should not; and a still node whose gradient flows through a moving
node inherits its error.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from .mobility_static import EVALUATION, EVENT_END_S, FRAME_S, PHASE_BY_KEY, Phase
from .mobility_variants import DEFAULT_OUTPUT, FOCUS_BETA, VARIANT_BY_KEY, variant_runs
from .stats import t_interval

NODE_COLUMNS = ["episode", "time", "node", "rounds", "absoluteError", "region"]
FRAME_COLUMNS = ["episode", "time", "node", "x", "y", "source", "region"]
COMMUNICATION_RANGE_M = 75.0  # the ConnectWithinDistance network model of simulationMobility.yml
# Still nodes by their shortest route to the source: through a moving node (the source included), or only through
# still ones; nodes in a component without the source are cut off.
ROUTES = ("moving", "through", "clear", "cut off")
TOPOLOGY_CARRIER = 5  # a fixed period (3 s): every scheduler of a seed and level sees the same positions
SNAPSHOT_SEED, SNAPSHOT_LEVEL = 1, 1.0  # the run drawn on the map, at its first evaluation episode
SNAPSHOT_WINDOW_S = 20.0  # rate and error of the map: means over the last 20 s
SNAPSHOT_SCHEDULERS = (TOPOLOGY_CARRIER, 0)  # fixed 3 s and the GNN


@dataclass(frozen=True)
class Window:
    key: str
    label: str
    phase: Phase


@dataclass(frozen=True)
class RegionScenario:
    variant: str
    prefix: str  # of the output files
    event: str  # what happens halfway, for the figures
    windows: tuple[Window, ...]  # the main one first
    static: str  # label of the static level
    snapshot_s: float  # time of the map, within the main window
    motion_end_s: float | None = None  # when the region stops, if it does

    @property
    def levels(self) -> tuple[float, ...]:
        return VARIANT_BY_KEY[self.variant].levels


SCENARIOS = (
    RegionScenario("events-region", "region", "region starts\nmoving",
                   (Window("after", "after the event", PHASE_BY_KEY["switch"]),
                    Window("before", "before (40–100 s)", PHASE_BY_KEY["tracking"])), "static\n(no event)", 150.0),
    RegionScenario("region-moves", "region_moves", "source\nswitch",
                   (Window("tracking", "tracking (40–100 s)", PHASE_BY_KEY["tracking"]),
                    Window("switch", "after the switch (100–200 s)", PHASE_BY_KEY["switch"])), "static", 90.0),
)
# The final experiment (``analysis.mobility_paper``): half of the nodes move from 100 s to 150 s.
FINAL_REGION = RegionScenario("final-region", "region", "region moves",
                              (Window("moving", "while it moves (100–150 s)", PHASE_BY_KEY["event"]),
                               Window("before", "before (40–100 s)", PHASE_BY_KEY["tracking"])), "static", 140.0,
                              EVENT_END_S)
SCENARIO_BY_VARIANT = {scenario.variant: scenario for scenario in (*SCENARIOS, FINAL_REGION)}


def positions_file(full: Path) -> Path:
    return full.with_name(full.name.replace("_full.csv", "_positions.csv"))


def group_series(positions: Path) -> pd.DataFrame:
    """Per group (moving region or not) and frame: rounds per node per second and mean absolute error (m).

    A node belongs to the region of an episode if it is flagged at any frame of that episode (the flag is set when
    the region starts moving); the rates are averaged over the evaluation episodes.
    """
    rows = pd.read_csv(positions, usecols=NODE_COLUMNS)
    rows = rows[rows.episode.isin(EVALUATION)]
    members = rows.groupby(["episode", "node"]).region.any().rename("moving")
    rows = rows.join(members, on=["episode", "node"])
    totals = rows.groupby(["episode", "moving", "time"])[["rounds", "absoluteError"]].mean().sort_index()
    steps = totals.groupby(level=["episode", "moving"]).diff().dropna() / FRAME_S
    series = steps.groupby(level=["moving", "time"]).mean().reset_index()
    return series.rename(columns={"rounds": "rate", "absoluteError": "error"})


def _runs(scenario: RegionScenario, beta: float):
    """The full exports of the scenario at ``beta`` (fixed periods have no beta of their own), by run."""
    variant = VARIANT_BY_KEY[scenario.variant]
    for (algorithm, run_beta, level, seed), files in sorted(variant_runs(variant).items()):
        if seed in variant.seeds and np.isclose(run_beta, beta) and "full" in files:
            yield algorithm, level, seed, files["full"]


def load(scenario: RegionScenario = SCENARIOS[0], beta: float = FOCUS_BETA) -> pd.DataFrame:
    """The group series of every run of the scenario at ``beta``."""
    frames = [group_series(positions_file(full)).assign(algorithm=algorithm, level=level, seed=seed)
              for algorithm, level, seed, full in _runs(scenario, beta)]
    return pd.concat(frames, ignore_index=True)


def _window_mean(series: pd.DataFrame, phase: Phase) -> pd.Series:
    inside = series[(series.time > phase.start) & (series.time <= phase.end)]
    return inside.groupby(["algorithm", "level", "seed", "moving"]).rate.mean()


def allocation(series: pd.DataFrame, windows: tuple[Window, ...] = SCENARIOS[0].windows) -> pd.DataFrame:
    """Per scheduler, level and window: round rate of the moving region over the still nodes."""
    records = []
    for window in windows:
        rates = _window_mean(series, window.phase).unstack("moving")
        if True not in rates or False not in rates:
            continue
        ratio = (rates[True] / rates[False]).dropna()
        for (algorithm, level), values in ratio.groupby(level=["algorithm", "level"]):
            interval = t_interval(values.to_numpy())
            records.append(dict(algorithm=algorithm, level=level, window=window.key, mean=interval.mean,
                                low=interval.low, high=interval.high, seeds=interval.n))
    return pd.DataFrame(records)


def shortest_paths(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the source and parent (row index, negative for none) of every node of one frame."""
    positions = frame[["x", "y"]].to_numpy(dtype=float)
    distances = cdist(positions, positions)
    links = csr_matrix(np.where(distances <= COMMUNICATION_RANGE_M, distances, 0.0))
    source = int(np.flatnonzero(frame.source.to_numpy(dtype=bool))[0])
    return dijkstra(links, indices=source, return_predecessors=True)


def route_classes(frame: pd.DataFrame) -> np.ndarray:
    """The route class (``ROUTES``) of every node of one frame, in the order of its rows."""
    moving = frame.region.to_numpy(dtype=bool)
    reach, parents = shortest_paths(frame)
    through = np.zeros(len(frame), dtype=bool)
    for node in np.argsort(reach):  # parents first
        parent = parents[node]
        if parent >= 0:
            through[node] = moving[parent] or through[parent]
    still = np.where(through, "through", "clear")
    return np.where(moving, "moving", np.where(np.isfinite(reach), still, "cut off"))


def routes(positions: Path) -> pd.DataFrame:
    """The route class of every node at every frame of the evaluation episodes."""
    rows = pd.read_csv(positions, usecols=FRAME_COLUMNS)
    rows = rows[rows.episode.isin(EVALUATION)].sort_values(["episode", "time", "node"])
    classes = [route_classes(frame) for _, frame in rows.groupby(["episode", "time"], sort=False)]
    return rows[["episode", "time", "node"]].assign(route=np.concatenate(classes))


def node_steps(positions: Path) -> pd.DataFrame:
    """Per node and frame of the evaluation episodes: rounds per second and mean absolute error since the last one."""
    rows = pd.read_csv(positions, usecols=NODE_COLUMNS)
    rows = rows[rows.episode.isin(EVALUATION)].sort_values(["episode", "node", "time"])
    steps = rows.groupby(["episode", "node"])[["rounds", "absoluteError"]].diff() / FRAME_S
    return rows[["episode", "time", "node"]].assign(rate=steps.rounds, error=steps.absoluteError).dropna()


def route_means(steps: pd.DataFrame, classes: pd.DataFrame, phase: Phase) -> pd.DataFrame:
    """Mean rate and error of each route class in ``phase``, and its share of the node-frames.

    A frame step is classified by the route at its end (the topology of the step it closes).
    """
    joined = steps.merge(classes, on=["episode", "time", "node"])
    inside = joined[(joined.time > phase.start) & (joined.time <= phase.end)]
    means = inside.groupby("route")[["rate", "error"]].mean()
    return means.assign(share=inside.route.value_counts(normalize=True)).reindex(list(ROUTES)).reset_index()


def route_table(scenario: RegionScenario, beta: float = FOCUS_BETA) -> pd.DataFrame:
    """Per scheduler, level, seed, window and route class: mean rate, error and share."""
    runs = list(_runs(scenario, beta))
    topologies = {(level, seed): routes(positions_file(full))
                  for algorithm, level, seed, full in runs if algorithm == TOPOLOGY_CARRIER}
    frames = []
    for algorithm, level, seed, full in runs:
        steps = node_steps(positions_file(full))
        for window in scenario.windows:
            means = route_means(steps, topologies[(level, seed)], window.phase)
            frames.append(means.assign(algorithm=algorithm, level=level, seed=seed, window=window.key))
    return pd.concat(frames, ignore_index=True)


def route_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% interval over seeds of every quantity of ``route_table``."""
    records = []
    for (algorithm, level, window, route), rows in table.groupby(["algorithm", "level", "window", "route"]):
        for quantity in ("rate", "error", "share"):
            interval = t_interval(rows[quantity].dropna().to_numpy())
            records.append(dict(algorithm=algorithm, level=level, window=window, route=route, quantity=quantity,
                                mean=interval.mean, low=interval.low, high=interval.high, seeds=interval.n))
    return pd.DataFrame(records)


def _window_totals(positions: Path, episode: int, end: float) -> pd.DataFrame:
    """Per node: rounds per second and mean absolute error over the ``SNAPSHOT_WINDOW_S`` before ``end``."""
    rows = pd.read_csv(positions, usecols=NODE_COLUMNS)
    rows = rows[(rows.episode == episode) & rows.time.isin([end - SNAPSHOT_WINDOW_S, end])]
    totals = rows.pivot(index="node", columns="time", values=["rounds", "absoluteError"])
    change = (totals.xs(end, axis=1, level="time") - totals.xs(end - SNAPSHOT_WINDOW_S, axis=1, level="time"))
    return (change / SNAPSHOT_WINDOW_S).rename(columns={"rounds": "rate", "absoluteError": "error"})


def snapshot(scenario: RegionScenario) -> pd.DataFrame:
    """The map of one frame: per node position, flags, route class, parent position, and the rate and error of
    each of ``SNAPSHOT_SCHEDULERS`` (``rate_<algorithm>``, ``error_<algorithm>``)."""
    runs = {algorithm: full for algorithm, level, seed, full in _runs(scenario, FOCUS_BETA)
            if seed == SNAPSHOT_SEED and np.isclose(level, SNAPSHOT_LEVEL)}
    episode, end = EVALUATION[0], scenario.snapshot_s
    frame = pd.read_csv(positions_file(runs[TOPOLOGY_CARRIER]), usecols=FRAME_COLUMNS)
    frame = frame[(frame.episode == episode) & np.isclose(frame.time, end)].sort_values("node").reset_index(drop=True)
    _, parents = shortest_paths(frame)
    linked = np.where(parents >= 0, parents, np.arange(len(frame)))
    frame = frame.assign(route=route_classes(frame), parent_x=frame.x.to_numpy()[linked],
                         parent_y=frame.y.to_numpy()[linked])
    for algorithm in SNAPSHOT_SCHEDULERS:
        totals = _window_totals(positions_file(runs[algorithm]), episode, end).add_suffix(f"_{algorithm}")
        frame = frame.join(totals, on="node")
    return frame


def available(scenario: RegionScenario) -> bool:
    return any(algorithm == TOPOLOGY_CARRIER for algorithm, *_ in _runs(scenario, FOCUS_BETA))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    from .figures import mobility_regions as figures

    arguments.out.mkdir(parents=True, exist_ok=True)
    for scenario in filter(available, SCENARIOS):
        series = load(scenario)
        ratios = allocation(series, scenario.windows)
        table = route_table(scenario)
        summary = route_summary(table)
        series.to_csv(arguments.out / f"{scenario.prefix}_series.csv", index=False)
        ratios.to_csv(arguments.out / f"{scenario.prefix}_allocation.csv", index=False)
        summary.to_csv(arguments.out / f"{scenario.prefix}_routes.csv", index=False)
        figures.render(scenario, series, ratios, summary, snapshot(scenario), arguments.out)
        print(f"Saved {scenario.variant} ({series.seed.nunique()} seeds) to {arguments.out}")


if __name__ == "__main__":
    main()
