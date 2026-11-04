"""Loading of every export, plus the derived results computed once and shared."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from . import results
from .data import episode_table, evaluation_summary, final_episode_traces, generalization_table
from .io import load_runs
from .methods import LEARNED, Method

# Sub-folders of the exports directory.
EPISODE_DIR = "simulation/episode"  # one row per episode, every method
FULL_DIRS = ("simulation/full", "simulationDeepOffline/full", "simulationDeepOfflineGNN/full")  # 2 s traces
GENERALIZATION_DIR = "generalization/episode"
EMBEDDINGS_DIR = "embeddings"
SPATIAL_DIR = "spatial"


@dataclass
class Dataset:
    """Everything the figures, tables and summary need, loaded once."""

    exports_dir: Path
    episodes: pd.DataFrame  # main experiment, one row per (run, episode)
    evaluation: pd.DataFrame  # main experiment, one row per run (mean of evaluation episodes)
    traces: pd.DataFrame  # main experiment, final-episode 2 s traces
    generalization: pd.DataFrame  # generalization experiment, one row per (run, episode)
    methods: tuple[Method, ...] = LEARNED  # the learned methods compared with the fixed periods

    @property
    def embeddings_dir(self) -> Path:
        return self.exports_dir / EMBEDDINGS_DIR

    @property
    def spatial_dir(self) -> Path:
        return self.exports_dir / SPATIAL_DIR

    # Derived results (bootstrap-heavy ones are computed lazily, once).
    @cached_property
    def gains(self) -> pd.DataFrame:
        return results.frontier_gains(self.evaluation, self.methods)

    @cached_property
    def significance(self) -> pd.DataFrame:
        return results.significance(self.evaluation)

    @cached_property
    def stability(self) -> pd.DataFrame:
        return results.stability(self.evaluation)

    @cached_property
    def switch(self) -> pd.DataFrame:
        return results.switch_responses(self.traces)

    @cached_property
    def generalization_points(self) -> pd.DataFrame:
        return results.generalization_points(self.generalization)

    @cached_property
    def generalization_gains(self) -> pd.DataFrame:
        return results.generalization_gains(self.generalization_points)


def load_dataset(exports_dir: Path) -> Dataset:
    """Parse every export needed by the analysis."""
    episodes = episode_table(load_runs(exports_dir / EPISODE_DIR))
    full_runs = [run for folder in FULL_DIRS for run in load_runs(exports_dir / folder)]
    return Dataset(
        exports_dir=exports_dir,
        episodes=episodes,
        evaluation=evaluation_summary(episodes),
        traces=final_episode_traces(full_runs),
        generalization=generalization_table(load_runs(exports_dir / GENERALIZATION_DIR)),
    )
