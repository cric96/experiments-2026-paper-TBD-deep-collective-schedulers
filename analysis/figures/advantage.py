"""The advantage of the learned schedules over the reactive baselines tuned by beta, at the same rounds, in 3D.

For every scenario of the beta study and every beta of a learned method, its gain at matched rounds minus that of the
better tuned baseline at the same rounds (the curves of the baselines interpolated in the logarithm of the rounds):
above zero the learned method is better, below it the baseline is. The comparison is at the same rounds, not at the
same beta: at the same beta the baselines run more rounds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import style
from .campaign import ARCHITECTURE_LABELS, BASELINE_METHODS, PROGRAMS, scenario
from experiments.campaign import LEARNED, grid_of

# A baseline is compared at rounds up to this factor beyond the ones it reaches, with its gain there: at beta 0.2 a
# learned method runs a little fewer rounds than the slowest tuned baseline.
ROUNDS_TOLERANCE = 1.2
# Width of a bar, in units of the beta and scenario grid.
BAR = 0.45
ZERO_PLANE_ALPHA = 0.12
# Cells this far (in colour scale units, from the middle) from zero are dark: their text is white.
DARK_CELL = 0.3
# Elevation and azimuth (degrees) of the view: the bars seen from the front left, the zero plane slightly from above.
VIEW = (30, -50)


def ridge_label(family: str, program: int, nodes: int, speed: float) -> str:
    return f"Motion {speed:g} m/s" if family == "dynamic" else PROGRAMS[program]


def mean_curve(rows: pd.DataFrame, variant: str, algorithm: int) -> pd.DataFrame:
    """The mean rounds (per node and second) and gain (%) of a method at every beta it ran at, ordered by rounds."""
    subset = rows[(rows.variant == variant) & (rows.algorithm == algorithm)]
    means = subset.groupby("beta")[["roundsPerNodeSecond", "gain"]].mean()
    return means.assign(gain=100 * means.gain).sort_values("roundsPerNodeSecond")


def at_rounds(curve: pd.DataFrame, rounds: np.ndarray) -> np.ndarray:
    """The gain of a curve at `rounds`, linear in the logarithm of the rounds; at its ends up to ROUNDS_TOLERANCE
    beyond them, NaN further away."""
    reached = curve.roundsPerNodeSecond.to_numpy()
    low, high = reached.min(), reached.max()
    within = (rounds >= low / ROUNDS_TOLERANCE) & (rounds <= high * ROUNDS_TOLERANCE)
    values = np.interp(np.log(np.clip(rounds, low, high)), np.log(reached), curve.gain.to_numpy())
    return np.where(within, values, np.nan)


def advantage_at_betas(rows: pd.DataFrame, final: str, algorithm: int, baselines: str) -> pd.Series:
    """The gain (points) of a learned method minus that of the better tuned baseline at the rounds of each of its
    betas; NaN where some tuned baseline does not reach those rounds."""
    learned = mean_curve(rows, final, algorithm)
    references = [mean_curve(rows, baselines, reference) for reference in BASELINE_METHODS]
    if learned.empty or any(curve.empty for curve in references):
        return pd.Series(dtype=float)
    rounds = learned.roundsPerNodeSecond.to_numpy()
    best = np.maximum.reduce([at_rounds(curve, rounds) for curve in references])
    return pd.Series(learned.gain.to_numpy() - best, index=learned.index).sort_index()


def advantage_grid(gains: pd.DataFrame, final: str, algorithm: int, scenarios: list[tuple],
                   baselines: str) -> pd.DataFrame:
    """The advantage of a learned method: one row per scenario, one column per beta of its grid."""
    rows = {ridge_label(*where): advantage_at_betas(scenario(gains, *where), final, algorithm, baselines)
            for where in scenarios}
    return pd.DataFrame(rows).T.reindex(columns=grid_of(final)[0])


def advantage_panel(ax, grid: pd.DataFrame, norm: Normalize) -> None:
    """One bar per scenario and beta, rising above the zero plane or sinking below it, coloured by its value."""
    columns, rows = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))
    values = grid.to_numpy().ravel()
    known = np.isfinite(values)
    x, y, z = columns.ravel()[known], rows.ravel()[known], values[known]
    ax.bar3d(x - BAR / 2, y - BAR / 2, np.minimum(z, 0), BAR, BAR, np.abs(z),
             color=mpl.colormaps[style.DIVERGING](norm(z)), edgecolor=style.INK_MUTED, linewidth=0.15, shade=True)
    missing = ~known
    ax.scatter(columns.ravel()[missing], rows.ravel()[missing], np.zeros(missing.sum()), marker="x", s=8,
               color=style.INK_MUTED, depthshade=False)
    plane_x, plane_y = np.meshgrid([-0.5, grid.shape[1] - 0.5], [-0.5, grid.shape[0] - 0.5])
    ax.plot_surface(plane_x, plane_y, np.zeros_like(plane_x), color=style.INK_MUTED, alpha=ZERO_PLANE_ALPHA,
                    linewidth=0)
    ax.set_xticks(range(grid.shape[1]), [f"{beta:g}" for beta in grid.columns])
    ax.set_yticks(range(grid.shape[0]), list(grid.index), ha="left", va="center")
    ax.set(xlim=(-0.5, grid.shape[1] - 0.5), ylim=(-0.5, grid.shape[0] - 0.5), zlim=(norm.vmin, norm.vmax))
    ax.set_xlabel("β of the learned method", labelpad=0)
    ax.set_zlabel("Gain − better tuned baseline\nat the same rounds (points)", labelpad=4)
    ax.tick_params(pad=0)
    ax.view_init(*VIEW)


def advantage_grids(gains: pd.DataFrame, finals: dict[int, str], scenarios: list[tuple],
                    baselines: str) -> tuple[dict[int, pd.DataFrame], Normalize]:
    """The advantage grid of every architecture and a colour (and height) scale symmetric around zero."""
    grids = {algorithm: advantage_grid(gains, finals[algorithm], algorithm, scenarios, baselines)
             for algorithm in LEARNED}
    values = np.concatenate([grid.to_numpy().ravel() for grid in grids.values()])
    finite = values[np.isfinite(values)]
    limit = max(np.abs(finite).max() if finite.size else 0.0, 1.0)
    return grids, Normalize(-limit, limit)


def advantage_3d_figure(gains: pd.DataFrame, out: Path, finals: dict[int, str], scenarios: list[tuple],
                        baselines: str) -> None:
    """One 3D panel per architecture on a shared vertical (and colour) scale."""
    grids, norm = advantage_grids(gains, finals, scenarios, baselines)
    fig = style.canvas("double", 3.4)
    for index, (algorithm, grid) in enumerate(grids.items()):
        ax = fig.add_subplot(1, len(grids), index + 1, projection="3d")
        advantage_panel(ax, grid, norm)
        ax.set_title(ARCHITECTURE_LABELS[algorithm], loc="left")
    colours = mpl.colormaps[style.DIVERGING]
    handles = [Patch(facecolor=colours(0.85), label="learned better"),
               Patch(facecolor=colours(0.15), label="tuned baseline better"),
               Line2D([], [], color=style.INK_MUTED, marker="x", linestyle="none", markersize=3,
                      label="no tuned baseline at those rounds")]
    style.figure_legend(fig, handles, ncol=len(handles))
    style.save(fig, out, "advantage_3d")


def cell_text(value: float) -> str:
    """The advantage in a cell of the map, signed, in whole points; a dash where there is no comparison."""
    if not np.isfinite(value):
        return "–"
    rounded = round(value)
    return f"{rounded:+d}".replace("-", "−") if rounded else "0"


def advantage_map_figure(gains: pd.DataFrame, out: Path, finals: dict[int, str], scenarios: list[tuple],
                         baselines: str) -> None:
    """The grids of the 3D figure, flat: one panel per architecture, a scenario per row, a beta per column, the
    advantage written in every cell."""
    grids, norm = advantage_grids(gains, finals, scenarios, baselines)
    fig, axes = style.figure("double", 2.3, 1, len(grids), sharey=True, layout="constrained")
    for ax, (algorithm, grid) in zip(axes, grids.items()):
        values = grid.to_numpy()
        image = ax.imshow(values, cmap=style.DIVERGING, norm=norm, aspect="auto")
        for (row, column), value in np.ndenumerate(values):
            dark = np.isfinite(value) and abs(norm(value) - 0.5) > DARK_CELL
            ax.text(column, row, cell_text(value), ha="center", va="center", fontsize=6.5,
                    color="white" if dark else style.INK)
        ax.set_xticks(range(grid.shape[1]), [f"{beta:g}" for beta in grid.columns])
        ax.set_yticks(range(grid.shape[0]), list(grid.index))
        ax.set_xlabel("β of the learned method")
        ax.set_title(ARCHITECTURE_LABELS[algorithm], loc="left")
        ax.grid(False)
    fig.colorbar(image, ax=axes, shrink=0.9, label="Gain − better tuned baseline\nat the same rounds (points)")
    style.save(fig, out, "advantage_map")
