"""The ladder of variants of the mobile experiment (``analysis.mobility_variants``).

* ``variants_ladder``: one panel per question, the cumulative steps in order on the x axis, mean and 95% interval over
  seeds for the two learners and the heuristic;
* ``variants_phase_gains``: where along the ladder the gain over the fixed periods lives (phases and levels), on the
  whole error and on the nodes connected to the source;
* ``scenario_gains``: the final configuration trained and evaluated in other scenarios (events, half of the nodes
  moving), gain per level and phase.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter

from .. import style
from ..mobility_pareto import METHODS
from ..mobility_variants import (CONNECTED, HEURISTIC, LADDER_END, LEARNERS, SCENARIO_PHASES, VARIANT_BY_KEY,
                                VARIANTS)

ALGORITHMS = (*LEARNERS, HEURISTIC)
OFFSET = 0.16  # horizontal shift between the methods drawn at the same variant
RATIO_TICKS = (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0)
PANELS = (
    ("gain_episode", "Gain, whole episode (level 1)", "gain"),
    ("gain_post_focus", "Gain after 40 s (level 1)", "gain"),
    ("gain_post_static", "Gain after 40 s (static)", "gain"),
    ("beta_range", "Rounds, β = 1 / β = 0.5 (level 1)", "ratio"),
    ("level_response", "Rounds, fastest / static level", "ratio"),
    ("reaction_focus", "Rate after / before switch (level 1)", "ratio"),
)
HEATMAP_COLUMNS = (
    ("gain_episode", "Whole\nepisode"),
    ("gain_bootstrap", "Start*\n0–40 s"),  # bootstrap from scratch, or a new source with the warm start
    ("gain_tracking", "Tracking\n40–100 s"),
    ("gain_switch", "Switch\n100–200 s"),
    ("gain_post_static", "Static\n40–200 s"),
    ("gain_post_focus", "Level 1\n40–200 s"),
    ("gain_post_fastest", "Fastest\n40–200 s"),
)
HEATMAP_GROUPS = (0.5, 3.5)  # column breaks: whole episode | phases at level 1 | after the bootstrap per level
# Last rows of the Brownian variants and of the cold-start ones.
SCENARIO_ENDS = ("short-retention", "waypoint-mobility")
LADDER = VARIANTS[:[variant.key for variant in VARIANTS].index(LADDER_END) + 1]
SCENARIO_ROWS = (  # (variant, scenario, training)
    ("dependants", "Every node moves", "trained: every node moving"),
    ("events-switch", "Every node moves", "trained: the two events"),
    ("events-region", "A region starts moving at 100 s", "trained: the two events"),
    ("region-moves", "Half of the nodes move", "trained: half moving"),
)
SCENARIO_LEVELS = ("Static", "1 m/s", "2 m/s")
PHASE_HEADERS = {"bootstrap": "0–40", "tracking": "40–100", "switch": "100–200"}
UNDETERMINED = 1.0  # a gain whose 95% interval is wider than ±100%: the reference error is close to zero
GAIN_LIMIT = 0.3  # colour saturation of the heatmaps (the numbers show the full value)
FADED = 0.3  # colour strength of the cells whose interval crosses zero


def _panel(ax, summary: pd.DataFrame, quantity: str, kind: str, variants: list) -> None:
    keys = [variant.key for variant in variants]
    for index, algorithm in enumerate(ALGORITHMS):
        rows = summary[(summary.algorithm == algorithm) & (summary.quantity == quantity)].set_index("variant")
        rows = rows.reindex(keys)
        x = np.arange(len(keys)) + (index - 1) * OFFSET
        means = rows["mean"].to_numpy(dtype=float)
        errors = [means - rows["low"].to_numpy(dtype=float), rows["high"].to_numpy(dtype=float) - means]
        ax.errorbar(x, means, yerr=errors, linewidth=0.8, elinewidth=0.7,
                    **style.line_style(METHODS[algorithm], alpha=0.55 if algorithm == HEURISTIC else 1.0))
    ax.axhline(0.0 if kind == "gain" else 1.0, color=style.INK_MUTED, linewidth=0.7, zorder=0)
    if kind == "gain":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(RATIO_TICKS))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(lambda value, _: f"×{value:g}")
    ax.set_xticks(range(len(keys)), [variant.label for variant in variants], rotation=55, ha="right",
                  rotation_mode="anchor", fontsize=6)
    ax.set_xlim(-0.5, len(keys) - 0.5)


def _ladder_variants(summary: pd.DataFrame) -> list:
    return [variant for variant in LADDER if variant.key in set(summary.variant)]


def ladder(summary: pd.DataFrame, out: Path) -> None:
    variants = _ladder_variants(summary)
    fig, axes = style.figure("double", height=4.8, nrows=2, ncols=3, layout="constrained")
    for letter, ax, (quantity, title, kind) in zip("abcdef", axes.flat, PANELS):
        _panel(ax, summary, quantity, kind, variants)
        ax.set_title(f"({letter}) {title}", loc="left", fontsize=7, pad=3)
    style.figure_legend(fig, [style.legend_handle(METHODS[algorithm]) for algorithm in ALGORITHMS], ncol=3)
    style.save(fig, out, "variants_ladder")


def _cell_colours(cells: pd.DataFrame, norm: TwoSlopeNorm) -> np.ndarray:
    """RGB of every cell; a cell whose interval crosses zero is drawn faded towards white, an undetermined one white."""
    colours = mpl.colormaps["RdBu"](norm(np.nan_to_num(cells["mean"].to_numpy(dtype=float))))[:, :3]
    significant = ((cells["low"] > 0) | (cells["high"] < 0)).to_numpy()[:, None]
    undetermined = ((cells["high"] - cells["low"]) / 2 > UNDETERMINED).to_numpy()[:, None]
    return np.where(undetermined, 1.0, np.where(significant, colours, 1 - FADED * (1 - colours)))


def _draw_cells(ax, cells: pd.DataFrame, shape: tuple[int, int], norm: TwoSlopeNorm, column_breaks=(),
                row_breaks=()) -> None:
    """A heatmap of gains: one cell per row of ``cells`` (row-major), the value written in it, white group breaks."""
    ax.imshow(_cell_colours(cells, norm).reshape(*shape, 3), aspect="auto")
    for index, cell in enumerate(cells.itertuples()):
        row, column = divmod(index, shape[1])
        significant = cell.low > 0 or cell.high < 0
        undetermined = (cell.high - cell.low) / 2 > UNDETERMINED
        text = "—" if not np.isfinite(cell.mean) else "n.d." if undetermined else f"{cell.mean * 100:+.0f}%"
        strong = significant and abs(cell.mean) > GAIN_LIMIT * 0.7
        ax.text(column, row, text, ha="center", va="center", fontsize=5.5,
                color="white" if strong else (style.INK if significant else style.INK_MUTED),
                fontweight="bold" if significant else "normal")
    for column in column_breaks:
        ax.axvline(column, color="white", linewidth=2.0)
    for row in row_breaks:
        ax.axhline(row, color="white", linewidth=2.0)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _heatmap(ax, summary: pd.DataFrame, algorithm: int, prefix: str, variants: list, norm: TwoSlopeNorm) -> None:
    indexed = summary[summary.algorithm == algorithm].set_index(["variant", "quantity"])
    cells = indexed.reindex([(variant.key, prefix + quantity) for variant in variants for quantity, _ in HEATMAP_COLUMNS])
    keys = [variant.key for variant in variants]
    breaks = [keys.index(end) + 0.5 for end in SCENARIO_ENDS if end in keys[:-1]]
    _draw_cells(ax, cells, (len(variants), len(HEATMAP_COLUMNS)), norm, HEATMAP_GROUPS, breaks)


def _gain_colourbar(fig, axes, norm: TwoSlopeNorm) -> None:
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu"), ax=axes, shrink=0.5, extend="both")
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    bar.set_label("Error reduction at equal rounds vs. fixed periods\n(blue: better; faded: 95% CI crosses 0)",
                  fontsize=6.5)


def phase_gains(summary: pd.DataFrame, out: Path) -> None:
    """Gain over the fixed periods along the ladder: rows variants, columns phases (level 1) and levels (40-200 s)."""
    variants = _ladder_variants(summary)
    fig, axes = style.figure("double", height=5.0, nrows=2, ncols=2, sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-GAIN_LIMIT, vcenter=0.0, vmax=GAIN_LIMIT)
    estimates = (("", "whole error"), (CONNECTED, "nodes connected to the source"))
    for row, (prefix, estimate) in enumerate(estimates):
        for column, algorithm in enumerate(LEARNERS):
            ax = axes[row, column]
            _heatmap(ax, summary, algorithm, prefix, variants, norm)
            ax.set_title(f"({'abcd'[row * 2 + column]}) {METHODS[algorithm].label}, {estimate}", loc="left",
                         fontsize=7, pad=3)
    axes[0, 0].set_yticks(range(len(variants)), [variant.label for variant in variants], fontsize=6.5)
    for ax in axes[1]:
        ax.set_xticks(range(len(HEATMAP_COLUMNS)), [label for _, label in HEATMAP_COLUMNS], fontsize=6)
    _gain_colourbar(fig, axes, norm)
    style.save(fig, out, "variants_phase_gains")


def _scenario_cells(table: pd.DataFrame, rows: list, columns: list) -> pd.DataFrame:
    """The cells of the scenario heatmap, row-major; ``columns`` are (level index, phase) pairs."""
    cells = []
    for variant, *_ in rows:
        levels = VARIANT_BY_KEY[variant].levels
        selected = table[table.variant == variant].set_index(["level", "phase"])
        cells.append(selected.reindex([(levels[level], phase) for level, phase in columns])[["mean", "low", "high"]])
    return pd.concat(cells)


def scenario_gains(table: pd.DataFrame, out: Path) -> None:
    """Gain of the final configuration per scenario (rows) and level x phase (columns), GNN and MLP."""
    rows = [row for row in SCENARIO_ROWS if row[0] in set(table.variant)]
    columns = [(level, phase) for level in range(len(SCENARIO_LEVELS)) for phase in SCENARIO_PHASES]
    fig, axes = style.figure("double", height=3.6, nrows=2, ncols=2, sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-GAIN_LIMIT, vcenter=0.0, vmax=GAIN_LIMIT)
    for row, (estimate, name) in enumerate((("whole", "whole error"), (CONNECTED, "nodes connected to the source"))):
        for column, algorithm in enumerate(LEARNERS):
            ax = axes[row, column]
            cells = _scenario_cells(table[(table.algorithm == algorithm) & (table.estimate == estimate)], rows,
                                    columns)
            breaks = [index * len(SCENARIO_PHASES) - 0.5 for index in range(1, len(SCENARIO_LEVELS))]
            _draw_cells(ax, cells, (len(rows), len(columns)), norm, breaks)
            ax.set_title(f"({'abcd'[row * 2 + column]}) {METHODS[algorithm].label}, {name}", loc="left", fontsize=7,
                         pad=3)
    axes[0, 0].set_yticks(range(len(rows)), [f"{scenario}\n({training})" for _, scenario, training in rows])
    for ax in axes[:, 0]:
        ax.tick_params(axis="y", labelsize=6)
    for ax in axes[1]:
        ax.set_xticks(range(len(columns)), [PHASE_HEADERS[phase] for _, phase in columns], fontsize=5.5)
        for index, level in enumerate(SCENARIO_LEVELS):
            ax.text((index + 0.5) * len(SCENARIO_PHASES) - 0.5, -0.11, level, ha="center", va="top", fontsize=6.5,
                    color=style.INK, transform=ax.get_xaxis_transform())
    fig.supxlabel("Phase (s) per level: a new source at 0 s, the event at 100 s (n.d.: 95% interval wider than "
                  "±100%, the reference error is ≈ 0)", fontsize=6)
    _gain_colourbar(fig, axes, norm)
    style.save(fig, out, "scenario_gains")


def render(summary: pd.DataFrame, scenarios: pd.DataFrame, out: Path) -> None:
    style.apply()
    ladder(summary, out)
    phase_gains(summary, out)
    if not scenarios.empty:
        scenario_gains(scenarios, out)
