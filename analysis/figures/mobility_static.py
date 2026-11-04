"""Figures of the static vs. mobile evaluation of the same policies (``analysis.mobility_static``).

Mobile-trained policies are drawn with solid lines and filled markers, static-trained ones with dashed
lines and open markers of the same architecture colour; the fixed periods are the grey frontier.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator, PercentFormatter

from .. import config, style
from ..mobility_pareto import FIXED_PERIODS, METHODS
from ..mobility_static import BOOTSTRAP_END_S, MOBILE_DIFFUSION, PHASE_BY_KEY, STATIC_DIFFUSION, fixed_reference
from ..stats import t_band, t_interval

MOBILE_TRAINED = (0, 1)
STATIC_TRAINED = (9, 8)
ORDER = (*MOBILE_TRAINED, *STATIC_TRAINED, 2)  # learned policies first, the heuristic last
SCENARIOS = {STATIC_DIFFUSION: "Static network ($D=0$)", MOBILE_DIFFUSION: "Mobile network ($D=1$ m$^2$/s)"}
FOCUS_BETA = config.FOCUS_WEIGHT
GAIN_LIMIT = 0.6  # colour saturation of the gain heatmaps (the numbers show the full value)
HEATMAP_PHASES = ("episode", "bootstrap", "tracking", "switch")
DYNAMICS_FIXED = (3, 6)  # fixed 1 s (the rate of static-trained policies on the move) and 4 s (mobile-trained)
MARK_EVERY = 10  # one marker every 20 s on the 2 s series
RATE_SMOOTHING_FRAMES = 2  # 4 s moving average: hides the phase-locked 2 s aliasing of ~4 s periods
ERROR_FLOOR_M = 0.1  # converged static gradients are exact up to float noise: draw them on this floor
ERROR_RANGE_M = (ERROR_FLOOR_M * 0.6, 900.0)
RATE_RANGE = (0.05, 2.5)
BOOTSTRAP_SHADE = "#ecebe6"


def _style(algorithm: int, **overrides) -> dict:
    method = METHODS[algorithm]
    static = algorithm in STATIC_TRAINED
    kwargs = style.line_style(
        method,
        linestyle="--" if static else "-",
        markerfacecolor="white" if static else method.color,
        markeredgecolor=method.color if static else "white",
        markeredgewidth=0.7 if static else 0.4,
    )
    kwargs.update(overrides)
    return kwargs


def _handles(algorithms=ORDER, fixed: bool = True) -> list[Line2D]:
    handles = [Line2D([], [], **_style(algorithm)) for algorithm in algorithms]
    if fixed:
        handles.append(Line2D([], [], **style.fixed_period_style(label="Fixed period (0.5–5 s)")))
    return handles


def _thousands(value: float, _position=None) -> str:
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _log_rounds_axis(ax, axis: str = "x") -> None:
    target = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    target.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    target.set_major_formatter(FuncFormatter(_thousands))
    target.set_minor_locator(NullLocator())


def _gain_axis(ax) -> None:
    ax.axhline(0, color=style.INK_SECONDARY, linewidth=0.6, linestyle="--", zorder=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


# --- Pareto frontiers: static vs. mobile, whole episode vs. after the bootstrap ------------------------


def _fixed_frontier(ax, data: pd.DataFrame, metric: str) -> None:
    fixed = fixed_reference(data).groupby("period")[["rounds", metric]].mean().sort_values("rounds")
    ax.plot(fixed.rounds, fixed[metric], **style.fixed_period_style(label="Fixed period (0.5–5 s)"))
    # Below-left of the markers: the adaptive points sit on or above the frontier.
    for period, point in fixed.iterrows():
        style.direct_label(ax, point.rounds, point[metric], f"{period:g} s", dx=-2, dy=-3, ha="right", va="top")


def _beta_curve(ax, data: pd.DataFrame, algorithm: int, metric: str) -> None:
    """Seed means per beta (beta increasing along the line); the paired uncertainty is in the gain figures."""
    means = data[data.algorithm == algorithm].groupby("beta")[["rounds", metric]].mean()
    ax.plot(means.rounds, means[metric], **_style(algorithm, linewidth=0.9))


def frontiers(metrics: pd.DataFrame, out: Path) -> None:
    """Integrated error vs. rounds: the whole episode (top) and the same runs after the bootstrap (bottom)."""
    fig, axes = style.figure(height=4.4, nrows=2, ncols=2)
    for column, (diffusion, title) in enumerate(SCENARIOS.items()):
        for row, phase in enumerate(("episode", "post_bootstrap")):
            ax = axes[row, column]
            data = metrics[(metrics.diffusion == diffusion) & (metrics.phase == phase)]
            _fixed_frontier(ax, data, "error")
            for algorithm in ORDER:
                _beta_curve(ax, data, algorithm, "error")
            _log_rounds_axis(ax)
            ax.set_xlabel(f"Rounds in {PHASE_BY_KEY[phase].start:g}–{PHASE_BY_KEY[phase].end:g} s (log scale)")
            ax.set_ylabel(f"Integrated error (s)\n{PHASE_BY_KEY[phase].label.lower()}" if column == 0
                          else "Integrated error (s)")
            ax.set_title(title if row == 0 else "", fontsize=8)
            style.panel_label(ax, "abcd"[row * 2 + column])
    axes[0, 1].text(0.99, 0.98, r"lines join $\beta$ = 0.5, 0.8, 0.95, 0.99, 1", transform=axes[0, 1].transAxes,
                    ha="right", va="top", fontsize=6, color=style.INK_MUTED)
    style.figure_legend(fig, _handles(), ncol=3)
    style.save(fig, out, "frontiers_static_vs_mobile")


# --- intra-episode dynamics -----------------------------------------------------------------------------


def _mean_series(data: pd.DataFrame, algorithm: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    rows = data[data.algorithm == algorithm].pivot_table(index="seed", columns="t", values=metric)
    mean, _, _ = t_band(rows.to_numpy())
    return rows.columns.to_numpy(), mean


def _rate_series(data: pd.DataFrame, algorithm: int) -> tuple[np.ndarray, np.ndarray]:
    times, rate = _mean_series(data, algorithm, "rate")
    return times, pd.Series(rate).rolling(RATE_SMOOTHING_FRAMES, min_periods=1).mean().to_numpy()


def _error_series(data: pd.DataFrame, algorithm: int) -> tuple[np.ndarray, np.ndarray]:
    times, error = _mean_series(data, algorithm, "absoluteError")
    return times, np.maximum(error, ERROR_FLOOR_M)


def _phases_background(ax, label: bool) -> None:
    ax.axvspan(0, BOOTSTRAP_END_S, color=BOOTSTRAP_SHADE, linewidth=0, zorder=0)
    style.mark_source_switch(ax, label=label, label_y=0.97)
    if label:
        ax.text(BOOTSTRAP_END_S / 2, 0.97, "bootstrap", transform=ax.get_xaxis_transform(), fontsize=6,
                color=style.INK_MUTED, ha="center", va="top")


def dynamics(series: pd.DataFrame, out: Path) -> None:
    """Round rate and mean error through the evaluation episodes (beta = 0.95), static vs. mobile."""
    fig, axes = style.figure(height=3.5, nrows=2, ncols=2, sharex=True, sharey="row")
    focus = series[np.isclose(series.beta, FOCUS_BETA)]
    for column, (diffusion, title) in enumerate(SCENARIOS.items()):
        data = focus[focus.diffusion == diffusion]
        rate_ax, error_ax = axes[:, column]
        for algorithm in DYNAMICS_FIXED:
            period = FIXED_PERIODS[algorithm]
            grey = style.fixed_period_color(period)
            rate_ax.axhline(1 / period, color=grey, linestyle="--", linewidth=0.9, zorder=2)
            style.direct_label(rate_ax, config.EPISODE_LENGTH_S, 1 / period, f"{period:g} s", dx=2, dy=4,
                               va="bottom", ha="right")
            error_ax.plot(*_error_series(data, algorithm), color=grey, linestyle="--", linewidth=0.9, zorder=2)
        for algorithm in (*MOBILE_TRAINED, *STATIC_TRAINED):
            line = _style(algorithm, linewidth=1.0, markevery=MARK_EVERY)
            rate_ax.plot(*_rate_series(data, algorithm), **line)
            error_ax.plot(*_error_series(data, algorithm), **line)
        for ax in (rate_ax, error_ax):
            _phases_background(ax, label=ax is rate_ax)
        rate_ax.set_title(title, fontsize=8)
        rate_ax.set_yscale("log")
        error_ax.set_yscale("log")
        error_ax.set_xlabel("Time within the evaluation episode (s)")
    axes[0, 0].set_ylim(*RATE_RANGE)
    axes[1, 0].set_ylim(*ERROR_RANGE_M)
    axes[0, 0].set_ylabel("Rounds per node per second\n(4 s moving average)")
    axes[1, 0].set_ylabel(f"Mean absolute error (m)\n(floor {ERROR_FLOOR_M:g} m = exact)")
    axes[1, 0].set_xlim(0, config.EPISODE_LENGTH_S)
    for index, ax in enumerate(axes.flat):
        style.panel_label(ax, "abcd"[index])
    handles = _handles((*MOBILE_TRAINED, *STATIC_TRAINED), fixed=False)
    handles += [Line2D([], [], color=style.fixed_period_color(FIXED_PERIODS[algorithm]), linestyle="--",
                       linewidth=0.9, label=f"Fixed {FIXED_PERIODS[algorithm]:g} s") for algorithm in DYNAMICS_FIXED]
    style.figure_legend(fig, handles, ncol=3)
    style.save(fig, out, "dynamics_static_vs_mobile")


# --- where the gain comes from: phases x scenarios ------------------------------------------------------


UNDEFINED_TEXT = {"converged": "no error\nleft", "outside_range": "outside\nfixed range"}


def _gain_cell(ax, row: int, column: int, cell: pd.Series) -> None:
    if not np.isfinite(cell["mean"]):
        ax.text(column, row, UNDEFINED_TEXT.get(cell["undefined"], "—"), ha="center", va="center", fontsize=5.5,
                color=style.INK_MUTED)
        return
    crosses_zero = not (cell["low"] > 0 or cell["high"] < 0)
    strong = abs(cell["mean"]) > GAIN_LIMIT * 0.6 and not crosses_zero
    text = f"{cell['mean']:.0%}\n[{cell['low'] * 100:.0f}, {cell['high'] * 100:.0f}]"
    ax.text(column, row, text, ha="center", va="center", fontsize=5.5,
            color="white" if strong else style.INK, fontweight="normal" if crosses_zero else "bold")


def _gain_heatmap(ax, gains: pd.DataFrame, metric: str, diffusion: float):
    cells = gains[(gains.gain == metric) & (gains.diffusion == diffusion) & np.isclose(gains.beta, FOCUS_BETA)]
    cells = cells.set_index(["algorithm", "phase"])
    matrix = np.array([[cells.loc[(algorithm, phase), "mean"] for phase in HEATMAP_PHASES] for algorithm in ORDER])
    norm = TwoSlopeNorm(vmin=-GAIN_LIMIT, vcenter=0, vmax=GAIN_LIMIT)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu", norm=norm, aspect="auto")
    for row, algorithm in enumerate(ORDER):
        for column, phase in enumerate(HEATMAP_PHASES):
            _gain_cell(ax, row, column, cells.loc[(algorithm, phase)])
    ax.set_xticks(range(len(HEATMAP_PHASES)),
                  [PHASE_BY_KEY[phase].label.replace(" (", "\n(") for phase in HEATMAP_PHASES], fontsize=6)
    ax.set_yticks(range(len(ORDER)), [METHODS[algorithm].label for algorithm in ORDER], fontsize=6.5)
    ax.axvline(0.5, color="white", linewidth=2.5)  # the whole episode is the sum of the phases on its right
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def phase_gains(gains: pd.DataFrame, out: Path) -> None:
    """Paired gains over the fixed-period frontier, per phase: rows metric, columns scenario (beta = 0.95)."""
    fig, axes = style.figure(height=4.6, nrows=2, ncols=2, sharey=True)
    titles = {"error": "Integrated-error reduction at equal rounds",
              "outside": "Outside-tolerance-time reduction at equal rounds"}
    for row, metric in enumerate(titles):
        for column, (diffusion, scenario) in enumerate(SCENARIOS.items()):
            ax = axes[row, column]
            image = _gain_heatmap(ax, gains, metric, diffusion)
            ax.set_title(f"{scenario}\n{titles[metric]}", fontsize=7)
    bar = fig.colorbar(image, ax=axes, shrink=0.6, extend="both")
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bar.set_label("Gain over fixed periods (blue: better)", fontsize=7)
    style.save(fig, out, "phase_gains_static_vs_mobile")


# --- dose response to the diffusion coefficient --------------------------------------------------------


def _diffusion_points(ax, data: pd.DataFrame, diffusions: list[float], value) -> None:
    offsets = np.linspace(-0.16, 0.16, len(ORDER))
    for offset, algorithm in zip(offsets, ORDER):
        points = [value(data[(data.algorithm == algorithm) & (data.diffusion == diffusion)])
                  for diffusion in diffusions]
        mean, low, high = map(np.array, zip(*points))
        x = np.arange(len(diffusions)) + offset
        ax.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="none", ecolor=METHODS[algorithm].color,
                    elinewidth=0.6, capsize=1.2, alpha=0.8)
        ax.plot(x, mean, **_style(algorithm, linewidth=0.8))


def _rounds_value(rows: pd.DataFrame) -> tuple[float, float, float]:
    interval = t_interval(rows.rounds)
    return interval.mean, interval.low, interval.high


def _gain_value(rows: pd.DataFrame) -> tuple[float, float, float]:
    cell = rows.iloc[0]
    return cell["mean"], cell["low"], cell["high"]


def diffusion_response(metrics: pd.DataFrame, gains: pd.DataFrame, out: Path) -> None:
    """Rounds and gains as the nodes move faster (beta = 0.95); D = 0 is the static network."""
    focus = metrics[np.isclose(metrics.beta, FOCUS_BETA) & (metrics.phase == "episode")]
    diffusions = sorted(focus[focus.algorithm.isin(ORDER)].diffusion.unique())
    fig, axes = style.figure(height=2.3, nrows=1, ncols=3)
    rounds_ax, episode_ax, steady_ax = axes
    fixed = fixed_reference(focus).groupby("period").rounds.mean()
    for period, rounds in fixed.items():
        rounds_ax.axhline(rounds, color=style.fixed_period_color(min(max(period, 1.0), 5.0)), linewidth=0.6,
                          linestyle=":", zorder=1)
        style.direct_label(rounds_ax, len(diffusions) - 0.5, rounds, f"{period:g} s", dx=1, va="center",
                           bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    _diffusion_points(rounds_ax, focus, diffusions, _rounds_value)
    _log_rounds_axis(rounds_ax, axis="y")
    rounds_ax.set_ylabel("Rounds per episode (log scale)")
    focus_gains = gains[np.isclose(gains.beta, FOCUS_BETA) & (gains.gain == "error")]
    for ax, phase in ((episode_ax, "episode"), (steady_ax, "post_bootstrap")):
        _diffusion_points(ax, focus_gains[focus_gains.phase == phase], diffusions, _gain_value)
        _gain_axis(ax)
        ax.set_ylabel(f"Error reduction at equal rounds\n{PHASE_BY_KEY[phase].label.lower()}")
    for index, ax in enumerate(axes):
        ax.set_xticks(range(len(diffusions)), [f"{diffusion:g}" for diffusion in diffusions])
        ax.set_xlim(-0.5, len(diffusions) - 0.5 + (0.45 if ax is rounds_ax else 0))
        ax.set_xlabel("Diffusion $D$ (m$^2$/s); 0 = static")
        style.panel_label(ax, "abc"[index])
    style.figure_legend(fig, _handles(fixed=False), ncol=5)
    style.save(fig, out, "diffusion_response")


# --- response to beta -----------------------------------------------------------------------------------


def beta_response(metrics: pd.DataFrame, out: Path) -> None:
    """Rounds and integrated error vs. beta, static vs. mobile: does beta still steer the trade-off?"""
    data = metrics[metrics.phase == "episode"]
    fig, axes = style.figure(height=3.5, nrows=2, ncols=2, sharex=True, sharey="row")
    x = np.arange(len(config.WEIGHTS))
    for column, (diffusion, title) in enumerate(SCENARIOS.items()):
        scenario = data[data.diffusion == diffusion]
        for row, metric in enumerate(("rounds", "error")):
            ax = axes[row, column]
            for algorithm in ORDER:
                rows = scenario[scenario.algorithm == algorithm].pivot_table(index="seed", columns="beta",
                                                                             values=metric)
                mean, low, high = t_band(rows[list(config.WEIGHTS)].to_numpy())
                ax.fill_between(x, low, high, color=METHODS[algorithm].color, alpha=0.1, linewidth=0)
                ax.plot(x, mean, **_style(algorithm, linewidth=0.9))
            ax.set_title(title if row == 0 else "", fontsize=8)
            style.panel_label(ax, "abcd"[row * 2 + column])
        axes[1, column].set_xticks(x, [f"{beta:g}" for beta in config.WEIGHTS])
        axes[1, column].set_xlabel(r"Trade-off weight $\beta$")
    _log_rounds_axis(axes[0, 0], axis="y")
    axes[0, 0].set_ylabel("Rounds per episode (log scale)")
    axes[1, 0].set_ylabel("Integrated error (s)")
    style.figure_legend(fig, _handles(fixed=False), ncol=5)
    style.save(fig, out, "beta_response_static_vs_mobile")


def render(metrics: pd.DataFrame, series: pd.DataFrame, gains: pd.DataFrame, out: Path) -> None:
    style.apply()
    scenarios = set(metrics.diffusion)
    if {STATIC_DIFFUSION, MOBILE_DIFFUSION} <= scenarios:
        frontiers(metrics, out)
        dynamics(series, out)
        phase_gains(gains, out)
        beta_response(metrics, out)
    if len(scenarios) > 2:
        diffusion_response(metrics, gains, out)
    plt.close("all")
