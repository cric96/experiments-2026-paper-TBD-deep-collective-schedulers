"""Paper-style Pareto analysis of the mobile beta sweep, with shared fixed-period controls.

The primary intervals use independent seed means, never individual episodes. Frontier gains
are paired within each seed, using the same deployment/motion/source trajectories.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator, PercentFormatter

from . import config, style
from .methods import GNN, TD3, HEURISTIC, Method
from .pareto import matched_error_saving, matched_ticks_gain, non_dominated
from .stats import t_interval

WEIGHTS = config.WEIGHTS
ADAPTIVE = (0, 1, 2, 8, 9)
FIXED_PERIODS = {10: .5, 3: 1., 4: 2., 5: 3., 6: 4., 7: 5.}
METHODS = {
    0: Method('mobile_gnn', 'GNN (mobile training)', 'Mobile GNN', GNN.color, GNN.marker),
    1: Method('mobile_mlp', 'MLP (mobile training)', 'Mobile MLP', TD3.color, TD3.marker),
    2: Method('mobile_heuristic', 'Heuristic', 'Heuristic', HEURISTIC.color, HEURISTIC.marker, 1.4),
    8: Method('static_mlp', 'MLP (static training)', 'Static MLP', TD3.color, TD3.marker),
    9: Method('static_gnn', 'GNN (static training)', 'Static GNN', GNN.color, GNN.marker),
}
SCENARIO = ['diffusion', 'tolerance', 'sourceChanges', 'protocol']
RAW_METRICS = ['roundsPerNodeSecond', 'normalizedError', 'toleratedError', 'absoluteError',
               'withinTolerance', 'reachableFraction', 'recoveryCensored', 'rewardRate']
GAIN_LABELS = {'error_at_ticks': 'Error reduction at matched rounds',
               'tolerated_at_ticks': 'Tolerated-error reduction at matched rounds',
               'rounds_at_error': 'Rounds saved at matched error'}


def prepare(episodes: pd.DataFrame) -> pd.DataFrame:
    """Validate the complete paired grid, collapse episodes, and adopt the paper's units."""
    if len(episodes[SCENARIO].drop_duplicates()) != 1:
        raise ValueError('Pareto analysis requires one mobility scenario and one protocol')
    # These conversions must track the simulator; do not silently label a smoke run as a full experiment.
    if not (episodes.protocol == 'dt0.25_train10_length200.0').all():
        raise ValueError('Paper-unit Pareto analysis requires the standard 200-second episode protocol')
    group = ['algorithm', 'beta', 'seed']
    expected_episodes = set(range(config.TRAINING_EPISODES, config.N_EPISODES))
    for key, rows in episodes.groupby(group):
        if len(rows) != len(expected_episodes) or set(rows.episode) != expected_episodes:
            raise ValueError(f'Incomplete evaluation episodes for {key}')
    data = episodes.groupby(group)[RAW_METRICS].mean().reset_index()
    expected_seeds = {1, 2, 3, 4}
    for algorithm in ADAPTIVE:
        for beta in WEIGHTS:
            seeds = set(data[(data.algorithm == algorithm) & np.isclose(data.beta, beta)].seed)
            if seeds != expected_seeds:
                raise ValueError(f'Incomplete beta sweep: algorithm={algorithm}, beta={beta}, seeds={sorted(seeds)}')
    for algorithm in FIXED_PERIODS:
        fixed = data[data.algorithm == algorithm]
        if set(fixed.seed) != expected_seeds or len(fixed) != len(expected_seeds):
            raise ValueError(f'Incomplete or duplicated beta-independent control: algorithm={algorithm}')
    data['ticks'] = data.roundsPerNodeSecond * config.N_NODES * config.EPISODE_LENGTH_S
    data['error'] = data.normalizedError * config.EPISODE_LENGTH_S
    data['tolerated'] = data.toleratedError * config.EPISODE_LENGTH_S
    data['outside'] = (1 - data.withinTolerance) * config.EPISODE_LENGTH_S
    data['period'] = data.algorithm.map(FIXED_PERIODS)
    return data


def frontier_gains(data: pd.DataFrame) -> pd.DataFrame:
    """Mean of the paired per-seed relative gains, with t intervals; no extrapolation."""
    records = []
    for algorithm in ADAPTIVE:
        for beta in WEIGHTS:
            point = data[(data.algorithm == algorithm) & np.isclose(data.beta, beta)]
            values = {key: [] for key in GAIN_LABELS}
            for row in point.itertuples():
                fixed = data[(data.seed == row.seed) & data.algorithm.isin(FIXED_PERIODS)].sort_values('ticks')
                values['error_at_ticks'].append(matched_ticks_gain(row.ticks, row.error, fixed.ticks, fixed.error))
                values['tolerated_at_ticks'].append(matched_ticks_gain(row.ticks, row.tolerated, fixed.ticks, fixed.tolerated))
                values['rounds_at_error'].append(matched_error_saving(row.ticks, row.error, fixed.ticks, fixed.error))
            for name, sample in values.items():
                interval = t_interval(sample)
                records.append(dict(algorithm=algorithm, method=METHODS[algorithm].short, beta=beta,
                    gain=name, mean=interval.mean, low=interval.low, high=interval.high, seeds=interval.n))
    return pd.DataFrame(records)


def _rows(data, algorithm, beta=None):
    selected = data[data.algorithm == algorithm]
    return selected if beta is None else selected[np.isclose(selected.beta, beta)]


def _line(algorithm):
    method = METHODS[algorithm]
    return style.line_style(method, linestyle='--' if algorithm in (8, 9) else '-', linewidth=.8,
                            markerfacecolor='white' if algorithm in (8, 9) else method.color,
                            markeredgecolor=method.color if algorithm in (8, 9) else 'white',
                            markeredgewidth=.65 if algorithm in (8, 9) else .4)


def _figure(rows, columns, height, **kwargs):
    """Readable report canvas; shared palette and typography, with room for annotations."""
    fig, axes = plt.subplots(rows, columns, figsize=(9, height), layout='none', squeeze=False, **kwargs)
    return fig, axes


def _fixed_curve(data, metric):
    return (data[data.algorithm.isin(FIXED_PERIODS)]
            .groupby('algorithm')[['ticks', metric, 'period']].mean().sort_values('ticks'))


def _mean_points(data, algorithm, metric):
    return [(beta, _rows(data, algorithm, beta).ticks.mean(), _rows(data, algorithm, beta)[metric].mean())
            for beta in WEIGHTS]


def _curve(ax, points, algorithm):
    attributes = _line(algorithm)
    attributes.update(linewidth=1.25 if algorithm in (0, 1) else .95,
                      markersize=4.7 if algorithm in (0, 1) else 4.1,
                      alpha=.98 if algorithm in (0, 1) else .78,
                      zorder=4 if algorithm in (0, 1) else 3)
    ax.plot([point[1] for point in points], [point[2] for point in points], **attributes)


def _label_beta_points(ax, points):
    """Place short labels without collisions; leader lines preserve the point association."""
    renderer = ax.figure.canvas.get_renderer()
    occupied = []
    markers = [Bbox.from_bounds(*((ax.transData.transform((x, y)) - 4).tolist()), 8, 8)
               for _, _, x, y in points]
    offsets = [(7, 9), (7, -12), (-7, 9), (-7, -12), (0, 19), (0, -22),
               (22, 0), (-22, 0), (13, 21), (-13, -23), (25, 14), (-25, -17)]
    bounds = ax.get_window_extent(renderer)
    for algorithm, beta, x, y in sorted(points, key=lambda point: (point[2], point[3])):
        candidates = []
        for index, (dx, dy) in enumerate(offsets):
            text = ax.annotate(f'$\\beta={beta:g}$', (x, y), xytext=(dx, dy), textcoords='offset points',
                               ha='left' if dx > 0 else 'right' if dx < 0 else 'center',
                               va='bottom' if dy > 0 else 'top' if dy < 0 else 'center', fontsize=7.2)
            box = text.get_window_extent(renderer).expanded(1.08, 1.15)
            outside = box.x0 < bounds.x0 or box.x1 > bounds.x1 or box.y0 < bounds.y0 or box.y1 > bounds.y1
            score = 10000 * outside + sum(box.overlaps(other) for other in occupied) * 500
            score += sum(box.overlaps(marker) for marker in markers) * 120 + index
            candidates.append((score, dx, dy, box))
            text.remove()
        _, dx, dy, box = min(candidates, key=lambda candidate: candidate[0])
        occupied.append(box)
        color = METHODS[algorithm].color
        ax.annotate(f'$\\beta={beta:g}$', (x, y), xytext=(dx, dy), textcoords='offset points',
                    ha='left' if dx > 0 else 'right' if dx < 0 else 'center',
                    va='bottom' if dy > 0 else 'top' if dy < 0 else 'center',
                    fontsize=7.2, color=color, zorder=6,
                    bbox=dict(facecolor='white', edgecolor='none', pad=.15, alpha=.92),
                    arrowprops=dict(arrowstyle='-', color=color, linewidth=.5, alpha=.75))


def _frontiers(data: pd.DataFrame, out: Path, metrics, labels, name) -> pd.DataFrame:
    fig, axes = _figure(2, 2, 6.5)
    fig.subplots_adjust(left=.09, right=.985, top=.83, bottom=.145, wspace=.28, hspace=.57)
    records = []
    zooms = []
    for column, (metric, label) in enumerate(zip(metrics, labels)):
        ax, detail = axes[:, column]
        fixed = _fixed_curve(data, metric)
        fixed_style = style.fixed_period_style(label='Fixed period (0.5–5 s)', linewidth=1.15, markersize=4.6)
        ax.plot(fixed.ticks, fixed[metric], **fixed_style)
        detail.plot(fixed.ticks, fixed[metric], **fixed_style)
        for _, row in fixed.iterrows():
            dx, dy = (-7, 7) if row.period == .5 else (5, -10) if metric == 'outside' and row.period == 4 else (5, 6)
            ax.annotate(f'{row.period:g} s', (row.ticks, row[metric]), xytext=(dx, dy),
                        textcoords='offset points', fontsize=7.8, color=style.INK_SECONDARY,
                        ha='right' if dx < 0 else 'left',
                        bbox=dict(facecolor='white', edgecolor='none', pad=.15, alpha=.85))
        mobile_points = []
        for algorithm in ADAPTIVE:
            points = _mean_points(data, algorithm, metric)
            _curve(ax, points, algorithm)
            if algorithm in (0, 1):
                _curve(detail, points, algorithm)
                mobile_points.extend((algorithm, beta, x, y) for beta, x, y in points)
            for beta, x, y in points:
                records.append(dict(metric=metric, algorithm=algorithm, beta=beta, ticks=x, value=y))
        for algorithm, row in fixed.iterrows():
            records.append(dict(metric=metric, algorithm=algorithm, beta=np.nan, ticks=row.ticks, value=row[metric]))
        xs = np.array([point[2] for point in mobile_points]); ys = np.array([point[3] for point in mobile_points])
        xpad = max(np.ptp(xs) * .18, 180.)
        ypad = max(np.ptp(ys) * .26, 1.1 if metric == 'outside' else .5)
        detail.set_xlim(xs.min()-xpad, xs.max()+xpad)
        detail.set_ylim(ys.min()-ypad, ys.max()+ypad)
        roi = Rectangle((xs.min()-xpad, ys.min()-ypad), np.ptp(xs)+2*xpad, np.ptp(ys)+2*ypad,
                        facecolor='#f0efeb', edgecolor='#a5a29a', linewidth=.65, alpha=.65, zorder=.5)
        ax.add_patch(roi)
        ax.set_xscale('log')
        ax.set_xticks([4000, 10000, 20000, 40000], ['4k', '10k', '20k', '40k'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Rounds per episode (log scale)')
        ax.set_ylabel(label)
        ax.set_title('Overview', loc='left', fontsize=10, color=style.INK_SECONDARY)
        ax.margins(x=.13, y=.16)
        detail.xaxis.set_major_locator(MaxNLocator(5))
        detail.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value/1000:g}k'))
        detail.set_xlabel('Rounds per episode')
        detail.set_ylabel(label)
        detail.set_title('Mobile policies · detail', loc='left', fontsize=10, color=style.INK_SECONDARY)
        zooms.append((detail, mobile_points))
    handles = [Line2D([], [], **_line(algorithm)) for algorithm in ADAPTIVE]
    handles.append(Line2D([], [], **style.fixed_period_style(label='Fixed period (0.5–5 s)')))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .985), ncol=3, fontsize=9)
    fig.text(.53, .018, 'Mean points across four seeds. Lower and farther left is better.\nTime and error are per node; 95% CIs are shown in the β plots.',
             ha='center', fontsize=8, color=style.INK_SECONDARY)
    fig.canvas.draw()
    for ax, points in zooms:
        _label_beta_points(ax, points)
    style.save(fig, out, name)
    return pd.DataFrame(records)


def pareto_frontiers(data: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Readable overview and mobile-policy zooms; uncertainty is displayed separately versus beta."""
    accuracy = _frontiers(data, out, ['outside', 'error'],
        ['Outside-tolerance time (s)', 'Integrated error (s)'],
        'mobility_pareto_frontiers')
    objective = _frontiers(data, out, ['error', 'tolerated'],
        ['Integrated error (s)', 'Error beyond tolerance (s)'],
        'mobility_objective_frontiers')
    points = pd.concat([accuracy, objective], ignore_index=True).drop_duplicates(
        ['metric', 'algorithm', 'beta']).reset_index(drop=True)
    points['nonDominated'] = False
    for metric, indices in points.groupby('metric').groups.items():
        subset = points.loc[indices]
        points.loc[indices, 'nonDominated'] = non_dominated(subset.ticks, subset.value)
    return points


def _versus_beta(data: pd.DataFrame, out: Path, metrics, labels, name):
    fig, axes = _figure(2, 3, 5.7)
    fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.83, wspace=.36, hspace=.50)
    families = [(0, 1), (2, 8, 9)]
    for row, algorithms in enumerate(families):
        for column, (metric, label) in enumerate(zip(metrics, labels)):
            ax = axes[row, column]
            for index, algorithm in enumerate(algorithms):
                intervals = [t_interval(_rows(data, algorithm, beta)[metric]) for beta in WEIGHTS]
                means = np.array([interval.mean for interval in intervals])
                low = np.array([interval.low for interval in intervals]); high = np.array([interval.high for interval in intervals])
                x = np.arange(len(WEIGHTS))
                attributes = _line(algorithm)
                attributes.update(linewidth=1.35, markersize=4.8)
                ax.plot(x, means, **attributes)
                ax.fill_between(x, low, high, color=METHODS[algorithm].color, alpha=.11 if row == 0 else .07, linewidth=0)
                # Thin whiskers retain exact pointwise intervals without competing with the means.
                ax.errorbar(x+(index-(len(algorithms)-1)/2)*.035, means,
                            yerr=[means-low, high-means], fmt='none', color=METHODS[algorithm].color,
                            elinewidth=.6, alpha=.4, capsize=1.5)
            ax.set_xticks(range(len(WEIGHTS)), [f'{beta:g}' for beta in WEIGHTS])
            ax.set_xlabel('Trade-off weight $\\beta$')
            ax.set_ylabel(label)
            ax.margins(x=.07, y=.15)
            if metric == 'ticks':
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value/1000:g}k'))
            if column == 0:
                title = 'Mobile training' if row == 0 else 'Static training and heuristic'
                ax.set_title(title, loc='left', fontsize=10, color=style.INK_SECONDARY)
    handles = [Line2D([], [], **_line(algorithm)) for algorithm in ADAPTIVE]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .985), ncol=3, fontsize=9)
    fig.text(.54, .018, 'Means and 95% CIs across four seeds; time and error are per node.\nRows use separate y ranges to reveal the response to β.',
             ha='center', fontsize=8, color=style.INK_SECONDARY)
    style.save(fig, out, name)


def tradeoff_vs_beta(data: pd.DataFrame, out: Path):
    _versus_beta(data, out, ['ticks', 'outside', 'error'],
        ['Rounds per episode', 'Outside-tolerance time (s)', 'Integrated error (s)'],
        'mobility_tradeoff_vs_beta')
    _versus_beta(data, out, ['ticks', 'error', 'tolerated'],
        ['Rounds per episode', 'Integrated error (s)', 'Error beyond tolerance (s)'],
        'mobility_objective_vs_beta')


def _gain_matrix(gains, name):
    selected = gains[gains.gain == name].set_index(['algorithm', 'beta'])
    matrix = np.array([[selected.loc[(algorithm, beta), 'mean'] for beta in WEIGHTS] for algorithm in ADAPTIVE])
    return selected, matrix


def _gain_intervals(gains, out):
    """A companion forest plot makes every interval readable without packing it into a cell."""
    fig, axes = _figure(5, 3, 9.2, sharex='col', sharey=True)
    fig.subplots_adjust(left=.13, right=.98, bottom=.075, top=.93, wspace=.19, hspace=.33)
    titles = ['Error reduction\nat matched rounds', 'Error beyond tolerance\nat matched rounds',
              'Rounds saved\nat matched error']
    for column, name in enumerate(GAIN_LABELS):
        selected, _ = _gain_matrix(gains, name)
        finite = selected[['mean', 'low', 'high']].to_numpy().ravel()
        finite = np.r_[0., finite[np.isfinite(finite)]]
        padding = max(np.ptp(finite) * .08, .02)
        limits = (finite.min()-padding, finite.max()+padding)
        for row, algorithm in enumerate(ADAPTIVE):
            ax = axes[row, column]
            method = METHODS[algorithm]
            ax.axvline(0, color=style.INK_SECONDARY, linewidth=.7, linestyle='--')
            for index, beta in enumerate(WEIGHTS):
                cell = selected.loc[(algorithm, beta)]
                y = len(WEIGHTS)-1-index
                if np.isfinite(cell['mean']):
                    uncertain = not (cell['low'] > 0 or cell['high'] < 0)
                    ax.errorbar(cell['mean'], y, xerr=[[cell['mean']-cell['low']], [cell['high']-cell['mean']]],
                                color=method.color, marker=method.marker, markersize=4.5,
                                markerfacecolor='white' if uncertain else method.color,
                                markeredgewidth=.8, elinewidth=.85, capsize=2)
                else:
                    ax.text(.5, y, 'outside reference', transform=ax.get_yaxis_transform(),
                            ha='center', va='center', fontsize=7, color=style.INK_MUTED)
            ax.set_xlim(*limits); ax.set_ylim(-.6, 4.6)
            ax.set_yticks(range(5), [f'{beta:g}' for beta in WEIGHTS[::-1]])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis='y', visible=False)
            if column == 0:
                ax.set_ylabel(METHODS[algorithm].short+'\n$\\beta$', fontsize=9)
            if row == 0:
                ax.set_title(titles[column], fontsize=10, pad=10)
    fig.text(.55, .025, 'Paired seed means ± 95% CI. Open markers indicate intervals crossing zero.',
             ha='center', fontsize=8, color=style.INK_SECONDARY)
    style.save(fig, out, 'mobility_frontier_gains_intervals')


def gain_heatmap(gains: pd.DataFrame, out: Path):
    fig, axes = _figure(1, 3, 3.2, sharey=True)
    axes = axes[0]
    fig.subplots_adjust(left=.13, right=.90, bottom=.26, top=.79, wspace=.10)
    norm = TwoSlopeNorm(vmin=-.6, vcenter=0, vmax=.6)
    titles = ['Error reduction\nat matched rounds', 'Error beyond tolerance\nat matched rounds',
              'Rounds saved\nat matched error']
    for ax, name, title in zip(axes, GAIN_LABELS, titles):
        selected, matrix = _gain_matrix(gains, name)
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap='RdBu', norm=norm, aspect='auto')
        for row, algorithm in enumerate(ADAPTIVE):
            for column, beta in enumerate(WEIGHTS):
                cell = selected.loc[(algorithm, beta)]
                if np.isfinite(cell['mean']):
                    text = f"{cell['mean']:.0%}"
                    if cell['low'] <= 0 <= cell['high']:
                        ax.add_patch(Rectangle((column-.5, row-.5), 1, 1, facecolor='none',
                                              edgecolor='#8e8e8e', linewidth=0, hatch='///'))
                else:
                    text = '—'
                ax.text(column, row, text, ha='center', va='center', fontsize=8.2,
                        color='white' if np.isfinite(cell['mean']) and abs(cell['mean'])>.35
                              and not cell['low'] <= 0 <= cell['high'] else style.INK,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.7, pad=.1)
                             if np.isfinite(cell['mean']) and cell['low'] <= 0 <= cell['high'] else None)
        ax.set_xticks(range(len(WEIGHTS)), [f'{beta:g}' for beta in WEIGHTS])
        ax.set_yticks(range(len(ADAPTIVE)), [METHODS[algorithm].short for algorithm in ADAPTIVE])
        ax.set_xlabel('$\\beta$')
        ax.set_title(title, fontsize=10, pad=10)
        ax.grid(False); ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    color_axis = fig.add_axes([.92, .23, .016, .54])
    bar = fig.colorbar(image, cax=color_axis, extend='both')
    bar.set_ticks([-.6, -.3, 0, .3, .6])
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bar.set_label('Relative gain', fontsize=9)
    fig.text(.53, .025, 'Hatched cells: 95% CI crosses zero. Colors clipped at ±60%; numbers show the full gain.',
             ha='center', fontsize=8, color=style.INK_SECONDARY)
    style.save(fig, out, 'mobility_frontier_gains')
    _gain_intervals(gains, out)


def render(episodes: pd.DataFrame, out: Path):
    data = prepare(episodes)
    out.mkdir(parents=True, exist_ok=True)
    style.apply()
    plt.rcParams.update({'figure.constrained_layout.use': False, 'figure.autolayout': False,
                         'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'legend.fontsize': 9, 'axes.titlesize': 10})
    points = pareto_frontiers(data, out)
    gains = frontier_gains(data)
    tradeoff_vs_beta(data, out)
    gain_heatmap(gains, out)
    data.to_csv(out/'mobility_per_seed.csv', index=False)
    points.to_csv(out/'mobility_pareto_points.csv', index=False)
    points[points.nonDominated].to_csv(out/'mobility_empirical_frontier.csv', index=False)
    gains.to_csv(out/'mobility_frontier_gains.csv', index=False)
