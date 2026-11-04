"""Manifest-driven analysis; no inference of protocols from legacy filenames.

python -m analysis.campaign build/learning-campaign/validation --out charts
All declared runs must be complete. Selection is written only for validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .pareto import frontier_value_at
from experiments.campaign import (validate, CORE, DYNAMIC, FAMILIES, FIXED, FOCUS_BETA, LEARNED, PROGRAMS, STATIC,
                                  TRAIN_SEEDS)

PAIR = ['training_seed', 'evaluation_seed']
SCENARIO = ['family', 'program', 'nodes', 'speed', 'phase']

EPISODE_LENGTH = 200.
BOOTSTRAP_SEED = 1729
PHASES = {'whole': (0, 200), 'bootstrap': (0, 40), 'quiet': (40, 100),
          'event': (100, 150), 'after': (150, 200)}


MEASURES = ('roundsPerNodeSecond', 'normalizedError', 'reachableFraction', 'absoluteError')


def phase_values(frames, terminal, start, end):
    """Means over [start, end) of the cumulative episode means exported every frame (and at the episode end)."""
    last = terminal if end == EPISODE_LENGTH else frames.loc[end]
    first = frames.loc[start] if start else None
    return {column: (float(last[column]) * end - (float(first[column]) * start if first is not None else 0.))
            / (end - start) for column in MEASURES}


def run_records(home, job):
    """One record per evaluation episode and phase of a run."""
    directory = home / 'exports' / job['key']
    if not validate(directory, job['settings']):
        raise ValueError(f'Incomplete run: {job["key"]}')
    if job['settings'].get('episodeLength', EPISODE_LENGTH) != EPISODE_LENGTH:
        raise ValueError('Smoke runs cannot be used for scientific selection')
    training = job['settings']['trainingEpisodes']
    full = pd.read_csv(next(directory.glob('*_full.csv')))
    terminal = pd.read_csv(next(directory.glob('*_episodes.csv'))).set_index('episode')
    full = full[full.episode >= training].copy()
    full['elapsed'] = full.time - full.episode * EPISODE_LENGTH
    description = {k: v for k, v in job.items() if k not in ('settings', 'shared')}
    return [description | dict(episode=episode, phase=phase,
                               **phase_values(frames.set_index('elapsed'), terminal.loc[episode], start, end))
            for episode, frames in full.groupby('episode') for phase, (start, end) in PHASES.items()]


def load(*homes):
    """Every evaluation episode of the campaigns; the shared controls are paired with every training seed of their
    split. Campaigns of one split (e.g. a second round of variants) are merged; a run may appear only once."""
    records, keys = [], set()
    for home, job in [(home, job) for home in homes for job in json.loads((home / 'manifest.json').read_text())]:
        if job['key'] in keys:
            raise ValueError(f'Run declared twice: {job["key"]}')
        keys.add(job['key'])
        runs = run_records(home, job)
        seeds = TRAIN_SEEDS[job['split']] if job.get('shared') else (job['training_seed'],)
        records += [record | dict(training_seed=seed) for seed in seeds for record in runs]
    return pd.DataFrame(records)


def gains(episodes):
    grouping = SCENARIO + PAIR + ['variant', 'algorithm', 'beta']
    means = episodes.groupby(grouping, as_index=False)[list(MEASURES)].mean()
    records = []
    for _, data in means.groupby(SCENARIO + PAIR):
        fixed = data[(data.variant == 'fixed') & data.algorithm.isin(FIXED)]
        if set(fixed.algorithm) != set(FIXED) or len(fixed) != len(FIXED):
            raise ValueError('Missing/duplicated matched fixed-period controls')
        for row in data[~data.algorithm.isin(FIXED)].to_dict('records'):
            reference = frontier_value_at(fixed.roundsPerNodeSecond, fixed.normalizedError,
                                          row['roundsPerNodeSecond'])
            row['gain'] = 1 - row['normalizedError'] / reference if reference > 0 else np.nan
            records.append(row)
    return pd.DataFrame(records)


def crossed_interval(frame, value, repetitions=2000):
    """Resample training seeds and topology seeds separately, preserving paired differences."""
    grid = frame.pivot(index='training_seed', columns='evaluation_seed', values=value).to_numpy()
    if not np.isfinite(grid).all():
        return dict(mean=float(np.nanmean(grid)), low=np.nan, high=np.nan, complete=False)
    if min(grid.shape) < 2:
        return dict(mean=float(grid.mean()), low=np.nan, high=np.nan, complete=False)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = rng.integers(grid.shape[0], size=(repetitions, grid.shape[0]))
    columns = rng.integers(grid.shape[1], size=(repetitions, grid.shape[1]))
    sample = grid[rows[:, :, None], columns[:, None, :]].mean(axis=(1, 2))
    low, high = np.quantile(sample, [.025, .975])
    return dict(mean=float(grid.mean()), low=float(low), high=float(high), complete=True)


def primary_scores(data):
    """Static gain and zero-shot gain on the other programs at the training population, at the focus beta."""
    whole = data[(data.phase == 'whole') & data.family.isin(['static', 'generalization']) & data.algorithm.isin(LEARNED)
                 & (data.nodes == 100) & (data.beta == FOCUS_BETA)]
    # Require complete support; undefined frontier gains cannot quietly change the selection population.
    groups = ['variant', 'algorithm'] + PAIR + ['family']
    scores = whole.groupby(groups).gain.agg(lambda values: values.mean() if values.notna().all() else np.nan)
    scores = scores.unstack('family')
    scores['score'] = (scores['static'] + scores['generalization']) / 2
    return scores.reset_index()[['variant', 'algorithm', *PAIR, 'score']]


def select(data):
    """Selection on the primary score; once for both architectures together (equal weights), once per
    architecture: they are separate methods, so the test keeps the variant selected for each one."""
    joint = compare(data)
    by_architecture = {int(algorithm): compare(data[data.algorithm == algorithm]) for algorithm in LEARNED}
    return dict(variant=joint['variant'], comparisons=joint['comparisons'],
                byArchitecture={algorithm: result['variant'] for algorithm, result in by_architecture.items()},
                architectureComparisons={algorithm: result['comparisons']
                                         for algorithm, result in by_architecture.items()},
                criterion='equal static and zero-shot macro gain',
                note='Validation selection only; requires independent test confirmation.')


def confirm(data, selected):
    """Test check of the validation `selected` variant of each architecture, on independent seeds: it is confirmed
    when the same rule, applied to the test data, selects it again. It is reported, never replaced."""
    test = select(data)
    return dict(selected=selected, testSelection=test['byArchitecture'],
                confirmed={algorithm: test['byArchitecture'][algorithm] == variant
                           for algorithm, variant in selected.items()},
                architectureComparisons=test['architectureComparisons'], jointComparisons=test['comparisons'],
                criterion=test['criterion'],
                note='Test split: training and evaluation seeds independent of the validation that selected.')

def compare(data):
    """Paired difference of each variant from `current`; the best one with a positive lower bound, if any."""
    scores = primary_scores(data)
    # Same architecture remains paired; both architectures have equal weight in variant selection.
    baseline = scores[scores.variant == 'current'].drop(columns='variant').rename(columns={'score': 'baseline'})
    comparisons = []
    for variant, values in scores.groupby('variant'):
        if variant == 'current':
            continue
        paired = values.merge(baseline, on=['algorithm', *PAIR], validate='one_to_one')
        paired['difference'] = paired.score - paired.baseline
        paired = paired.groupby(PAIR, as_index=False).difference.agg(lambda x: x.mean() if x.notna().all() else np.nan)
        comparisons.append(dict(variant=variant, **crossed_interval(paired, 'difference')))
    supported = [row for row in comparisons if row['complete'] and row['low'] > 0]
    winner = max(supported, key=lambda row: row['mean'])['variant'] if supported else 'current'
    return dict(variant=winner, comparisons=comparisons)


KEYS = ['variant', 'algorithm', 'beta', *SCENARIO]


def write_tables(home, episodes, data):
    """Paired gains, a per-scenario summary with crossed intervals, and descriptive variability."""
    data.to_csv(home / 'paired-gains.csv', index=False)
    summary = []
    for key, rows in data.groupby(KEYS):
        estimates = {value: crossed_interval(rows.groupby(PAIR, as_index=False)[value].mean(), value)
                     for value in ('gain', 'roundsPerNodeSecond', 'normalizedError')}
        summary.append(dict(zip(KEYS, key)) | {f'{value}_{part}': estimate[part] for value, estimate in
                                               estimates.items() for part in ('mean', 'low', 'high')})
    pd.DataFrame(summary).to_csv(home / 'summary.csv', index=False)
    # Descriptive, not an additive decomposition: spread across episodes, topologies and trained policies.
    means = episodes.groupby([*KEYS, *PAIR]).normalizedError.mean().reset_index()
    variability = pd.DataFrame({
        'withinRun': episodes.groupby([*KEYS, *PAIR]).normalizedError.std().groupby(KEYS).mean(),
        'acrossTopologies': means.groupby([*KEYS, 'training_seed']).normalizedError.std().groupby(KEYS).mean(),
        'acrossPolicies': means.groupby([*KEYS, 'evaluation_seed']).normalizedError.std().groupby(KEYS).mean(),
        'mean': means.groupby(KEYS).normalizedError.mean(),
    })
    variability['policyCV'] = variability.acrossPolicies / variability['mean']
    variability.reset_index().to_csv(home / 'variability.csv', index=False)


def final_variants(data):
    """The final method of each architecture: the variant of its beta study (selected on validation)."""
    swept = data[(data.beta != FOCUS_BETA) & data.algorithm.isin(LEARNED)]
    return {algorithm: next(iter(swept[swept.algorithm == algorithm].variant.unique()), 'current')
            for algorithm in LEARNED}


def draw_validation(data, out):
    from .figures import campaign as figures
    groups = {'Static gradient': [STATIC], 'Programs, 100 nodes': PROGRAMS, 'Motion 1 m/s': [DYNAMIC[0]]}
    figures.ablation_figure(data, out / 'static', groups, 'validation_ablation')


def draw_test(home, episodes, data, out):
    from .figures import campaign as figures
    from .campaign_snapshots import draw
    from .campaign_paper import draw as draw_paper
    final = final_variants(data)
    figures.family_figure(episodes, data, out / 'static', 'static_tradeoff', [STATIC],
                          ['Gradient, 100 nodes, source switch at 100 s'], final)
    figures.family_figure(episodes, data, out / 'dynamic', 'dynamic_tradeoff', DYNAMIC,
                          [f'Half of the nodes move at {speed:g} m/s (100–150 s)' for *_, speed in DYNAMIC], final)
    figures.family_figure(episodes, data, out / 'generalization', 'programs_tradeoff', PROGRAMS,
                          [f'{figures.PROGRAMS[program]}, 100 nodes (zero-shot)' for _, program, _, _ in PROGRAMS],
                          final)
    figures.generalization_figure(data, out / 'generalization', final)
    figures.improvement_figure(data, out / 'generalization', final)
    figures.beta_figure(episodes, out / 'beta-study', final, CORE)
    draw(home, out, final[LEARNED[0]])
    draw_paper(home, out, final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('home', type=Path, help='Campaign folder, where tables and decisions are written')
    parser.add_argument('--with', dest='others', type=Path, nargs='*', default=[],
                        help='Further campaign folders of the same split (e.g. a second round of variants)')
    parser.add_argument('--out', type=Path, default=Path('charts'))
    args = parser.parse_args()
    from . import style
    style.apply()
    for family in FAMILIES:
        (args.out / family).mkdir(parents=True, exist_ok=True)
    episodes = load(args.home, *args.others)
    data = gains(episodes)
    write_tables(args.home, episodes, data)
    validation = set(episodes.split) == {'validation'}
    decision = select(data) if validation else confirm(data, final_variants(data))
    (args.home / ('selection.json' if validation else 'confirmation.json')).write_text(
        json.dumps(decision, indent=2, allow_nan=True))
    if validation:
        draw_validation(data, args.out)
    else:
        draw_test(args.home, episodes, data, args.out)


if __name__ == '__main__':
    main()
