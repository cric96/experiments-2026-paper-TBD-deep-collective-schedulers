"""The campaign figures render from a complete synthetic test campaign."""
from itertools import product

import numpy as np
import pandas as pd

from analysis.campaign import PHASES, gains
from analysis.figures import campaign as figures
from experiments.campaign import (BASELINES, CORE, FIXED, LEARNED, RANGE_BASELINES, SCENARIOS, SPLITS, TRAIN_SEEDS,
                                  grid_of)

PERIODS = dict(zip(FIXED, (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)))
FINALS = {0: 'current', 1: 'balanced'}
FULL_RANGE = {0: 'legacy-full-range', 1: 'legacy-full-range'}


def synthetic_episodes(finals=FINALS, tuned=None) -> pd.DataFrame:
    """Error proportional to the period; learned policies 30% better than the fixed periods at the same rounds. The
    `finals` at every beta of their grid, the reactive baselines `tuned` by beta at every beta of theirs."""
    rng = np.random.default_rng(3)
    records = []
    seeds = list(product(TRAIN_SEEDS['test'], SPLITS['test']))
    for (family, program, nodes, speed), phase, (training, evaluation) in product(SCENARIOS['test'], PHASES, seeds):
        common = dict(family=family, program=program, nodes=nodes, speed=speed, phase=phase, training_seed=training,
                      evaluation_seed=evaluation, episode=10, reachableFraction=1.0, absoluteError=1.0)
        for algorithm, period in PERIODS.items():
            records.append(common | dict(variant='fixed', algorithm=algorithm, beta=.95, roundsPerNodeSecond=1 / period,
                                         normalizedError=.02 * period))
        rows = [('fixed', algorithm, .95) for algorithm in BASELINES]
        rows += [(variant, algorithm, .95) for variant in ('current', 'balanced', 'legacy') for algorithm in LEARNED]
        rows += [(finals[algorithm], algorithm, beta) for algorithm in LEARNED for beta in grid_of(finals[algorithm])[0]
                 if (finals[algorithm], algorithm, beta) not in rows]
        rows += [(tuned, algorithm, beta) for algorithm in BASELINES for beta in grid_of(tuned)[0]] if tuned else []
        for variant, algorithm, beta in rows:
            period = 1.5 + 2 * (1 - beta) + rng.uniform(0, .5)
            records.append(common | dict(variant=variant, algorithm=algorithm, beta=beta,
                                         roundsPerNodeSecond=1 / period, normalizedError=.014 * period))
    return pd.DataFrame(records)


def test_every_family_renders(tmp_path):
    episodes = synthetic_episodes()
    data = gains(episodes)
    figures.family_figure(episodes, data, tmp_path / 'static', 'static_tradeoff', [CORE[0]], ['Static'], FINALS)
    figures.generalization_figure(data, tmp_path / 'generalization', FINALS)
    figures.improvement_figure(data, tmp_path / 'generalization', FINALS)
    figures.beta_figure(episodes, tmp_path / 'beta-study', FINALS, CORE)
    figures.ablation_figure(data, tmp_path / 'static', {'Static': [CORE[0]], 'Programs': CORE[2:]}, 'ablation')
    rendered = {path.name for path in tmp_path.rglob('*.pdf')}
    assert rendered == {'static_tradeoff.pdf', 'zero_shot_gains.pdf', 'improvement_over_current.pdf',
                        'beta_tradeoff.pdf', 'ablation.pdf'}


def test_report_tables(tmp_path):
    import json
    from analysis.campaign import write_tables, select
    from analysis.campaign_report import report
    episodes = synthetic_episodes()
    data = gains(episodes)
    write_tables(tmp_path, episodes, data)
    (tmp_path / 'confirmation.json').write_text(json.dumps(select(data)))
    text = report(tmp_path)
    assert '## Whole episode' in text and 'GNN, + balanced replay' in text and 'policyCV' in text
    assert '| GNN, current |' in text


def test_a_final_control_is_drawn_once():
    assert figures.roles('legacy') == {'final': 'legacy', 'current': 'current'}
    assert figures.roles('current') == {'final': 'current', 'legacy': 'legacy'}
    assert figures.roles('balanced') == {'final': 'balanced', 'current': 'current', 'legacy': 'legacy'}


def test_the_phase_panels_compare_the_reactive_baselines():
    series = figures.compared_series({0: 'legacy', 1: 'legacy'})
    assert [(variant, algorithm) for _, variant, algorithm, _, _ in series[-len(BASELINES):]] == \
        [('fixed', algorithm) for algorithm in BASELINES]
    assert len({label for *_, label in series}) == len(series)


def test_the_overview_lists_every_schedule_once():
    from analysis.campaign_paper_summary import overview_entries
    for finals in ({0: 'legacy', 1: 'legacy'}, {0: 'current', 1: 'balanced'}):
        labels = [label for label, *_ in overview_entries(finals)]
        assert len(labels) == len(set(labels))


def test_the_ablation_compares_with_any_reference_at_the_focus_beta(tmp_path):
    data = gains(synthetic_episodes())
    figures.ablation_figure(data, tmp_path, {'Static': [CORE[0]]}, 'ablation', reference='legacy')
    assert (tmp_path / 'ablation.pdf').exists()


def test_the_zero_shot_grid_of_a_shared_baseline_has_no_interval_mark():
    episodes = synthetic_episodes()
    # The baselines run once per topology, under the first training seed, like the fixed periods.
    shared = episodes[~(episodes.algorithm.isin(BASELINES) & (episodes.training_seed != TRAIN_SEEDS['test'][0]))]
    table, cells = figures.gain_grid(gains(shared), 'fixed', BASELINES[0], interval=False)
    assert np.isfinite(table).all() and not any(text.endswith('*') for *_, text in cells)


def test_every_method_is_compared_at_the_focus_beta_of_its_scale():
    from analysis.campaign import at_focus, final_variants
    episodes = synthetic_episodes(FULL_RANGE, RANGE_BASELINES)
    focus = at_focus(episodes).groupby('variant').beta.unique()
    assert {variant: list(betas) for variant, betas in focus.items()} == {
        'fixed': [.95], 'current': [.95], 'balanced': [.95], 'legacy': [.95], 'legacy-full-range': [.8],
        RANGE_BASELINES: [.8]}
    assert final_variants(gains(episodes)) == FULL_RANGE


def test_a_full_range_campaign_is_compared_with_its_tuned_baselines(tmp_path):
    episodes = synthetic_episodes(FULL_RANGE, RANGE_BASELINES)
    data = gains(episodes)
    baselines = figures.reactive_variant(data.variant, FULL_RANGE)
    assert baselines == RANGE_BASELINES and figures.reactive_variant(synthetic_episodes().variant, FINALS) == 'fixed'
    series = figures.compared_series(FULL_RANGE, baselines)
    assert [(variant, algorithm) for _, variant, algorithm, _, _ in series[-len(BASELINES):]] == \
        [(RANGE_BASELINES, algorithm) for algorithm in BASELINES]
    assert len({label for *_, label in series}) == len(series)
    figures.family_figure(episodes, data, tmp_path / 'static', 'static_tradeoff', [CORE[0]], ['Static'], FULL_RANGE,
                          baselines)
    figures.generalization_figure(data, tmp_path / 'generalization', FULL_RANGE, baselines)
    figures.improvement_figure(data, tmp_path / 'generalization', FULL_RANGE)
    figures.beta_figure(episodes, tmp_path / 'beta-study', FULL_RANGE, CORE, baselines)
    rendered = {path.name for path in tmp_path.rglob('*.pdf')}
    assert rendered == {'static_tradeoff.pdf', 'zero_shot_gains.pdf', 'improvement_over_current.pdf',
                        'beta_tradeoff.pdf'}


def test_the_report_tells_the_tuned_baselines_from_the_untuned_ones(tmp_path):
    import json
    from analysis.campaign import write_tables, select
    from analysis.campaign_report import report
    episodes = synthetic_episodes(FULL_RANGE, RANGE_BASELINES)
    data = gains(episodes)
    write_tables(tmp_path, episodes, data)
    (tmp_path / 'confirmation.json').write_text(json.dumps(select(data)))
    text = report(tmp_path)
    assert '| Trickle |' in text and '| Trickle, tuned by β |' in text
    assert 'β = 0.8 (`legacy-full-range`, `range-baselines`)' in text


def test_the_snapshots_are_read_at_the_beta_at_which_the_gnn_records_its_attention(tmp_path):
    import json
    from analysis.campaign_snapshots import EMBEDDINGS, INSPECTED_EVALUATION_SEED, SNAPSHOT_BETA, inspected_runs
    jobs = [dict(key=f'static-{beta}', family='static', variant='legacy-full-range', algorithm=0, beta=beta, nodes=100,
                 training_seed=TRAIN_SEEDS['test'][0], evaluation_seed=INSPECTED_EVALUATION_SEED) for beta in (.8, .95)]
    (tmp_path / 'manifest.json').write_text(json.dumps(jobs))
    run = tmp_path / 'exports' / 'static-0.95' / 'spatial'
    run.mkdir(parents=True)
    (run / EMBEDDINGS).write_text('tEp,attEntropy,maxAlpha\n85,1,0.5\n')
    [(job, _)] = inspected_runs(tmp_path, 'legacy-full-range')
    assert job['beta'] == SNAPSHOT_BETA == .95


def test_the_titles_of_the_tuned_baselines_fit_the_zero_shot_panels():
    from analysis.figures.campaign import zero_shot_panels
    titles = [title for _, _, title, _ in zero_shot_panels(FULL_RANGE, RANGE_BASELINES)]
    assert titles[-2:] == ['Trickle, tuned by β, one run per topology', 'Gradual, tuned by β, one run per topology']


def test_a_campaign_can_be_compared_without_the_controls():
    from analysis.campaign_paper_summary import overview_entries
    import matplotlib.pyplot as plt
    assert figures.roles('legacy-full-range', controls=()) == {'final': 'legacy-full-range'}
    series = figures.compared_series(FULL_RANGE, RANGE_BASELINES, controls=())
    assert [(variant, algorithm) for _, variant, algorithm, _, _ in series] == \
        [*(('legacy-full-range', algorithm) for algorithm in LEARNED), *((RANGE_BASELINES, algorithm)
                                                                         for algorithm in BASELINES)]
    episodes = synthetic_episodes(FULL_RANGE, RANGE_BASELINES)
    _, ax = plt.subplots()
    figures.pareto_panel(ax, episodes, CORE[0], FULL_RANGE, baselines=RANGE_BASELINES, controls=())
    labels = ax.get_legend_handles_labels()[1]
    assert not any('current' in label or 'legacy objective' in label for label in labels)
    entries = [label for label, *_ in overview_entries(FULL_RANGE, RANGE_BASELINES, controls=())]
    assert not any('legacy objective' in label for label in entries)
    plt.close('all')


def test_the_advantage_over_the_baselines_is_paired_with_the_better_tuned_baseline():
    rows = []
    for training, evaluation in product(TRAIN_SEEDS['test'][:2], SPLITS['test']):
        common = dict(family='static', program=0, nodes=100, speed=0., phase='whole', training_seed=training,
                      evaluation_seed=evaluation, beta=.8)
        rows += [common | dict(variant='legacy-full-range', algorithm=LEARNED[0], gain=.40 + .01 * training),
                 common | dict(variant=RANGE_BASELINES, algorithm=BASELINES[0], gain=.30),
                 common | dict(variant=RANGE_BASELINES, algorithm=BASELINES[1], gain=.35 + .01 * training)]
    frame = pd.DataFrame(rows)
    assert figures.better_baseline(frame, RANGE_BASELINES) == BASELINES[1]
    point = figures.paired_advantage(frame, 'legacy-full-range', LEARNED[0], RANGE_BASELINES, BASELINES[1])
    # Paired by seeds: the difference is the same for every pair, so the interval collapses on it.
    assert np.allclose([point['mean'], point['low'], point['high']], .05)


def test_the_benefit_figures_render(tmp_path):
    data = gains(synthetic_episodes(FULL_RANGE, RANGE_BASELINES))
    figures.advantage_figure(data, tmp_path, FULL_RANGE, RANGE_BASELINES)
    figures.frontier_gains_figure(data, tmp_path, FULL_RANGE, CORE, RANGE_BASELINES)
    assert {path.name for path in tmp_path.glob('*.pdf')} == \
        {'advantage_over_baselines.pdf', 'frontier_gains_programs.pdf'}


def test_the_tuned_baselines_replace_the_untuned_ones():
    from analysis.campaign_paper import pareto_legend
    from analysis.methods import TRICKLE
    import matplotlib.pyplot as plt
    assert figures.untuned_points(figures.UNTUNED) and not figures.untuned_points(RANGE_BASELINES)
    fig = plt.figure()
    pareto_legend(fig, FULL_RANGE, RANGE_BASELINES)
    labels = [text.get_text() for text in fig.legends[0].texts]
    assert TRICKLE.label not in labels and 'Trickle, tuned by β' in labels
    plt.close('all')
