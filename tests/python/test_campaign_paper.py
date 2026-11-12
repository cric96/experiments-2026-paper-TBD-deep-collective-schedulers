import numpy as np
import pandas as pd
import pytest

from analysis.campaign_paper import (PERIOD_WINDOWS, ROUNDS_RANGE, motion_episodes, reward_at, rounds_whiskers,
                                     windowed)
from analysis.stats import Interval

SCHEDULE = 'SourceSwitch,SourceSwitch,RegionStartsMoving,SourceSwitch,SourceSwitch,RegionStartsMoving'


def test_reward_of_a_schedule_at_any_beta_matches_the_exported_reward():
    # A row of a campaign export at beta 0.95: rewardRate -0.060405...
    rows = pd.DataFrame(dict(normalizedError=[0.04417913306439097], roundsPerNodeSecond=[0.3687]))
    assert reward_at(rows, 0.95).iloc[0] == pytest.approx(-0.06040517641117143)
    assert reward_at(rows, 1.0).iloc[0] == pytest.approx(-0.04417913306439097)


def test_motion_episodes_follow_the_cyclic_training_schedule():
    job = dict(settings=dict(trainingEvents=SCHEDULE, trainingEpisodes=10))
    assert motion_episodes(job) == [3, 6, 9]


def test_the_episodes_of_a_longer_training_are_aligned_on_the_start_of_the_evaluation(tmp_path):
    from analysis.campaign_paper import episode_table
    from analysis.config import TRAINING_EPISODES
    jobs = []
    for key, training, total in (('paper', TRAINING_EPISODES, 2 * TRAINING_EPISODES),
                                 ('longer', 2 * TRAINING_EPISODES, 3 * TRAINING_EPISODES)):
        folder = tmp_path / 'exports' / key
        folder.mkdir(parents=True)
        pd.DataFrame(dict(episode=range(total), training=[e < training for e in range(total)], roundsPerNodeSecond=0.5,
                          normalizedError=0.1, absoluteError=10.0, recoverySeconds=20.0, recoveryCensored=0.0,
                          rewardRate=-0.1)).to_csv(folder / f'{key}_episodes.csv', index=False)
        jobs.append(dict(key=key, variant=key, settings=dict(trainingEpisodes=training, trainingEvents=SCHEDULE)))
    table = episode_table(tmp_path, jobs)
    evaluation = table[table.episode > TRAINING_EPISODES]
    assert (evaluation.groupby('key').size() == TRAINING_EPISODES).all()
    assert table[table.key == 'paper'].episode.tolist() == list(range(1, 2 * TRAINING_EPISODES + 1))
    assert table[table.key == 'longer'].episode.max() == 2 * TRAINING_EPISODES
    # The motion of a longer training is marked where its episodes are drawn.
    assert motion_episodes(jobs[1]) == [episode - TRAINING_EPISODES for episode in (3, 6, 9, 12, 15, 18)
                                        if episode > TRAINING_EPISODES]


def test_windows_are_recovered_from_the_running_means(tmp_path):
    length, step = 200.0, 2.0
    elapsed = np.arange(0.0, length, step)
    # Instantaneous rate 0.5 throughout; error 10 m in every window, then 30 m after 100 s.
    error = np.where(elapsed > 100.0, 30.0, 10.0)
    integral = np.concatenate([[0.0], np.cumsum(error[1:] * step)])
    running = np.divide(integral, elapsed, out=np.zeros_like(elapsed), where=elapsed > 0)
    rows = [dict(episode=episode, time=episode * length + t, roundsPerNodeSecond=0.5 if t > 0 else 0.0,
                 absoluteError=value) for episode in (0, 1) for t, value in zip(elapsed, running)]
    folder = tmp_path / 'exports' / 'run'
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / 'run_full.csv', index=False)
    windows = windowed(tmp_path, dict(key='run', settings=dict(trainingEpisodes=1)))
    assert windows.rate.dropna().to_numpy() == pytest.approx(0.5)
    assert windows.rate.isna().sum() == PERIOD_WINDOWS - 1
    assert windows.deviation.loc[50.0] == pytest.approx(10.0)
    assert windows.deviation.loc[150.0] == pytest.approx(30.0)


def test_round_intervals_are_clipped_to_the_reachable_range():
    low, high = rounds_whiskers(Interval(mean=5000.0, low=1000.0, high=60000.0, n=6))
    assert 5000.0 - low[0] == pytest.approx(ROUNDS_RANGE[0])
    assert 5000.0 + high[0] == pytest.approx(ROUNDS_RANGE[1])


def test_paper_table_repeats_the_fixed_periods_at_every_beta_with_their_reward():
    from analysis.campaign_paper_summary import paper_evaluation
    from experiments.campaign import BETAS
    rows = []
    for algorithm, beta, training in [(0, 0.95, 21), (0, 0.95, 22), (5, 0.95, 21)]:
        for evaluation_seed in (201, 202):
            for episode in (10, 11, 12):  # the first is a training episode (10 training episodes)
                rows.append(dict(variant='fixed' if algorithm == 5 else 'legacy', algorithm=algorithm, beta=beta,
                                 training_seed=training, evaluation_seed=evaluation_seed,
                                 episode=episode, rounds=6000.0, absoluteError=20.0, recoverySeconds=30.0,
                                 rewardRate=-0.05, normalizedError=0.04, roundsPerNodeSecond=0.3))
    table = paper_evaluation(pd.DataFrame(rows), training=10)
    learned = table[table.method == 'gnn']
    assert sorted(learned.seed) == [21, 22]
    fixed = table[table.method == 'fixed']
    # Every evaluation episode is a topology of its own: 2 seeds x 2 evaluation episodes, at every beta.
    assert len(fixed) == 4 * len(BETAS) and fixed.groupby('weight').seed.nunique().eq(4).all()
    assert fixed[fixed.weight == 1.0].reward.iloc[0] == pytest.approx(-0.04)
    assert set(fixed.period) == {3.0}


def test_paper_table_repeats_the_reactive_baselines_at_every_beta_with_their_reward():
    from analysis.campaign_paper_summary import paper_evaluation
    from experiments.campaign import BASELINES, BETAS
    rows = [dict(variant='fixed', algorithm=algorithm, beta=0.95, training_seed=21, evaluation_seed=evaluation_seed,
                 episode=episode, rounds=6000.0, absoluteError=20.0, recoverySeconds=30.0, rewardRate=-0.05,
                 normalizedError=0.04, roundsPerNodeSecond=0.3)
            for algorithm in (5, *BASELINES) for evaluation_seed in (201, 202) for episode in (10, 11, 12)]
    table = paper_evaluation(pd.DataFrame(rows), training=10)
    for key in ('trickle', 'gradual'):
        baseline = table[table.method == key]
        # They ignore beta and do not learn: every evaluation episode is a topology of its own, at every beta.
        assert len(baseline) == 4 * len(BETAS) and baseline.groupby('weight').seed.nunique().eq(4).all()
        assert baseline[baseline.weight == 1.0].reward.iloc[0] == pytest.approx(-0.04)
        assert baseline.period.isna().all()


def synthetic_paper_table(variant='legacy', tuned=None) -> pd.DataFrame:
    """Episodes 1-20 of the static scenario: learned methods (`variant`) per training seed and beta of their grid,
    shared controls once, and the reactive baselines `tuned` by beta at every beta of their grid."""
    from experiments.campaign import BASELINES, FIXED, LEARNED, TRAIN_SEEDS, grid_of
    rng = np.random.default_rng(5)
    periods = dict(zip(FIXED, (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)))
    schedules = [('fixed', algorithm, 0.95, 21, period) for algorithm, period in periods.items()]
    schedules += [('fixed', algorithm, 0.95, 21, 2.5) for algorithm in BASELINES]
    schedules += [(variant, algorithm, beta, seed, 1.5 + 2 * (1 - beta)) for algorithm in LEARNED
                  for beta in grid_of(variant)[0] for seed in TRAIN_SEEDS['test']]
    schedules += [(tuned, algorithm, beta, 21, 1.2 + 2.5 * (1 - beta)) for algorithm in BASELINES
                  for beta in grid_of(tuned)[0]] if tuned else []
    rows = []
    for (method, algorithm, beta, seed, period), evaluation_seed, episode in \
            ((schedule, evaluation, episode) for schedule in schedules for evaluation in (201, 202)
             for episode in range(1, 21)):
        noise = rng.uniform(0.9, 1.1)
        rows.append(dict(variant=method, algorithm=algorithm, beta=beta, training_seed=seed,
                         evaluation_seed=evaluation_seed, episode=episode, roundsPerNodeSecond=1 / period,
                         rounds=20000 / period, normalizedError=0.02 * period * noise,
                         absoluteError=25.0 * period * noise, recoverySeconds=15.0 * period * noise,
                         rewardRate=-beta * 0.02 * period * noise - (1 - beta) / period))
    return pd.DataFrame(rows)


def test_the_summary_figures_compare_the_reactive_baselines(tmp_path):
    from analysis.campaign_paper_summary import summary_figures, summary_methods
    finals = {0: 'legacy', 1: 'legacy'}
    assert [method.key for method in summary_methods(finals)] == ['gnn', 'td3', 'trickle', 'gradual']
    summary_figures(synthetic_paper_table(), finals, 10, tmp_path)
    assert {path.name for path in tmp_path.glob('*.pdf')} == \
        {'tradeoff_vs_beta.pdf', 'summary_beta095.pdf', 'frontier_gains_heatmap.pdf'}


def test_the_learning_curves_draw_the_reactive_baselines(tmp_path):
    from analysis.campaign_paper import curve_handles, learning_curves_metrics, learning_curves_reward
    from analysis.methods import GRADUAL, TRICKLE
    import matplotlib.pyplot as plt
    finals = {0: 'legacy', 1: 'legacy'}
    labels = [handle.get_label() for handle in curve_handles(finals, 'Fixed period')]
    assert {TRICKLE.label, GRADUAL.label} <= set(labels) and len(labels) == len(set(labels))
    learning_curves_reward(synthetic_paper_table(), finals, [3, 6, 9], 10, tmp_path)
    learning_curves_metrics(synthetic_paper_table(), finals, [3, 6, 9], tmp_path)
    assert {path.name for path in tmp_path.glob('*.pdf')} == \
        {'learning_curves_reward.pdf', 'learning_curves_metrics.pdf'}
    plt.close('all')


def test_a_censored_recovery_lasts_at_least_until_the_end_of_the_episode():
    from analysis.campaign import EPISODE_LENGTH
    from analysis.campaign_paper import EVENT_S, censored_recovery
    frame = pd.DataFrame(dict(recoverySeconds=[61.5, np.nan], recoveryCensored=[0.0, 1.0]))
    assert censored_recovery(frame).recoverySeconds.tolist() == [61.5, EPISODE_LENGTH - EVENT_S]


def test_a_full_range_campaign_draws_its_tuned_baselines_on_its_betas(tmp_path):
    from analysis.campaign_paper import learning_curves_metrics, learning_curves_reward, pareto_frontiers
    from analysis.campaign_paper_summary import paper_evaluation, summary_figures
    from experiments.campaign import RANGE_BASELINES, grid_of
    import matplotlib.pyplot as plt
    finals = {0: 'legacy-full-range', 1: 'legacy-full-range'}
    betas = grid_of('legacy-full-range')[0]
    table = synthetic_paper_table('legacy-full-range', RANGE_BASELINES)
    trickle = paper_evaluation(table, 10, RANGE_BASELINES, betas).query("method == 'trickle'")
    # Tuned at each beta, the baseline runs more rounds as beta grows.
    assert sorted(trickle.weight.unique()) == betas
    assert trickle.groupby('weight').ticks.mean().is_monotonic_increasing
    learning_curves_reward(table, finals, [3, 6, 9], 10, tmp_path, RANGE_BASELINES)
    learning_curves_metrics(table, finals, [3, 6, 9], tmp_path, RANGE_BASELINES)
    pareto_frontiers(table, finals, 10, tmp_path, RANGE_BASELINES)
    summary_figures(table, finals, 10, tmp_path, RANGE_BASELINES)
    # The single-beta summary is drawn at the focus beta of the scale, 0.8.
    assert {path.name for path in tmp_path.glob('*.pdf')} == \
        {'learning_curves_reward.pdf', 'learning_curves_metrics.pdf', 'pareto_frontiers.pdf', 'tradeoff_vs_beta.pdf',
         'summary_beta08.pdf', 'frontier_gains_heatmap.pdf'}
    plt.close('all')


def test_the_summary_labels_the_tuned_baselines_as_tuned():
    from analysis.campaign_paper_summary import summary_methods
    from experiments.campaign import RANGE_BASELINES
    shorts = [method.short for method in summary_methods({0: 'legacy-full-range', 1: 'legacy-full-range'},
                                                         RANGE_BASELINES)]
    assert shorts[-2:] == ['Trickle, tuned by β', 'Gradual, tuned by β']


def test_the_beta_note_of_a_pareto_panel_fits_in_the_narrowest_panel():
    from analysis import style
    from analysis.campaign_paper import pareto_panel
    from experiments.campaign import RANGE_BASELINES
    import matplotlib.pyplot as plt
    style.apply()
    fig, axes = style.figure("double", 5.4, 2, 3)
    table = synthetic_paper_table('legacy-full-range', RANGE_BASELINES)
    pareto_panel(axes[0, 0], table, 'normalizedError', {0: 'legacy-full-range', 1: 'legacy-full-range'}, 10,
                 RANGE_BASELINES)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel = axes[0, 0].get_window_extent(renderer)
    note = next(text for text in axes[0, 0].texts if 'left to right' in text.get_text())
    assert note.get_window_extent(renderer).x1 <= panel.x1
    plt.close('all')


def test_a_tuned_baseline_is_matched_to_the_rounds_of_the_learned_method():
    from analysis.campaign_paper_summary import matched_beta
    from experiments.campaign import BASELINES, RANGE_BASELINES
    table = synthetic_paper_table('legacy-full-range', RANGE_BASELINES)
    # The synthetic tuned baselines run 20000 / (1.2 + 2.5 (1 - beta)) rounds: 9091 at beta 0.6, 11765 at 0.8.
    assert matched_beta(table, BASELINES[0], RANGE_BASELINES, 20000 / 2.3, 10) == 0.6
    assert matched_beta(table, BASELINES[0], RANGE_BASELINES, 20000 / 1.9, 10) == 0.8


def test_the_overview_reads_a_baseline_at_its_matched_beta():
    from analysis.campaign_paper_summary import matched_label, pair_values
    from analysis.methods import TRICKLE
    from experiments.campaign import BASELINES, RANGE_BASELINES
    table = synthetic_paper_table('legacy-full-range', RANGE_BASELINES)
    values = pair_values(table, dict(algorithm=BASELINES[0], variant=RANGE_BASELINES, beta=0.6), 'rounds', 10)
    assert np.allclose(values, 20000 / 2.2)
    assert matched_label(TRICKLE, [0.6, 0.6]) == "Trickle at the GNN's rounds (β = 0.6)"
    assert matched_label(TRICKLE, [0.8, 0.6, 0.6]) == "Trickle at the GNN's rounds (β = 0.6–0.8)"


def test_the_rounds_whiskers_are_never_negative_for_a_mean_just_beyond_the_reachable_range():
    from analysis.campaign_paper import ROUNDS_RANGE, rounds_whiskers
    from analysis.stats import Interval
    slow = Interval(mean=ROUNDS_RANGE[0] - 16, low=ROUNDS_RANGE[0] - 22, high=ROUNDS_RANGE[0] - 10, n=4)
    fast = Interval(mean=ROUNDS_RANGE[1] + 5, low=ROUNDS_RANGE[1] - 5, high=ROUNDS_RANGE[1] + 15, n=4)
    # The interval clipped to the reachable range: the side beyond the range shrinks to the mean, the other remains.
    assert rounds_whiskers(slow) == [[0], [6]]
    assert rounds_whiskers(fast) == [[10], [0]]
