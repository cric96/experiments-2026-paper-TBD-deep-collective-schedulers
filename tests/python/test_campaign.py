import json
import numpy as np
import pandas as pd
import pytest
from collections import Counter
from experiments.campaign import (jobs, COMMON, SPLITS, TRAIN_SEEDS, BETAS, FOCUS_BETA, FIXED, BASELINES, VARIANTS,
                                  CORE, SCENARIOS, STATIC as STATIC_SCENARIO, archive_charts, validate)
from analysis.campaign import crossed_interval, gains, primary_scores


def test_disjoint_splits_and_paired_controls():
    assert not set(TRAIN_SEEDS['validation']) & set(TRAIN_SEEDS['test'])
    assert not set(SPLITS['test']) & set(SPLITS['validation'])
    plan = list(jobs('validation', ['current', 'transferable']))
    keys = [job['key'] for job in plan]
    assert len(keys) == len(set(keys))
    for job in plan:
        assert job['settings']['warmStart'] == 0
        assert job['settings']['trainingSpeeds'] == '0,0,1,0,0,2'
    assert {j['nodes'] for j in plan} == {100}
    assert {j['program'] for j in plan} == set(range(5))
    learned = [j for j in plan if not j['shared']]
    assert len({(j['training_seed'], j['evaluation_seed']) for j in learned}) == 8
    assert {j['beta'] for j in learned} == {FOCUS_BETA}


def test_shared_controls_run_once_per_evaluation_seed():
    plan = list(jobs('validation', ['current']))
    shared = [j for j in plan if j['shared']]
    assert all(j['variant'] == 'fixed' and j['training_seed'] == TRAIN_SEEDS['validation'][0] for j in shared)
    per_topology = Counter((j['family'], j['program'], j['speed'], j['evaluation_seed']) for j in shared)
    assert set(per_topology.values()) == {len(FIXED) + len(BASELINES)}
    assert not any(j['shared'] for j in plan if j['variant'] != 'fixed')


def test_test_split_covers_sizes_legacy_and_beta_study():
    plan = list(jobs('test', ['current', 'balanced']))
    assert {j['nodes'] for j in plan} == {50, 100, 200}
    betas = {(j['variant'], j['beta']) for j in plan if not j['shared']}
    assert {beta for variant, beta in betas if variant == 'balanced'} == set(BETAS)
    assert {beta for variant, beta in betas if variant in ('current', 'legacy')} == {FOCUS_BETA}
    assert {j['nodes'] for j in plan if j['variant'] == 'legacy'} == {100}
    # The reactive baselines ignore beta: once per topology, at the focus beta; the heuristic is no longer a control.
    assert {j['beta'] for j in plan if j['algorithm'] in BASELINES} == {FOCUS_BETA}
    assert {j['algorithm'] for j in plan if j['shared']} == {*FIXED, *BASELINES}


def write_run(directory, settings, rounds, error):
    """Synthetic exports of a run whose cumulative means are constant over every evaluation episode."""
    directory.mkdir(parents=True)
    total, training, length = settings['episodes'], settings['trainingEpisodes'], 200.
    columns = 'episode,training,time,roundsPerNodeSecond,normalizedError,reachableFraction,absoluteError'
    episodes = [f'{e},{str(e < training).lower()},{(e + 1) * length},{rounds},{error},1.0,{error}' for e in range(total)]
    (directory / 'run_episodes.csv').write_text('\n'.join([columns, *episodes]) + '\n')
    frames = [f'{e},{str(e < training).lower()},{e * length + elapsed},{rounds},{error},1.0,{error}'
              for e in range(total) for elapsed in range(0, 200, 2)]
    (directory / 'run_full.csv').write_text('\n'.join([columns, *frames]) + '\n')


def test_load_pairs_shared_controls_with_every_training_seed(tmp_path):
    from analysis.campaign import load
    plan = [j for j in jobs('validation', ['current']) if j['family'] == 'static' and j['evaluation_seed'] == 101
            and (j['shared'] or j['training_seed'] == 11) and j['beta'] == FOCUS_BETA]
    (tmp_path / 'manifest.json').write_text(json.dumps(plan))
    for index, job in enumerate(plan):
        write_run(tmp_path / 'exports' / job['key'], job['settings'], rounds=.2 + index / 10, error=1 / (index + 1))
    episodes = load(tmp_path)
    shared = episodes[episodes.variant == 'fixed']
    assert set(shared.training_seed) == set(TRAIN_SEEDS['validation'])
    assert set(episodes[episodes.variant == 'current'].training_seed) == {11}
    assert set(episodes.episode) == set(range(10, 20))
    whole = episodes[(episodes.phase == 'whole') & (episodes.algorithm == 3)]
    assert np.allclose(whole.roundsPerNodeSecond, whole.roundsPerNodeSecond.iloc[0])


def test_matched_update_budget():
    plan = list(jobs('validation', ['long-training', 'matched-updates']))
    budgets = {j['variant']: j['settings']['trainingEpisodes'] * j['settings']['stepsPerEpisode']
               for j in plan if j['variant'] != 'fixed'}
    assert budgets['long-training'] == budgets['matched-updates']


def test_crossed_uncertainty_not_episode_pseudoreplication():
    frame = pd.DataFrame([dict(training_seed=t, evaluation_seed=e, value=t + e)
                          for t in range(4) for e in range(4)])
    interval = crossed_interval(frame, 'value')
    assert interval['mean'] == 3
    assert interval['low'] < 3 < interval['high']
    frame.loc[0, 'value'] = np.nan
    assert not crossed_interval(frame, 'value')['complete']


def test_reject_incomplete_terminal_rows(tmp_path):
    (tmp_path / 'run_episodes.csv').write_text('episode,training,time\n0,true,200\n')
    assert not validate(tmp_path, {'episodes': 2, 'trainingEpisodes': 1})


def test_frontier_no_extrapolation_and_undefined_zero_error():
    rows = []
    for algorithm, rate in zip((10, 3, 4, 5, 6, 7), (2, 1, .5, 1/3, .25, .2)):
        rows.append(dict(family='static', program=0, nodes=100, speed=0, phase='whole', training_seed=11,
                         evaluation_seed=101, variant='fixed', algorithm=algorithm, beta=.95,
                         roundsPerNodeSecond=rate, normalizedError=1/rate, absoluteError=1,
                         reachableFraction=1))
    rows.append(dict(rows[0], algorithm=0, variant='current', roundsPerNodeSecond=3))
    result = gains(pd.DataFrame(rows))
    assert np.isnan(result.gain.iloc[0])
    with pytest.raises(ValueError):
        gains(pd.DataFrame(rows[1:]))


def test_selection_requires_paired_improvement():
    from analysis.campaign import select
    records = []
    for variant, gain in [('current', .1), ('transferable', .2)]:
        for training in (11, 12):
            for evaluation in (101, 102):
                for family in ('static', 'generalization'):
                    for algorithm in (0, 1):
                        records.append(dict(variant=variant, algorithm=algorithm, family=family, gain=gain,
                                            phase='whole', training_seed=training, evaluation_seed=evaluation,
                                            nodes=100, beta=FOCUS_BETA))
    data = pd.DataFrame(records)
    assert select(data)['variant'] == 'transferable'
    # Other betas and populations are reported, never selected on.
    noise = data[data.variant == 'current'].assign(gain=.9)
    assert select(pd.concat([data, noise.assign(beta=.5), noise.assign(nodes=200)]))['variant'] == 'transferable'
    data.loc[data.variant == 'transferable', 'gain'] = .05
    assert select(data)['variant'] == 'current'


def test_selection_per_architecture():
    from analysis.campaign import select
    records = [dict(variant=variant, algorithm=algorithm, family=family, phase='whole', training_seed=training,
                    evaluation_seed=evaluation, nodes=100, beta=FOCUS_BETA,
                    gain=.4 if (variant, algorithm) == ('balanced', 1) else .2 + training / 1000 + evaluation / 1e4)
               for variant in ('current', 'balanced') for algorithm in (0, 1) for family in ('static', 'generalization')
               for training in (11, 12, 13) for evaluation in (101, 102)]
    decision = select(pd.DataFrame(records))
    assert decision['byArchitecture'] == {0: 'current', 1: 'balanced'}


def test_confirmation_checks_the_validation_selection_without_reselecting():
    from analysis.campaign import confirm
    records = [dict(variant=variant, algorithm=algorithm, family=family, phase='whole', training_seed=training,
                    evaluation_seed=evaluation, nodes=100, beta=FOCUS_BETA,
                    gain=.4 if (variant, algorithm) == ('balanced', 1) else .2 + training / 1000 + evaluation / 1e4)
               for variant in ('current', 'balanced') for algorithm in (0, 1) for family in ('static', 'generalization')
               for training in (21, 22, 23) for evaluation in (201, 202)]
    data = pd.DataFrame(records)
    decision = confirm(data, {0: 'current', 1: 'balanced'})
    assert decision['selected'] == {0: 'current', 1: 'balanced'}
    assert decision['confirmed'] == {0: True, 1: True}
    # A selection that the independent seeds contradict is reported as such, never replaced.
    decision = confirm(data, {0: 'balanced', 1: 'current'})
    assert decision['selected'] == {0: 'balanced', 1: 'current'}
    assert decision['confirmed'] == {0: False, 1: False}


def test_test_plan_sweeps_beta_for_the_final_method_of_each_architecture():
    plan = list(jobs('test', ['current', 'balanced'], finals={0: 'current', 1: 'balanced'}))
    swept = {(j['variant'], j['algorithm']) for j in plan if not j['shared'] and j['beta'] != FOCUS_BETA}
    assert swept == {('current', 0), ('balanced', 1)}


def test_diagnostic_campaign_restricts_runs_and_overrides_the_controls_too():
    plan = list(jobs('validation', ['long-retention'], algorithms=(0,), families=('static', 'generalization'),
                     overrides=dict(retention=15.)))
    learned = [j for j in plan if not j['shared']]
    assert {j['algorithm'] for j in learned} == {0}
    assert {j['family'] for j in plan} == {'static', 'generalization'}
    assert all(j['settings']['retention'] == 15. for j in plan)
    assert any(j['shared'] for j in plan)


def test_execution_order_runs_the_chosen_families_first_and_keeps_the_manifest():
    from experiments.campaign import execution_order
    plan = list(jobs('test', ['current', 'balanced'], finals={0: 'current', 1: 'balanced'}))
    order = execution_order(plan, ('static',))
    static = [job for job in plan if job['family'] == 'static']
    assert order[:len(static)] == static
    assert sorted(job['key'] for job in order) == sorted(job['key'] for job in plan)
    assert execution_order(plan, ()) == plan


def test_test_campaign_overrides_every_run_legacy_and_beta_study_included():
    plan = list(jobs('test', ['current', 'balanced'], finals={0: 'current', 1: 'balanced'},
                     overrides=dict(retention=15.)))
    assert {j['settings']['retention'] for j in plan} == {15.}
    assert {j['variant'] for j in plan} == {'current', 'balanced', 'legacy', 'fixed'}


def test_beta_study_only_adds_the_other_betas_of_the_finals_to_a_test_campaign():
    finals = {0: 'legacy', 1: 'legacy'}
    plan = list(jobs('test', ['current', 'legacy'], shared=False, finals=finals, families=('static', 'generalization'),
                     beta_study_only=True))
    assert {(j['variant'], j['algorithm']) for j in plan} == {('legacy', 0), ('legacy', 1)}
    assert {j['beta'] for j in plan} == set(BETAS) - {FOCUS_BETA}
    assert {(j['family'], j['nodes']) for j in plan} == {('static', 100), ('generalization', 100)}
    assert len(plan) == 4 * 6 * 2 * 2 * 5


def test_paper_training_variants_run_the_original_learner_with_the_training_of_the_paper():
    plan = [j for j in jobs('validation', ['legacy-paper-steps', 'legacy-static-training', 'legacy-paper-training'],
                            shared=False)]
    settings = {(j['variant'], j['algorithm']): j['settings'] for j in plan}
    assert {s['algoId'] for (_, algorithm), s in settings.items() if algorithm == 0} == {9}
    assert {s['algoId'] for (_, algorithm), s in settings.items() if algorithm == 1} == {8}
    steps = {variant: s['stepsPerEpisode'] for (variant, _), s in settings.items()}
    assert steps == {'legacy-paper-steps': 110, 'legacy-static-training': 440, 'legacy-paper-training': 110}
    training = {variant: (s['trainingEvents'], s['trainingSpeeds']) for (variant, _), s in settings.items()}
    assert training['legacy-static-training'] == training['legacy-paper-training'] == ('SourceSwitch', '0')
    assert training['legacy-paper-steps'] == (COMMON['trainingEvents'], COMMON['trainingSpeeds'])


def test_every_variant_has_a_label():
    from analysis.figures.campaign import VARIANT_LABELS
    assert set(VARIANTS) <= set(VARIANT_LABELS)


def test_every_baseline_has_a_method():
    from analysis.figures.campaign import BASELINE_METHODS
    assert list(BASELINE_METHODS) == list(BASELINES)
    assert len({method.key for method in BASELINE_METHODS.values()}) == len(BASELINES)


def test_reward_shape_variants_reshape_the_original_learner_only():
    plan = list(jobs('validation', ['legacy', 'legacy-quiet', 'legacy-graded'], shared=False))
    shapes = {(j['variant'], j['settings']['algoId']): (j['settings'].get('steadyState'), j['settings'].get('intensityGain'))
              for j in plan}
    assert {algo for _, algo in shapes} == {8, 9}
    assert {shape for (variant, _), shape in shapes.items() if variant == 'legacy'} == {(None, None)}
    assert {shape for (variant, _), shape in shapes.items() if variant == 'legacy-quiet'} == {(0.01, None)}
    assert {shape for (variant, _), shape in shapes.items() if variant == 'legacy-graded'} == {(0.005, 1.0)}


def test_awaiting_variants_see_and_reward_a_missing_value():
    plan = list(jobs('validation', ['legacy-quiet-awaiting', 'legacy-graded-awaiting'], shared=False))
    assert {(j['settings']['missingValue'], j['settings']['missingValueIntensity']) for j in plan} == {(1, 1.0)}
    assert {j['settings']['algoId'] for j in plan} == {8, 9}


def test_a_validation_campaign_may_sweep_beta_on_some_families():
    plan = list(jobs('validation', ['legacy-quiet'], shared=False, swept=('static',)))
    betas = {(j['family'], j['beta']) for j in plan}
    assert {beta for family, beta in betas if family == 'static'} == set(BETAS)
    assert {beta for family, beta in betas if family != 'static'} == {FOCUS_BETA}


def test_a_test_campaign_without_controls_runs_the_finals_only():
    finals = {0: 'legacy-graded-awaiting', 1: 'legacy-graded-awaiting'}
    plan = list(jobs('test', ['legacy-graded-awaiting'], shared=False, finals=finals, controls=False))
    assert {j['variant'] for j in plan} == {'legacy-graded-awaiting'}
    focus = [j for j in plan if j['beta'] == FOCUS_BETA]
    swept = [j for j in plan if j['beta'] != FOCUS_BETA]
    pairs = len(TRAIN_SEEDS['test']) * len(SPLITS['test'])
    assert len(focus) == len(SCENARIOS['test']) * 2 * pairs
    assert len(swept) == len(CORE) * (len(BETAS) - 1) * 2 * pairs


def test_the_linear_variant_sweeps_its_own_grid_of_betas():
    from experiments.campaign import grid_of
    betas, focus = grid_of('legacy-linear-awaiting')
    assert betas == [.2, .4, .6, .8, 1.] and focus in betas
    assert grid_of('legacy-quiet-awaiting') == (BETAS, FOCUS_BETA)
    plan = list(jobs('validation', ['legacy-linear-awaiting'], shared=False, swept=('static',)))
    assert {j['beta'] for j in plan if j['family'] == 'static'} == set(betas)
    assert {j['beta'] for j in plan if j['family'] != 'static'} == {focus}
    assert {j['settings']['betaScale'] for j in plan} == {'Linear'}


def test_the_patient_linear_variant_sees_the_missing_value_without_hurrying():
    from experiments.campaign import grid_of
    plan = list(jobs('validation', ['legacy-linear-patient'], shared=False))
    settings = {key: {j['settings'][key] for j in plan} for key in ('betaScale', 'missingValue', 'missingValueIntensity')}
    assert settings == dict(betaScale={'Linear'}, missingValue={1}, missingValueIntensity={0.0})
    assert grid_of('legacy-linear-patient') == grid_of('legacy-linear-awaiting')


def test_the_full_range_variant_reaches_the_fast_periods_with_its_own_tuned_baselines():
    from experiments.campaign import grid_of
    plan = list(jobs('validation', ['legacy-full-range'], shared=False, swept=('static',), tuned_baselines=True))
    learned = [j for j in plan if j['variant'] == 'legacy-full-range']
    tuned = [j for j in plan if j['variant'] == 'range-baselines']
    betas, focus = grid_of('legacy-full-range')
    assert learned and tuned and max(betas) < 1 and focus in betas
    assert {(j['settings']['betaScale'], j['settings']['steadyState']) for j in learned + tuned} == {('FullRange', 1.0)}
    assert {j['settings']['tunedBaselines'] for j in tuned} == {2}
    assert {j['beta'] for j in tuned if j['family'] == 'static'} == set(betas)


def test_a_test_campaign_sweeps_the_grid_of_each_final():
    finals = {0: 'legacy-linear-awaiting', 1: 'legacy-linear-awaiting'}
    plan = list(jobs('test', ['legacy-linear-awaiting'], shared=False, finals=finals, controls=False))
    betas, focus = [.2, .4, .6, .8, 1.], .8
    assert {j['beta'] for j in plan} == set(betas)
    assert {j['family'] for j in plan if j['beta'] != focus} <= {scenario[0] for scenario in CORE}


def test_tuned_baselines_run_once_per_beta_of_the_grid_and_evaluation_seed():
    plan = list(jobs('validation', ['legacy-linear-awaiting'], shared=False, swept=('static',), tuned_baselines=True))
    tuned = [j for j in plan if j['variant'] == 'tuned-baselines']
    assert {j['algorithm'] for j in tuned} == set(BASELINES)
    assert all(j['shared'] and j['settings']['tunedBaselines'] == 1 for j in tuned)
    static = Counter((j['algorithm'], j['beta']) for j in tuned if j['family'] == 'static')
    assert set(static.values()) == {len(SPLITS['validation'])}
    assert {beta for _, beta in static} == {.2, .4, .6, .8, 1.}
    assert {j['beta'] for j in tuned if j['family'] != 'static'} == {.8}
    assert len({j['key'] for j in plan}) == len(plan)


def test_every_variant_is_compared_at_the_focus_beta_of_its_scale():
    from experiments.campaign import RANGE_BASELINES, TUNED_BASELINES, focus_of
    assert focus_of('fixed') == focus_of('legacy-quiet-awaiting') == FOCUS_BETA
    assert focus_of('legacy-full-range') == focus_of(RANGE_BASELINES) == .8
    assert focus_of('legacy-linear-awaiting') == focus_of(TUNED_BASELINES) == .8


def test_each_variant_has_the_baselines_tuned_on_its_scale():
    from experiments.campaign import RANGE_BASELINES, TUNED_BASELINES, grid_of, tuned_control_of
    assert tuned_control_of('legacy-full-range') == RANGE_BASELINES
    assert tuned_control_of('legacy-linear-awaiting') == TUNED_BASELINES
    for variant in ('legacy-full-range', 'legacy-linear-awaiting'):
        assert grid_of(tuned_control_of(variant)) == grid_of(variant)


def test_the_reactive_pilot_is_compared_with_baselines_tuned_on_its_own_objective():
    from experiments.campaign import (CONTROL_SETTINGS, REACTIVE_BASELINES, REACTIVE_GAIN, VARIANTS, focus_of,
                                      grid_of, tuned_control_of)
    variant = 'legacy-full-range-reactive'
    assert VARIANTS[variant] == VARIANTS['legacy-full-range'] | dict(intensityGain=REACTIVE_GAIN)
    assert tuned_control_of(variant) == REACTIVE_BASELINES
    assert CONTROL_SETTINGS[REACTIVE_BASELINES]['intensityGain'] == REACTIVE_GAIN
    assert grid_of(variant) == grid_of(REACTIVE_BASELINES) == grid_of('legacy-full-range')
    assert focus_of(variant) == .8


def test_plan_b_trains_the_full_range_learners_on_the_measured_error_like_the_original_ones():
    from experiments.campaign import (ERROR_REFERENCE_TIME, RANGE_BASELINES, STATIC, grid_of, settings_of,
                                      tuned_control_of)
    variant = 'error-full-range'
    for algorithm in (0, 1):
        settings = settings_of(variant, algorithm, .8, 11, 101, STATIC, smoke=False)
        original = settings_of('legacy-full-range', algorithm, .8, 11, 101, STATIC, smoke=False)
        # The learners of the measured costs, with the encoding, beta scale and replay of the original ones.
        assert settings['algoId'] == algorithm and original['algoId'] == {0: 9, 1: 8}[algorithm]
        assert settings['mobileEncoding'] == 0 and settings['completeReplay'] == 1 and settings['finalMultiplier'] == 0
        assert all(settings[key] == original[key] for key in ('betaScale', 'missingValue', 'stepsPerEpisode'))
        # The cost of the metrics: the error at the scale of the gradient, without tolerance.
        assert (settings['costSignal'], settings['lossScale'], settings['trainingTolerances']) == ('Error', 500., '0.0')
        assert settings['costReferenceTime'] == ERROR_REFERENCE_TIME
        # The original learners train in a static network (MobilityConfig.levelOf).
        assert settings['trainingSpeeds'] == '0'
    assert tuned_control_of(variant) == RANGE_BASELINES and grid_of(variant) == grid_of('legacy-full-range')


def test_the_neighbourhood_error_variant_differs_from_plan_b_only_in_the_cost_signal():
    from experiments.campaign import VARIANTS, grid_of, tuned_control_of
    variant = 'neighbourhood-error-full-range'
    assert VARIANTS[variant] == VARIANTS['error-full-range'] | dict(costSignal='Neighbourhood')
    assert tuned_control_of(variant) == tuned_control_of('error-full-range')
    assert grid_of(variant) == grid_of('error-full-range')


def test_the_network_error_variant_trains_on_the_global_objective_in_a_composite_episode():
    from experiments.campaign import DYNAMIC, NETWORK_VARIANTS, STATIC, grid_of, settings_of, tuned_control_of
    variant = 'network-error-full-range'
    neighbourhood = 'neighbourhood-error-full-range'
    assert VARIANTS[variant] == VARIANTS[neighbourhood] | dict(
        costSignal='Network', privilegedCritic=1, trainingDiagnostics=1, trainingEvents='SourceSwitchThenMotion',
        trainingSpeeds='1,2', motionDelay=50., trainingMotionDuration=30.)
    for other in NETWORK_VARIANTS:
        assert tuned_control_of(other) == tuned_control_of(neighbourhood) and grid_of(other) == grid_of(neighbourhood)
    for other in NETWORK_VARIANTS[:3]:
        assert {key: value for key, value in VARIANTS[other].items() if key != 'spatialDiscount'} == VARIANTS[variant]
    assert [VARIANTS[other].get('spatialDiscount', 1.) for other in NETWORK_VARIANTS[:3]] == [1., .5, .75]
    # The dynamic evaluation keeps the motion of the baselines: only the training motion is shorter.
    for scenario in (STATIC, DYNAMIC[0]):
        settings = settings_of(variant, 0, .8, 11, 101, scenario, smoke=False)
        assert settings['motionDuration'] == COMMON['motionDuration'] and settings['trainingMotionDuration'] == 30.


def test_each_contingency_of_the_network_error_changes_one_setting():
    from experiments.campaign import NETWORK_CONTINGENCIES
    for variant, settings in NETWORK_CONTINGENCIES.items():
        reference = VARIANTS['network-error-discount-0.5' if '0.5' in variant else 'network-error-full-range']
        changed = {key for key in settings.keys() | reference.keys() if settings.get(key) != reference.get(key)}
        expected = {'trainingEvents', 'trainingSpeeds', 'motionDelay', 'trainingMotionDuration'} \
            if variant.endswith('static-training') else {next(iter(changed))}
        assert changed == expected and len(changed) >= 1


def test_the_richer_variants_keep_the_privileged_critic_and_the_evaluation_episodes():
    from experiments.campaign import NETWORK_RICHER
    reference = VARIANTS['network-error-discount-0.5']
    for variant, settings in NETWORK_RICHER.items():
        assert settings['privilegedCritic'] == 1 and settings['spatialDiscount'] == .5
        assert settings.get('richContext', 0) == (1 if 'rich-context' in variant else 0)
        training = settings.get('trainingEpisodes', COMMON['trainingEpisodes'])
        assert training == (20 if 'more-data' in variant else COMMON['trainingEpisodes'])
        assert settings.get('episodes', COMMON['episodes']) - training == COMMON['episodes'] - COMMON['trainingEpisodes']
        assert {key for key in settings.keys() | reference.keys() if settings.get(key) != reference.get(key)} <= {
            'richContext', 'trainingEpisodes', 'episodes'}


def test_the_recovery_variants_lengthen_the_horizon_of_the_learners_and_of_the_cost_alike():
    from experiments.campaign import NETWORK_RECOVERY
    for variant, settings in NETWORK_RECOVERY.items():
        assert settings['gamma'] == settings['costDiscount'] == .95 and settings['extraSwitches'] == '50.0'
        assert settings['privilegedCritic'] == (0 if 'unprivileged' in variant else 1)


def test_the_hybrid_variants_add_the_full_range_surrogate_to_the_plain_critic_of_the_recovery_pilot():
    from experiments.campaign import FULL_RANGE_SHAPE, NETWORK_HYBRID, SURROGATE_WEIGHTS
    base = VARIANTS['network-error-discount-0.5-unprivileged-gamma-0.95-two-switches']
    assert [settings['surrogateWeight'] for settings in NETWORK_HYBRID.values()] == list(SURROGATE_WEIGHTS)
    for settings in NETWORK_HYBRID.values():
        assert settings['privilegedCritic'] == 0 and settings['costSignal'] == 'Network'
        assert settings == base | FULL_RANGE_SHAPE | dict(missingValueIntensity=1.0,
                                                          surrogateWeight=settings['surrogateWeight'])


def test_the_stable_hybrid_variants_train_twice_as_long_on_the_same_objective():
    from experiments.campaign import HYBRID_BASE, MORE_DATA, NETWORK_STABLE_HYBRID
    assert list(NETWORK_STABLE_HYBRID) == [f'network-error-hybrid-{weight}-more-data' for weight in ('0.5', '1', '2', '4')]
    for settings in NETWORK_STABLE_HYBRID.values():
        assert settings == HYBRID_BASE | MORE_DATA | dict(surrogateWeight=settings['surrogateWeight'])


def test_the_scheduled_hybrid_lets_beta_choose_the_surrogate_weight():
    from experiments.campaign import HYBRID_BASE, MORE_DATA
    assert VARIANTS['network-error-hybrid-scheduled'] == HYBRID_BASE | MORE_DATA | dict(surrogateWeight=.2,
                                                                                       highSurrogateWeight=2.)


def test_the_baseline_grid_runs_trickle_and_the_gradual_controller_once_per_topology_at_every_pair_of_periods():
    from experiments.campaign import BASELINE_PERIODS, baseline_grid_variant
    plan = list(jobs('validation', [], shared=False, baseline_grid=True))
    assert len(plan) == len(BASELINE_PERIODS) * len(CORE) * len(SPLITS['validation']) * len(BASELINES)
    assert len({job['key'] for job in plan}) == len(plan)
    for job in plan:
        settings = job['settings']
        assert job['algorithm'] in BASELINES and job['shared'] and settings['tunedBaselines'] == 0
        assert job['variant'] == baseline_grid_variant(settings['baselineMinPeriod'], settings['baselineMaxPeriod'])
    # Every pair moves between two periods; the untuned baselines are one of them.
    assert all(shortest < longest for shortest, longest in BASELINE_PERIODS) and (0.5, 5.0) in BASELINE_PERIODS


def test_a_test_campaign_runs_the_baseline_grid_on_the_core_scenarios_only():
    from experiments.campaign import BASELINE_PERIODS, baseline_grid_variant
    variant = 'network-error-hybrid-scheduled'
    plan = list(jobs('test', [variant], shared=False, finals={0: variant, 1: variant}, controls=False,
                     baseline_grid=True))
    grid = [job for job in plan if job['variant'] in {baseline_grid_variant(*pair) for pair in BASELINE_PERIODS}]
    assert len(grid) == len(BASELINE_PERIODS) * len(CORE) * len(SPLITS['test']) * len(BASELINES)
    assert {(j['family'], j['program'], j['nodes'], j['speed']) for j in grid} == set(CORE)
    assert {j['nodes'] for j in plan if j['variant'] == variant} == {50, 100, 200}


def test_a_composite_test_campaign_evaluates_the_composite_episode_of_the_training_at_every_beta():
    from experiments.campaign import COMPOSITE, BASELINE_PERIODS, baseline_grid_variant, grid_of
    variant = 'network-error-hybrid-scheduled'
    plan = list(jobs('test', [variant], finals={0: variant, 1: variant}, controls=False, tuned_baselines=True,
                     baseline_grid=True, composite=True))
    assert {(j['family'], j['program'], j['nodes'], j['speed']) for j in plan} == {COMPOSITE}
    assert {j['settings']['episodeEvent'] for j in plan} == {'SourceSwitchThenMotion'}
    # The evaluation moves every node from the instant at which the training of the network variants does.
    assert {j['settings']['motionDelay'] for j in plan} == {VARIANTS[variant]['motionDelay']}
    learned = [j for j in plan if j['variant'] == variant]
    pairs = len(TRAIN_SEEDS['test']) * len(SPLITS['test'])
    assert len(learned) == len(grid_of(variant)[0]) * 2 * pairs
    assert len([j for j in plan if j['variant'] == 'range-baselines']) == len(grid_of(variant)[0]) * 2 * 2
    grid = {baseline_grid_variant(*pair) for pair in BASELINE_PERIODS}
    assert len([j for j in plan if j['variant'] in grid]) == len(BASELINE_PERIODS) * 2 * 2


def test_the_composite_scenario_changes_only_the_evaluation_episode_of_a_learner():
    from experiments.campaign import COMPOSITE, settings_of
    for variant in ('network-error-hybrid-scheduled', 'legacy-full-range'):
        composite = settings_of(variant, 0, .8, 21, 201, COMPOSITE, False)
        static = settings_of(variant, 0, .8, 21, 201, STATIC_SCENARIO, False)
        changed = {name for name in composite.keys() | static.keys() if composite.get(name) != static.get(name)}
        assert changed <= {'episodeEvent', 'speed', 'motionDelay'}
