"""Prepare/run isolated, resumable static and zero-shot experiments.

python -m experiments.campaign prepare --split validation
python -m experiments.campaign run --split validation --workers 2
Test preparation requires a selection.json made by analysis.campaign on validation data.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / 'build/learning-campaign'
BETAS = [.5, .8, .95, .99, 1.]
# Selection and the cross-scenario comparisons use one trade-off; the beta study sweeps the others.
FOCUS_BETA = .95
# Disjoint seeds per split: validation selects, test confirms on new policies and new topologies.
TRAIN_SEEDS = {'validation': (11, 12, 13, 14), 'test': (21, 22, 23, 24, 25, 26)}
SPLITS = {'validation': (101, 102), 'test': (201, 202)}
LEARNED = (0, 1)  # GNN, MLP
# Fixed periods 0.5, 1, ..., 5 s and the reactive baselines: no learning, so their evaluation does not depend on the
# training seed (checked bit for bit); they run once per evaluation seed and are shared by every training seed.
FIXED = (10, 3, 4, 5, 6, 7)
# Trickle (RFC 6206) and the gradual (AIMD) controller of the period: they ignore beta, so they run at the focus beta
# only. They replace the heuristic (algoId 2) of the campaigns before 2026-11-04.
BASELINES = (11, 12)
# (family, program, nodes, speed): program codes follow BuildingBlock (0 gradient, 1 hop, 2 broadcast, 3 collect,
# 4 election); every policy is trained on the gradient only.
STATIC = ('static', 0, 100, 0.)
DYNAMIC = [('dynamic', 0, 100, 1.), ('dynamic', 0, 100, 2.)]
PROGRAMS = [('generalization', program, 100, 0.) for program in range(1, 5)]
SIZES = [('generalization', program, nodes, 0.) for program in range(5) for nodes in (50, 200)]
# Validation selects on the static network and on program transfer, and keeps one dynamic control.
CORE = [STATIC, DYNAMIC[0], *PROGRAMS]
SCENARIOS = {'validation': CORE, 'test': [STATIC, *DYNAMIC, *PROGRAMS, *SIZES]}
# Test: the legacy control only where the previous experiments can be compared; beta study on the core scenarios.
LEGACY_SCENARIOS = [STATIC, *DYNAMIC, *PROGRAMS]
# Seconds after the source switch (halfway) at which every node of a composite episode (SourceSwitchThenMotion) starts
# moving, in the training of the network-error variants and in the composite scenario.
COMPOSITE_MOTION_DELAY = 50.
# The composite episode as an evaluation scenario (test, on its own campaign): the source switches halfway, then every
# node moves at 1 m/s from COMPOSITE_MOTION_DELAY later to the end of the episode (motionDuration).
COMPOSITE = ('composite', 0, 100, 1.)
# The scenarios of the beta study of a test campaign and of the grid of the baselines.
COMPARED = [*CORE, COMPOSITE]
# What a family changes in the evaluation episodes: the event halfway (a source switch by default).
FAMILY_SETTINGS = {'dynamic': dict(episodeEvent='RegionStartsMoving'),
                   'composite': dict(episodeEvent='SourceSwitchThenMotion', motionDelay=COMPOSITE_MOTION_DELAY)}
# Same objective for feature ablations. Legacy control alone uses the original objective.
# Retention of the neighbours' data: with the message timing of Alchemist 25 (SendPreviousExport) an export arrives
# up to one period old and is replaced one period later, so it must outlast two of the longest periods (5 s); 15 s
# is the paper's value (the campaigns before 2026-11-04 used 6 s with the immediate send of Alchemist 45).
COMMON = dict(warmStart=0, mobilityModel='Waypoint', costSignal='Staleness', lossScale=50.,
              retention=15., trainingEvents='SourceSwitch,SourceSwitch,RegionStartsMoving,SourceSwitch,SourceSwitch,RegionStartsMoving',
              trainingSpeeds='0,0,1,0,0,2', motionDuration=50., trainingEpisodes=10, episodes=20,
              stepsPerEpisode=440, finalMultiplier=1, ownSpeed=1, neighbourSpeed=1,
              errorTolerance=0., exportDirectory='', completeReplay=0)
# The reward shape of the full-range scale (TradeOff.BetaScale.FullRange), shared by its learners and baselines.
FULL_RANGE_SHAPE = dict(betaScale='FullRange', steadyState=1.0)
# Instability gain of the reactive pilot (2026-11-06), four times the default 8 of TradeOff.RewardShape.
REACTIVE_GAIN = 32.
# Seconds for which the learners trained on the measured error pay a held normalized error of one as much as a round
# at beta 0.5 (MobilityConfig.costReferenceTime): on the fixed periods of the static gradient (validation-full-range)
# the best period then goes from 5 s at beta 0.2 to about 0.7 s at 0.95, as the full-range objective asks of a stable
# node (5 - 4.5 beta s); with 1 s it would stay at 5 s up to beta 0.5.
ERROR_REFERENCE_TIME = 0.25
VARIANTS = {
    'current': dict(unexplainedChange=1, dependants=1),
    'legacy': dict(legacy=True),
    'old-encoding': dict(mobileEncoding=0, ownSpeed=0, neighbourSpeed=0),
    'transferable': {},
    'balanced': dict(balanceScenarios=1),
    'log-exploration': dict(balanceScenarios=1, periodExploration='LogUniform'),
    'long-training': dict(balanceScenarios=1, periodExploration='LogUniform', trainingEpisodes=30, episodes=40),
    # 30 * 440 == 10 * 1320; same updates as long-training, fewer collected episodes.
    'matched-updates': dict(balanceScenarios=1, periodExploration='LogUniform', stepsPerEpisode=1320),
    # Transferable features without the base ones that assume the gradient's semantics (residual, children, order).
    'agnostic': dict(programAgnostic=1),
    # Diagnosis of the GNN transfer: `current` with one setting of the original experiment restored, then all three.
    # Cost and scale shape only the training costs; the retention changes the network, so its controls are its own.
    # Run with the 6 s retention of that time (--set retention=6.0): 15 s is now the default.
    'error-cost': dict(unexplainedChange=1, dependants=1, costSignal='Error'),
    'wide-loss-scale': dict(unexplainedChange=1, dependants=1, lossScale=500.),
    'long-retention': dict(unexplainedChange=1, dependants=1, retention=15.),
    'paper-settings': dict(unexplainedChange=1, dependants=1, costSignal='Error', lossScale=500., retention=15.),
    # Diagnosis of the gap between the original learner here and in the paper experiment (both with the Alchemist 25
    # timing): the paper trained it with its default 110 gradient steps per episode and on source switches only.
    'legacy-paper-steps': dict(legacy=True, stepsPerEpisode=110),
    'legacy-static-training': dict(legacy=True, trainingEvents='SourceSwitch', trainingSpeeds='0'),
    'legacy-paper-training': dict(legacy=True, stepsPerEpisode=110, trainingEvents='SourceSwitch', trainingSpeeds='0'),
    # The original learner with a reshaped reward (TradeOff.RewardShape, 2026-11-05). Its default steady-state pressure
    # (0.06) asks a stable node for 2.6 s at beta 0.95, while the reactive baselines sleep 5 s; its intensity gain (8)
    # asks an unstable node for about 0.5 s at every beta >= 0.8, so beta mostly tunes the rounds of stable nodes.
    # Quiet: stable nodes sleep longer (4.3 s at beta 0.95, 2 s at beta 1). Graded: stable nodes sleep almost 5 s up to
    # beta 0.95, and beta tunes the reaction of the unstable ones (2.7 s at beta 0.5, 0.7 s at 0.95).
    'legacy-quiet': dict(legacy=True, steadyState=0.01),
    'legacy-graded': dict(legacy=True, steadyState=0.005, intensityGain=1.0),
    # The same, and a node without a value yet (the field has not reached it) sees it in its state and counts as fully
    # unstable: with the other shapes it looks converged and sleeps through the start-up, where the baselines win.
    'legacy-quiet-awaiting': dict(legacy=True, steadyState=0.01, missingValue=1, missingValueIntensity=1.0),
    'legacy-graded-awaiting': dict(legacy=True, steadyState=0.005, intensityGain=1.0, missingValue=1,
                                   missingValueIntensity=1.0),
    # Beta as a linear dial (TradeOff.BetaScale.Linear): on the ratio scale the best period saturates (a node without
    # a value or unstable asks for 0.5-1 s at every beta >= 0.5, a stable one for about 5 s up to 0.95), so beta
    # changes little; on the linear scale an urgent node asks for 5 - 4.5 beta s and a stable one (urgency 0.2) for
    # 5 - 0.9 beta s, on its own grid of betas (GRIDS).
    'legacy-linear-awaiting': dict(legacy=True, betaScale='Linear', steadyState=0.2, missingValue=1,
                                   missingValueIntensity=1.0),
    # The same, but a node without a value only sees it in its state, without counting as unstable: it hurries only
    # when its output or its neighbourhood changes (as Trickle after a change), instead of spending rounds while it
    # waits for the field to reach it.
    'legacy-linear-patient': dict(legacy=True, betaScale='Linear', steadyState=0.2, missingValue=1,
                                  missingValueIntensity=0.0),
    # Beta over the whole range of periods (TradeOff.BetaScale.FullRange): on the other scales a stable node asks for
    # 4.1 s (linear) or 2 s (ratio) even at beta 1, so no beta comes near the fast fixed periods; here it asks for
    # 5 - 4.5 beta s (0.7 s at 0.95) and an unstable one divides its odds of sleeping by 1 + 8 I.
    'legacy-full-range': dict(legacy=True, **FULL_RANGE_SHAPE, missingValue=1, missingValueIntensity=1.0),
    # The same with a stronger instability gain (pilot, 2026-11-06): the learner is lazier than its objective at every
    # beta and loses the start-up at low beta and the events to the tuned baselines; an unstable node now divides its
    # odds of sleeping by 1 + 32 I (about 1 s instead of 1.9 s at beta 0.2), a stable one asks for the same period.
    'legacy-full-range-reactive': dict(legacy=True, **FULL_RANGE_SHAPE, missingValue=1, missingValueIntensity=1.0,
                                       intensityGain=REACTIVE_GAIN),
    # The full-range learners (original encoding, missing value, full-range beta) trained on the objective of the
    # evaluation instead of the instability surrogate (plan B, 2026-11-09): a decision pays the error of the output it
    # holds against the ground truth, normalized as in the metrics (500 m) and integrated until the next round, against
    # one round: -beta E / T - (1 - beta), the objective of the `current` learners without their encoding. Like the
    # original learners they train in a static network and keep every step.
    'error-full-range': dict(betaScale='FullRange', missingValue=1, mobileEncoding=0, costSignal='Error',
                             lossScale=500., costReferenceTime=ERROR_REFERENCE_TIME, trainingTolerances='0.0',
                             trainingSpeeds='0', completeReplay=1, finalMultiplier=0),
    # The same, paying the mean error of the node and of its neighbours (CostSignal.Neighbourhood): with its own error
    # alone (validation-error-full-range, static) a node sleeps 5 s as soon as its value is correct, although its
    # neighbours receive that value only at its next round, so the field spreads at about 5 s per hop.
    'neighbourhood-error-full-range': dict(betaScale='FullRange', missingValue=1, mobileEncoding=0,
                                           costSignal='Neighbourhood', lossScale=500.,
                                           costReferenceTime=ERROR_REFERENCE_TIME, trainingTolerances='0.0',
                                           trainingSpeeds='0', completeReplay=1, finalMultiplier=0),
}
# The full-range learners trained on the global objective (2026-11-10): the neighbourhood error charges only the
# neighbours, an order of magnitude less than the effect of a sleep on the whole network, so the learners covered only
# the low-rounds end of the front (up to 0.37 rounds/s). A decision now pays the errors of the network discounted by the
# spatial discount at every hop (CostSignal.Network: summed, divided by the effective size of the network, the energy
# divided alike), discounted within its sleep like the rewards that follow it; the replay is stratified by the node's
# own regime. Their critics also see the node's own loss, its charged cost and the mean loss of the network
# (privilegedCritic), never their actors. They train on a composite episode (SourceSwitchThenMotion): start-up and quiet
# until 100 s, a source switch at 100 s, every node moving at 1 or 2 m/s from 150 to 180 s, then still. The training
# motion lasts trainingMotionDuration, so that the dynamic evaluation keeps its motionDuration (50 s). Training
# diagnostics: <run>_training.csv.
NETWORK_ERROR = dict(VARIANTS['neighbourhood-error-full-range'], costSignal='Network', privilegedCritic=1,
                     trainingDiagnostics=1, trainingEvents='SourceSwitchThenMotion', trainingSpeeds='1,2',
                     motionDelay=COMPOSITE_MOTION_DELAY, trainingMotionDuration=30.)
# The spatial discounts of the pilot of the global objective (validation-network-error-discount).
NETWORK_DISCOUNTS = (.5, .75)
VARIANTS['network-error-full-range'] = NETWORK_ERROR
VARIANTS.update({f'network-error-discount-{discount:g}': dict(NETWORK_ERROR, spatialDiscount=discount)
                 for discount in NETWORK_DISCOUNTS})
NETWORK_VARIANTS = ('network-error-full-range', *(f'network-error-discount-{discount:g}' for discount in NETWORK_DISCOUNTS))
# The contingencies of the global objective (2026-11-10, validation-network-error-contingency), after no discount passed
# the first gate: the critic's best period barely depends on beta (about 3 s at every beta, the actor even lazier).
# With rho 1: three times the gradient steps (the actor lags behind the critic), the error twice as expensive against a
# round (cost reference time 0.1 s instead of 0.25 s), the squared loss on the convergence values (the Huber loss shrinks
# the rare large costs). With rho 0.5, which ablate the novelties against the neighbourhood error: no privileged critic,
# and the static training of the neighbourhood error instead of the composite episode.
NETWORK_HALF_DISCOUNT = VARIANTS['network-error-discount-0.5']
NETWORK_CONTINGENCIES = {
    'network-error-long-training': dict(NETWORK_ERROR, stepsPerEpisode=3 * COMMON['stepsPerEpisode']),
    'network-error-wide-price': dict(NETWORK_ERROR, costReferenceTime=.1),
    'network-error-squared-loss': dict(NETWORK_ERROR, convergenceLoss='Squared'),
    'network-error-discount-0.5-unprivileged': dict(NETWORK_HALF_DISCOUNT, privilegedCritic=0),
    'network-error-discount-0.5-static-training': dict(NETWORK_HALF_DISCOUNT, trainingEvents='SourceSwitch',
                                                       trainingSpeeds='0', motionDelay=0., trainingMotionDuration=-1.),
}
VARIANTS.update(NETWORK_CONTINGENCIES)
NETWORK_VARIANTS = (*NETWORK_VARIANTS, *NETWORK_CONTINGENCIES)
# Contingency C3, chosen by the user (2026-11-10) over dropping the privileged critic, with rho 0.5: a richer context of
# the critics (the share of the nodes the field has not reached and the age of the last change of the network), more
# data (20 training episodes instead of 10, the same 10 evaluation ones), and both.
MORE_DATA = dict(trainingEpisodes=2 * COMMON['trainingEpisodes'], episodes=COMMON['episodes'] + COMMON['trainingEpisodes'])
NETWORK_RICHER = {
    'network-error-discount-0.5-rich-context': dict(NETWORK_HALF_DISCOUNT, richContext=1),
    'network-error-discount-0.5-more-data': dict(NETWORK_HALF_DISCOUNT, **MORE_DATA),
    'network-error-discount-0.5-rich-context-more-data': dict(NETWORK_HALF_DISCOUNT, richContext=1, **MORE_DATA),
}
VARIANTS.update(NETWORK_RICHER)
NETWORK_VARIANTS = (*NETWORK_VARIANTS, *NETWORK_RICHER)
# Contingency C4 (2026-11-10), after C3: the recovery after a switch is 1.1-1.8 times the envelope's and resting nodes
# sleep 4-5 s at every beta. A longer horizon (gamma 0.95 per second for the learners and for the cost within a sleep,
# about 20 s instead of 10 s) and a second source switch in every training episode (at 50 s, before the one at 100 s),
# with rho 0.5 and the privileged critic, and without it for comparison.
LONG_HORIZON = dict(gamma=.95, costDiscount=.95, extraSwitches='50.0')
NETWORK_RECOVERY = {
    'network-error-discount-0.5-gamma-0.95-two-switches': dict(NETWORK_HALF_DISCOUNT, **LONG_HORIZON),
    'network-error-discount-0.5-unprivileged-gamma-0.95-two-switches': dict(NETWORK_HALF_DISCOUNT, privilegedCritic=0,
                                                                            **LONG_HORIZON),
}
VARIANTS.update(NETWORK_RECOVERY)
NETWORK_VARIANTS = (*NETWORK_VARIANTS, *NETWORK_RECOVERY)
# Fallback C8, chosen by the user (2026-11-10) after C1-C4: the plain critic of C4 keeps a gain of 1.2-1.7 over the
# envelope, but at about 0.3 rounds/s whatever beta. Its convergence cost becomes hybrid: the network error plus a weight
# times the convergence surrogate of the full-range reward shape, (1 + 8 I) u^2 with a missing value as I = 1
# (HybridConvergence), which tuned the rounds with beta (legacy-full-range, 0.26-0.91 rounds/s). For a resting node and
# an effective size of about 10, the weights 0.1, 0.3 and 1 move its best period over beta 0.2-0.95 from 5 to about
# 1.15, 0.8 and 0.6 s.
SURROGATE_WEIGHTS = (.1, .3, 1.)
HYBRID_BASE = dict(NETWORK_RECOVERY['network-error-discount-0.5-unprivileged-gamma-0.95-two-switches'],
                   **FULL_RANGE_SHAPE, missingValueIntensity=1.0)
NETWORK_HYBRID = {f'network-error-hybrid-{weight:g}': dict(HYBRID_BASE, surrogateWeight=weight)
                  for weight in SURROGATE_WEIGHTS}
VARIANTS.update(NETWORK_HYBRID)
NETWORK_VARIANTS = (*NETWORK_VARIANTS, *NETWORK_HYBRID)
# The hybrid pilot (2026-11-10): weight 1 gives a beta dial (0.4 to 1-1.4 rounds/s with a gain) but 7 of 20 runs of a
# seed and a beta collapse to the fastest period, weight 0.3 is stable without a dial. Against the collapses, chosen by
# the user: weights 0.5 and 1 with 20 training episodes (more data steadied C3).
# Then (validation-network-error-stable-hybrid) no weight collapses with 20 episodes, but the rounds span only 0.35-0.45
# (weight 0.5) and 0.4-0.8 per second (weight 1): weights 2 and 4 widen the dial, as the user chose.
STABLE_HYBRID_WEIGHTS = (.5, 1., 2., 4.)
NETWORK_STABLE_HYBRID = {f'network-error-hybrid-{weight:g}-more-data': dict(HYBRID_BASE, surrogateWeight=weight, **MORE_DATA)
                         for weight in STABLE_HYBRID_WEIGHTS}
VARIANTS.update(NETWORK_STABLE_HYBRID)
NETWORK_VARIANTS = (*NETWORK_VARIANTS, *NETWORK_STABLE_HYBRID)
# The weight sets a band of rounds and beta moves only about 1.5 times within it (validation-network-error-wide-hybrid;
# weight 4 collapses 12 runs of 20): beta chooses the weight too, as the user chose, geometrically from 0.2 at the
# lowest beta of the grid to 2 at the highest one (SurrogateWeights), with 20 training episodes.
NETWORK_SCHEDULED_HYBRID = {'network-error-hybrid-scheduled': dict(HYBRID_BASE, surrogateWeight=.2,
                                                                   highSurrogateWeight=2., **MORE_DATA)}
VARIANTS.update(NETWORK_SCHEDULED_HYBRID)
NETWORK_VARIANTS = (*NETWORK_VARIANTS, *NETWORK_SCHEDULED_HYBRID)
# Trickle and the gradual controller tuned by beta: after a change they fall back to the best period of an urgent node
# on the linear scale instead of 0.5 s. Shared controls like the fixed periods, on the grid of the learned variants.
TUNED_BASELINES = 'tuned-baselines'
# On the full-range scale they move between the best periods of an unstable and of a stable node (BaselineTuning).
RANGE_BASELINES = 'range-baselines'
# The same, between the best periods of the objective of the reactive pilot.
REACTIVE_BASELINES = 'reactive-baselines'
CONTROL_SETTINGS = {TUNED_BASELINES: dict(tunedBaselines=1),
                    RANGE_BASELINES: dict(tunedBaselines=2, **FULL_RANGE_SHAPE),
                    REACTIVE_BASELINES: dict(tunedBaselines=2, **FULL_RANGE_SHAPE, intensityGain=REACTIVE_GAIN)}
# The tuned baselines that go with a variant (TUNED_BASELINES by default).
TUNED_CONTROLS = {'legacy-full-range': RANGE_BASELINES, 'legacy-full-range-reactive': REACTIVE_BASELINES,
                  'error-full-range': RANGE_BASELINES, 'neighbourhood-error-full-range': RANGE_BASELINES,
                  **{variant: RANGE_BASELINES for variant in NETWORK_VARIANTS}}
# Trickle and the gradual controller on a grid of their two periods (2026-11-09), whatever beta: the shortest one, after
# a change, and the longest one, when stable (s). Tuned by beta they move on one curve of this grid set by the
# surrogate objective, not on its best front (at beta 0.2 the untuned ones, 0.5-5 s, have a lower error with the same
# rounds); the lower envelope of the grid is the best front that these baselines reach.
BASELINE_SHORTEST_PERIODS = (.5, 1., 1.5, 2., 3.)
BASELINE_LONGEST_PERIODS = (1., 1.5, 2., 3., 4., 5.)
BASELINE_PERIODS = [(shortest, longest) for shortest in BASELINE_SHORTEST_PERIODS
                    for longest in BASELINE_LONGEST_PERIODS if shortest < longest]


def baseline_grid_variant(shortest, longest):
    """The control of the reactive baselines that move between `shortest` and `longest` (s)."""
    return f'baseline-periods-{shortest:g}-{longest:g}'


CONTROL_SETTINGS.update({baseline_grid_variant(shortest, longest):
                         dict(tunedBaselines=0, baselineMinPeriod=shortest, baselineMaxPeriod=longest)
                         for shortest, longest in BASELINE_PERIODS})
# Betas swept and focus beta of the variants (and of their tuned baselines) that do not use the ratio scale of the
# original objective.
LINEAR_GRID = ([.2, .4, .6, .8, 1.], .8)
FULL_RANGE_GRID = ([.2, .4, .6, .8, .95], .8)
GRIDS = {'legacy-linear-awaiting': LINEAR_GRID, 'legacy-linear-patient': LINEAR_GRID, TUNED_BASELINES: LINEAR_GRID,
         'legacy-full-range': FULL_RANGE_GRID, RANGE_BASELINES: FULL_RANGE_GRID,
         'legacy-full-range-reactive': FULL_RANGE_GRID, REACTIVE_BASELINES: FULL_RANGE_GRID,
         'error-full-range': FULL_RANGE_GRID, 'neighbourhood-error-full-range': FULL_RANGE_GRID,
         **{variant: FULL_RANGE_GRID for variant in NETWORK_VARIANTS}}
FAMILIES = ('static', 'dynamic', 'generalization', 'beta-study', 'attention', 'filmstrip', 'learning')


def grid_of(variant):
    """The betas swept and the focus beta of `variant`."""
    return GRIDS.get(variant, (BETAS, FOCUS_BETA))


def focus_of(variant):
    """The beta at which `variant` is compared with the other methods (the shared controls run at FOCUS_BETA)."""
    return grid_of(variant)[1]


def tuned_control_of(variant):
    """Trickle and the gradual controller tuned by beta on the scale of `variant`."""
    return TUNED_CONTROLS.get(variant, TUNED_BASELINES)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_charts(charts=ROOT / 'charts'):
    """Move current outputs once, preserve existing archives, record exact source evidence."""
    from datetime import datetime, timezone
    sources = [p for p in charts.iterdir() if p.name != 'archive'] if charts.exists() else []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = charts / 'archive' / stamp
    destination.mkdir(parents=True)
    manifest = {'files': {}, 'exports': {}, 'gitDiff': 'working-tree.diff',
                'sourceNote': 'Working tree at archive time; original figure-generation revision is not verified.'}
    for source in sources:
        for path in ([source] if source.is_file() else source.rglob('*')):
            if path.is_file():
                manifest['files'][str(path.relative_to(charts))] = digest(path)
        shutil.move(str(source), destination / source.name)
    for path in (ROOT / 'build/exports').rglob('*_episodes.csv'):
        manifest['exports'][str(path.relative_to(ROOT))] = digest(path)
    (destination / 'working-tree.diff').write_bytes(subprocess.check_output(['git', 'diff'], cwd=ROOT))
    # Untracked experiment code is part of the evidence, too.
    for folder in ('src', 'analysis', 'experiments'):
        shutil.copytree(ROOT / folder, destination / 'source' / folder,
                        ignore=shutil.ignore_patterns('__pycache__'))
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for name in FAMILIES:
        (charts / name).mkdir(parents=True, exist_ok=True)
    return destination


def settings_of(variant, algorithm, beta, training_seed, evaluation_seed, scenario, smoke, overrides=None):
    family, program, nodes, speed = scenario
    settings = {**COMMON, **(overrides or {}), **VARIANTS.get(variant, CONTROL_SETTINGS.get(variant, {}))}
    legacy = settings.pop('legacy', False)
    settings.update(random=training_seed, evaluationSeed=evaluation_seed, evaluationNodes=nodes,
                    algoId=({0: 9, 1: 8}[algorithm] if legacy else algorithm), weight=beta,
                    targetProgram=program, speed=speed, episodeEvent='SourceSwitch')
    settings.update(FAMILY_SETTINGS.get(family, {}))
    if smoke:
        settings.update(trainingEpisodes=1, episodes=2, episodeLength=20., stepsPerEpisode=1, motionDuration=5.)
    return settings


def job(split, variant, algorithm, beta, training_seed, evaluation_seed, scenario, smoke=False, shared=False,
        overrides=None):
    family, program, nodes, speed = scenario
    key = f'{family}-p{program}-n{nodes}-v{speed}-{variant}-a{algorithm}-b{beta}-t{training_seed}-e{evaluation_seed}'
    return dict(key=key, split=split, variant=variant, family=family, program=program, nodes=nodes,
                algorithm=algorithm, beta=beta, training_seed=training_seed, evaluation_seed=evaluation_seed,
                speed=speed, shared=shared,
                settings=settings_of(variant, algorithm, beta, training_seed, evaluation_seed, scenario, smoke,
                                     overrides))


def seeds_of(split, smoke):
    """Training and evaluation seeds: a smoke test uses the first pair only."""
    training, evaluation = TRAIN_SEEDS[split], SPLITS[split]
    return (training[:1], evaluation[:1]) if smoke else (training, evaluation)


def learned_jobs(split, variant, scenarios, betas, smoke=False, algorithms=LEARNED, overrides=None):
    training, evaluation = seeds_of(split, smoke)
    for scenario, beta, training_seed, evaluation_seed, algorithm in product(
            scenarios, betas, training, evaluation, algorithms):
        yield job(split, variant, algorithm, beta, training_seed, evaluation_seed, scenario, smoke,
                  overrides=overrides)


def shared_jobs(split, scenarios, smoke=False, overrides=None, algorithms=(*FIXED, *BASELINES)):
    """Fixed periods and reactive baselines, once per evaluation seed (under the first training seed of the split)."""
    seed = TRAIN_SEEDS[split][0]
    for scenario, evaluation_seed in product(scenarios, seeds_of(split, smoke)[1]):
        for algorithm in algorithms:
            yield job(split, 'fixed', algorithm, FOCUS_BETA, seed, evaluation_seed, scenario, smoke, shared=True,
                      overrides=overrides)


def tuned_baseline_jobs(split, scenarios, swept_scenarios, grid, smoke=False, overrides=None, control=TUNED_BASELINES):
    """Trickle and the gradual controller tuned by beta (as `control`), once per evaluation seed: every beta of `grid`
    (betas, focus) on `swept_scenarios`, the focus beta on the others."""
    betas, focus = grid
    seed = TRAIN_SEEDS[split][0]
    for scenario in scenarios:
        for beta, evaluation_seed, algorithm in product(betas if scenario in swept_scenarios else [focus],
                                                        seeds_of(split, smoke)[1], BASELINES):
            yield job(split, control, algorithm, beta, seed, evaluation_seed, scenario, smoke, shared=True,
                      overrides=overrides)


def baseline_grid_jobs(split, scenarios, smoke=False, overrides=None):
    """Trickle and the gradual controller at every pair of periods of the grid, once per evaluation seed (they do not
    learn: the training seed changes nothing), at the focus beta, which they ignore."""
    seed = TRAIN_SEEDS[split][0]
    for scenario, (shortest, longest), evaluation_seed, algorithm in product(
            scenarios, BASELINE_PERIODS, seeds_of(split, smoke)[1], BASELINES):
        yield job(split, baseline_grid_variant(shortest, longest), algorithm, FOCUS_BETA, seed, evaluation_seed,
                  scenario, smoke, shared=True, overrides=overrides)


def jobs(split, variants, smoke=False, shared=True, finals=None, algorithms=LEARNED, families=None, overrides=None,
         beta_study_only=False, swept=(), controls=True, tuned_baselines=False, baseline_grid=False, composite=False):
    """Validation: every variant on the core scenarios at the focus beta. Test: `variants` (the control and the
    selections) on every scenario, the legacy control on the comparable ones, and the beta study of the final variant
    of each architecture (`finals`: algorithm -> variant, by default the last of `variants`). `algorithms` and
    `families` restrict a diagnostic campaign; `overrides` set a setting of every run, the controls included;
    `beta_study_only` keeps only the other betas of the finals (a supplement to a test campaign that has the rest);
    `swept` families run every variant at every beta (a validation of the trade-off curves); without `controls`, a
    test campaign leaves out the legacy control (another campaign of the same seeds has it); `tuned_baselines` adds
    the baselines tuned by beta on the grid of the last variant, swept where the variants are; `baseline_grid` the
    baselines on the grid of their periods (BASELINE_PERIODS), on the core scenarios only; `composite` replaces the
    scenarios with the composite one (COMPOSITE). Each variant sweeps its own grid of betas (`grid_of`)."""
    scenarios = [COMPOSITE] if composite else [STATIC, *PROGRAMS] if smoke else SCENARIOS[split]
    if families:
        scenarios = [scenario for scenario in scenarios if scenario[0] in families]
    for variant in ([] if beta_study_only else variants):
        for scenario in scenarios:
            grid, focus = grid_of(variant)
            betas = grid if scenario[0] in swept else [focus]
            yield from learned_jobs(split, variant, [scenario], betas, smoke, algorithms, overrides)
    beta_scenarios = []
    if split == 'test':
        if controls and 'legacy' not in variants and not beta_study_only:
            yield from learned_jobs(split, 'legacy', [s for s in LEGACY_SCENARIOS if s in scenarios], [FOCUS_BETA], smoke,
                                    overrides=overrides)
        beta_scenarios = [s for s in COMPARED if s in scenarios]
        for algorithm in LEARNED:
            final = (finals or {}).get(algorithm, variants[-1])
            grid, focus = grid_of(final)
            others = [beta for beta in grid if beta != focus]
            yield from learned_jobs(split, final, beta_scenarios, others, smoke, (algorithm,), overrides)
    if shared:
        yield from shared_jobs(split, scenarios, smoke, overrides)
    if tuned_baselines:
        swept_scenarios = [s for s in scenarios if s[0] in swept or s in beta_scenarios]
        yield from tuned_baseline_jobs(split, scenarios, swept_scenarios, grid_of(variants[-1]), smoke, overrides,
                                       tuned_control_of(variants[-1]))
    if baseline_grid:
        yield from baseline_grid_jobs(split, [s for s in scenarios if s in COMPARED], smoke, overrides)


def prepare(split, variants, smoke=False, name=None, shared=True, finals=None, algorithms=LEARNED, families=None,
            overrides=None, beta_study_only=False, swept=(), controls=True, tuned_baselines=False, baseline_grid=False,
            composite=False):
    home = HOME / (name or ('smoke' if smoke else split))
    home.mkdir(parents=True, exist_ok=True)
    plan = list(jobs(split, variants, smoke, shared, finals, algorithms, families, overrides, beta_study_only, swept,
                     controls, tuned_baselines, baseline_grid, composite))
    manifest = home / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != plan:
            raise ValueError('Existing manifest differs; use a fresh campaign directory')
        return home, plan
    # Snapshot compiled runtime and sources so later changes cannot silently alter resumed runs.
    runtime = json.loads((ROOT / 'build/campaign-runtime.json').read_text())
    entries = []
    for i, entry in enumerate(runtime['classpath'].split(os.pathsep)):
        path = Path(entry)
        if path.is_dir() and path.is_relative_to(ROOT / 'build'):
            target = home / 'runtime' / str(i)
            shutil.copytree(path, target)
            entries.append(str(target))
        else:
            entries.append(entry)
    runtime['classpath'] = os.pathsep.join(entries)
    (home / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    shutil.copy2(ROOT / 'src/main/yaml/simulationMobility.yml', home / 'simulation.yml')
    (home / 'source.diff').write_bytes(subprocess.check_output(['git', 'diff'], cwd=ROOT))
    for folder in ('src', 'analysis', 'experiments'):
        shutil.copytree(ROOT / folder, home / 'source' / folder, ignore=shutil.ignore_patterns('__pycache__'))
    hashes = {str(path.relative_to(home)): digest(path) for path in (home / 'source').rglob('*') if path.is_file()}
    (home / 'source-checksums.json').write_text(json.dumps(hashes, indent=2))
    manifest.write_text(json.dumps(plan, indent=2))
    return home, plan


def execution_order(plan, first):
    """The runs of the `first` families before the others, each group in the declared order (the manifest keeps it)."""
    return sorted(plan, key=lambda job: job['family'] not in first)


def validate(directory, settings):
    import csv
    paths = list(directory.glob('*_episodes.csv'))
    if len(paths) != 1:
        return False
    with paths[0].open() as stream:
        rows = list(csv.DictReader(stream))
    total, training = settings['episodes'], settings['trainingEpisodes']
    length = settings.get('episodeLength', 200.)
    return (len(rows) == total and [int(r['episode']) for r in rows] == list(range(total))
            and [r['training'] == 'true' for r in rows] == [i < training for i in range(total)]
            and abs(float(rows[-1]['time']) - total * length) < 1e-7)


def execute(home, job):
    directory = home / 'exports' / job['key']
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / 'status.json'
    if status_path.exists() and json.loads(status_path.read_text())['state'] == 'complete' and validate(directory, job['settings']):
        return True
    runtime = json.loads((home / 'runtime.json').read_text())
    settings = dict(job['settings'], exportDirectory=str(directory))
    override = dict(variables={k: dict(type='ArbitraryVariable', parameters=[v, [v]]) for k, v in settings.items()},
                    launcher=dict(type='DefaultLauncher', parameters=dict(batch=[])))
    seed = settings['random']
    override['variables']['random'] = dict(min=seed, max=seed + 1, step=1, default=seed)
    command = [runtime['java'], '-Xmx3g', '-XX:ActiveProcessorCount=2',
               '-Dai.djl.pytorch.num_threads=1', '-Dai.djl.pytorch.num_interop_threads=1',
               '-cp', runtime['classpath'], 'it.unibo.alchemist.Alchemist', 'run', str(home / 'simulation.yml'),
               '--override', json.dumps(override)]
    (directory / 'command.json').write_text(json.dumps(command, indent=2))
    status_path.write_text(json.dumps(dict(state='running')))
    with (directory / 'run.log').open('w') as output:
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                                stdout=output, stderr=subprocess.STDOUT)
    valid = result.returncode == 0 and validate(directory, settings)
    status_path.write_text(json.dumps(dict(state='complete' if valid else 'failed', returncode=result.returncode)))
    print(f'{job["key"]}: {"complete" if valid else "FAILED"}', flush=True)
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['archive', 'prepare', 'run'])
    parser.add_argument('--split', choices=SPLITS, default='validation')
    parser.add_argument('--variants', nargs='*', choices=VARIANTS, default=list(VARIANTS),
                        help='Learned variants (none, with no value: e.g. a campaign of baselines only)')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--limit', type=int, help='Run only the first N declared jobs; manifest remains complete')
    parser.add_argument('--name', help='Campaign folder (default: the split, or smoke)')
    parser.add_argument('--without-shared', action='store_true',
                        help='Only the learned variants: the analysis takes the shared controls from another campaign')
    parser.add_argument('--selection', default=str(HOME / 'validation/selection.json'),
                        help='Validation decision that locks the test variants')
    parser.add_argument('--algorithms', nargs='+', type=int, choices=LEARNED, default=list(LEARNED),
                        help='Learned architectures to run (0 GNN, 1 MLP)')
    parser.add_argument('--families', nargs='+', choices=('static', 'dynamic', 'generalization'),
                        help='Only the scenarios of these families')
    parser.add_argument('--first', nargs='+', default=[], choices=('static', 'dynamic', 'generalization'),
                        help='Run the scenarios of these families before the others (e.g. static, for the beta study)')
    parser.add_argument('--beta-study-only', action='store_true',
                        help='Test: only the other betas of the selected variants (with --without-shared)')
    parser.add_argument('--sweep-families', nargs='+', default=[], choices=('static', 'dynamic', 'generalization'),
                        help='Validation: run every variant at every beta on these families (trade-off curves)')
    parser.add_argument('--without-controls', action='store_true',
                        help='Test: only the selected variants, not the current and legacy controls (a campaign of '
                             'the same seeds has them)')
    parser.add_argument('--tuned-baselines', action='store_true',
                        help='Also Trickle and the gradual controller tuned by beta, on the grid of the variants')
    parser.add_argument('--baseline-grid', action='store_true',
                        help='Also Trickle and the gradual controller on the grid of their periods (BASELINE_PERIODS)')
    parser.add_argument('--composite', action='store_true',
                        help='Only the composite scenario (COMPOSITE: source switch, then every node moves), with the '
                             'beta study on it')
    parser.add_argument('--set', dest='overrides', nargs='+', default=[], metavar='NAME=VALUE',
                        help='Setting of every run of the campaign, the shared controls included (e.g. retention=15)')
    args = parser.parse_args()
    if args.action == 'archive':
        print(archive_charts()); return
    variants, finals = args.variants, None
    if args.split == 'test' and not args.smoke:
        selection = json.loads(Path(args.selection).read_text())
        finals = {int(algorithm): variant for algorithm, variant in
                  selection.get('byArchitecture', {str(a): selection['variant'] for a in LEARNED}).items()}
        # The control first, then the selections.
        variants = list(dict.fromkeys([*([] if args.without_controls else ['current']), *finals.values()]))
    overrides = {name: json.loads(value) for name, value in (item.split('=', 1) for item in args.overrides)}
    home, plan = prepare(args.split, variants, args.smoke, args.name, not args.without_shared, finals,
                         tuple(args.algorithms), args.families, overrides or None, args.beta_study_only,
                         tuple(args.sweep_families), not args.without_controls, args.tuned_baselines,
                         args.baseline_grid, args.composite)
    print(f'{len(plan)} runs: {home}', flush=True)
    if args.action == 'run':
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            order = execution_order(plan[:args.limit], args.first)
            success = list(executor.map(lambda job: execute(home, job), order))
        if not all(success):
            raise SystemExit('Campaign contains failed runs; inspect run.log')

if __name__ == '__main__':
    main()
