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
}
# Betas swept and focus beta of the variants that do not use the ratio scale of the original objective.
LINEAR_GRID = ([.2, .4, .6, .8, 1.], .8)
GRIDS = {'legacy-linear-awaiting': LINEAR_GRID, 'legacy-linear-patient': LINEAR_GRID,
         'legacy-full-range': ([.2, .4, .6, .8, .95], .8)}
# Trickle and the gradual controller tuned by beta: after a change they fall back to the best period of an urgent node
# on the linear scale instead of 0.5 s. Shared controls like the fixed periods, on the grid of the learned variants.
TUNED_BASELINES = 'tuned-baselines'
# On the full-range scale they move between the best periods of an unstable and of a stable node (BaselineTuning).
RANGE_BASELINES = 'range-baselines'
CONTROL_SETTINGS = {TUNED_BASELINES: dict(tunedBaselines=1), RANGE_BASELINES: dict(tunedBaselines=2, **FULL_RANGE_SHAPE)}
# The tuned baselines that go with a variant (TUNED_BASELINES by default).
TUNED_CONTROLS = {'legacy-full-range': RANGE_BASELINES}
FAMILIES = ('static', 'dynamic', 'generalization', 'beta-study', 'attention', 'filmstrip', 'learning')


def grid_of(variant):
    """The betas swept and the focus beta of `variant`."""
    return GRIDS.get(variant, (BETAS, FOCUS_BETA))


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
                    targetProgram=program, speed=speed,
                    episodeEvent='RegionStartsMoving' if family == 'dynamic' else 'SourceSwitch')
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


def jobs(split, variants, smoke=False, shared=True, finals=None, algorithms=LEARNED, families=None, overrides=None,
         beta_study_only=False, swept=(), controls=True, tuned_baselines=False):
    """Validation: every variant on the core scenarios at the focus beta. Test: `variants` (the control and the
    selections) on every scenario, the legacy control on the comparable ones, and the beta study of the final variant
    of each architecture (`finals`: algorithm -> variant, by default the last of `variants`). `algorithms` and
    `families` restrict a diagnostic campaign; `overrides` set a setting of every run, the controls included;
    `beta_study_only` keeps only the other betas of the finals (a supplement to a test campaign that has the rest);
    `swept` families run every variant at every beta (a validation of the trade-off curves); without `controls`, a
    test campaign leaves out the legacy control (another campaign of the same seeds has it); `tuned_baselines` adds
    the baselines tuned by beta on the grid of the last variant, swept where the variants are. Each variant sweeps its
    own grid of betas (`grid_of`)."""
    scenarios = SCENARIOS[split]
    if smoke:
        scenarios = [STATIC, *PROGRAMS]
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
        beta_scenarios = [s for s in CORE if s in scenarios]
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
                                       TUNED_CONTROLS.get(variants[-1], TUNED_BASELINES))


def prepare(split, variants, smoke=False, name=None, shared=True, finals=None, algorithms=LEARNED, families=None,
            overrides=None, beta_study_only=False, swept=(), controls=True, tuned_baselines=False):
    home = HOME / (name or ('smoke' if smoke else split))
    home.mkdir(parents=True, exist_ok=True)
    plan = list(jobs(split, variants, smoke, shared, finals, algorithms, families, overrides, beta_study_only, swept,
                     controls, tuned_baselines))
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
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
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
                         tuple(args.sweep_families), not args.without_controls, args.tuned_baselines)
    print(f'{len(plan)} runs: {home}', flush=True)
    if args.action == 'run':
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            order = execution_order(plan[:args.limit], args.first)
            success = list(executor.map(lambda job: execute(home, job), order))
        if not all(success):
            raise SystemExit('Campaign contains failed runs; inspect run.log')

if __name__ == '__main__':
    main()
