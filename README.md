# Learning when to run aggregate programs

Experiments for the TETCI paper on **learning the scheduling of aggregate-computing rounds**: every node of a
network runs a self-stabilizing [ScaFi](https://scafi.github.io/) program on the
[Alchemist](https://alchemistsimulator.github.io/) simulator and, after each round, a policy shared by all the nodes
decides how long the node sleeps before the next one. Waking up often makes the network converge quickly after a
change; sleeping saves energy. The policies are learned with tabular, offline deep and graph-based reinforcement
learning, and compared with fixed periods and a hand-crafted heuristic.

## The experiment

- **Network.** 100 nodes on a randomly perturbed 10 × 10 grid of 500 m × 500 m, communicating within 75 m.
- **Programs.** The nodes compute a *gradient* (distance from a source). The source moves to a far node every 100 s,
  so the network periodically has to re-converge. The generalization experiment also uses the *hop-count gradient*,
  *broadcast*, *collect* and sparse *leader election* (S) building blocks.
- **Episodes.** A run lasts 20 episodes of 200 s; every episode starts on a new deployment. Policies learn during the
  first 10 episodes (with decaying exploration) and are only exploited afterwards.
- **Actions.** The next wake-up period: one of 1–5 s (discrete learners) or any value in [0.5, 5] s (continuous ones).
- **Objective.** A trade-off weight β ∈ {0.5, 0.8, 0.95, 0.99, 1} balances convergence and energy. With the
  normalized period u ∈ [0, 1], a round costs β · I(s) · u² + (1 − β) · (1 − u)², where the *instability intensity*
  I(s) ∈ [0, 1] tells how much the node and its neighbourhood are changing (see `TradeOff` and `InstabilityIntensity`).
- **Metrics.** Rounds per episode (energy proxy), error integrated over time against the centrally computed ground
  truth, time spent with an unstable output, and reward.

| Policy | Learner | Simulation file | Gradle task |
| --- | --- | --- | --- |
| Fixed periods (1–5 s) | `SameActionAgent` | `simulationBaseline.yml` | `runSimulationBaseline` |
| Tabular Q-learning | `QLearning` | `simulation.yml` | `runSimulation` |
| Discrete BCQ (offline) | `OfflineLearner` | `simulationDeepOffline.yml` | `runSimulationDeepOffline` |
| Continuous actor-critic (offline) | `OfflineLearnerContinuous` | `simulationDeepOfflineContinous.yml` | `runSimulationDeepOfflineContinous` |
| EdgeGAT graph policy (offline) | `OfflineLearnerGNN` | `simulationDeepOfflineGNN.yml` | `runSimulationDeepOfflineGNN` |
| All of the above + heuristic, on unseen programs | `algoId` 0–9 | `simulationGeneralization.yml` | `runSimulationGeneralization` |

The offline learners are β-conditioned: every logged trajectory is relabelled for each value of β, so a single
policy serves every trade-off. In the generalization experiment the policies trained on the gradient are evaluated,
without retraining, on the other building blocks.

## Running the simulations

Requirements: a JDK ≥ 17 to run Gradle; the JDK used by the simulations is downloaded automatically, and so is
PyTorch (by DJL, on first use).

```bash
./gradlew runSimulationDeepOfflineGNN   # one experiment
./gradlew runAll                        # every experiment (hours)
```

The simulations run on Alchemist 45 with the ScaFi incarnation (Scala 3). `gradle.properties` controls the runs:
with `batch=true` the simulations are headless and sweep the variables listed
in `variables` (seeds `random` 1–4, weights β, fixed `actions`); with `batch=false` a single run opens the graphical
interface with the effects in `effects/`. The exports are written to `build/exports/` (`*/episode` once per episode,
`*/full` every 2 s; `embeddings/` and `spatial/` hold the inspections of the run with seed 2).

**Message timing.** Every simulation file sends the exports with the `SendPreviousExport` action instead of the
incarnation's `- program: send`: when a node wakes up it sends its neighbours the export of its previous round, then
computes the new one, so an export reaches the neighbours one period after it was computed. This is how the
experiments behaved on Alchemist 25, whose `send` fired at the next wake-up of the node. Alchemist 45 sends an export
as soon as its round completes: the gradient re-converges 3–4 times faster, and the advantage of adapting the period
shrinks (with the paper's simulations, the learned gains over the fixed-period frontier fall from about 40% to about
10% for every method, tabular Q-learning included). With `SendPreviousExport` the fixed periods reproduce the
Alchemist 25 exports bit for bit (rounds, unstable time and error of every episode).

## Charts, tables and statistics

`plotter.py` turns the exports into the paper figures (vector PDF + PNG), LaTeX `booktabs` tables with CSV copies,
and a generated `summary.md` with the key numbers:

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python plotter.py --exports build/exports --out charts     # add --strict to fail on any warning
```

The code lives in the `analysis` package: `io` (Alchemist CSV parser), `data`/`dataset` (tidy tables), `stats` and
`pareto` (confidence intervals, significance tests, frontier comparisons), `results`, `style` (IEEE paper style and
the per-method colours and markers), `figures/*` (one module per figure family), `tables` and `report`.

## Development

```bash
./gradlew spotlessApply test        # format and run the Scala tests
.venv/bin/python -m pytest          # tests of the analysis
```

The Scala tests include golden masters that run every simulation for a few episodes and compare the state of the
nodes with pinned values, so refactorings can be proven not to change the experiments. `CLAUDE.md` describes the
architecture, the style and the workflow of the code base. The CI runs the formatting check, the Scala tests and the
analysis tests on every push.

### Attention on other programs

`gnn_program_attention_filmstrip` compares hop-count, broadcast, collect and leader election with the gradient:
node colours show embedding PC1 and arrows point to the most-attended neighbour.
A single PCA basis and colour scale are fitted over all five programs, so the colours are comparable.
`gnn_program_response_filmstrips` also saves one figure per program with output, round rate,
absolute error, and embedding/attention. Output and error scales are specific to each program.
These are descriptions of the policy's internal state, not causal attributions.

Snapshots come from seed 2, episode 16 (the last evaluation episode at beta = 0.95),
with the same trained-on-gradient policy and deployment protocol for every program.
They are stored separately in `build/exports/spatial/generalization/<program>/`.
To collect them and generate the figures:

```bash
./gradlew runSimulationGeneralization -Pvariables=progId -Pbatch=true
.venv/bin/python plotter.py --only gnn_program_attention_filmstrip gnn_program_response_filmstrips
```

The simulation command uses the default `algoId=9` (GNN) and `random=2`.
The snapshots' legacy `gradient` column contains the active program's output; its ground truth
and error are evaluated for that program.

### Leader election and transfer

Program `progId=4` / `leaderElection` implements bounded priority claims with grain **400 m**.
Every device is eligible: a candidate with the `source` flag has higher priority, then lower ids win;
equal candidate priorities prefer the shorter path. At 100 s of each evaluation episode, an actually elected
leader is physically removed (boosted candidate first, then lowest id), rather than moving the source flag.
The node stays absent until the next episode's fresh deployment. Training still uses source moves on the gradient.
Each round relays the strongest claim that remains within the grain.
On the inspected deployment, the reference has two leaders before failure and three afterwards.
The program composes **S followed by G**: a gradient from the elected leaders broadcasts their ids,
so every node outputs the id of its nearest elected leader (lower id breaks equal-distance ties).
The actual elected-leader flag is stored separately as `electedLeader` and exported as `isLeader`;
leader removal and plot markers use that flag, independently of the broadcast output.
Ids are categorical: error is 0 for the correct leader id and 1 otherwise, regardless of numeric separation.
The central reference first resolves the S candidates in priority order on the unclaimed graph,
then computes nearest-leader ids on the full surviving topology. It is recomputed after node removal.
The removed node's completed rounds, reward, error and unstable time are retained in the episode metrics;
mean cumulative metrics keep the initial population denominator (100), with no further contribution after failure.

All learned methods still train on the Euclidean gradient for ten episodes, then transfer to the
five target programs for ten episodes, with the same beta sweep and fixed-period baselines.
As for the other transfer results, the reported runs use seed 2; episode ranges are not across-seed
confidence intervals. The scheduling policy observes the broadcast ids, making changes of leader affiliation
visible to followers. Internal claim and gradient changes can still precede observable output changes.

Run only the new program for all ten algorithm configurations, then regenerate every output:

```bash
./gradlew runSimulationGeneralization -PtargetProgram=4 -Pvariables=algoId -Pbatch=true
.venv/bin/python plotter.py --strict
```

The attention filmstrip includes S using the same PCA basis as the other programs.
Stars mark the source (the boosted candidate for S); orange outlined squares mark currently elected
leaders in S's attention panels. Crosses mark the removed node at its original position.
Arrows retain their original meaning: the most-attended neighbour.
S's response filmstrip extends to 198 s (the attention recording ends at 199 s) to show the slower
re-election; the comparison filmstrip keeps the common instants of the other programs.
In the recorded GNN run (seed 2, beta 0.95, episode 16), every surviving node has the correct leader id
from 185 s through the final 199 s snapshot. Output colours identify nominal leader ids; the error row shows
wrong-id indicators, rather than distances between ids.

## Mobile nodes and error tolerance

`simulationMobility.yml` adds a separate experiment on the Euclidean gradient. Nodes perform reflected Brownian
motion in the 500 m square, independently of their computation rounds. Each coordinate has increment variance
`2 * diffusion * movementInterval`; `diffusion` is in m²/s and the default movement interval is 0.25 s.
The root random generator is injected by Alchemist. Separate streams for movement, source selection,
redeployment and learning are derived from it. Node ordinals, rather than absolute Alchemist ids, identify the
trajectories, so the same seed produces the same movement and source events across scheduling policies.

The mobile objective tolerates absolute errors up to `errorTolerance` metres. Above that threshold, its error
rate is `min(1, max(0, absoluteError - errorTolerance) / 500)`. Each decision receives the negative of
`beta * integratedErrorRate + (1 - beta) * numberOfRounds`, with one second and one round as reference units.
`beta=1` deliberately ignores energy in this experiment. The evaluator integrates the held output against the
current shortest paths after every simulator event, including sleep and terminal partial intervals. Before the
first output, the error penalty is maximal. Reachability is reported separately: two infinite gradients agree;
a finite/infinite mismatch costs the full error scale. `absoluteError` uses 500 m for unavailable outputs and
finite/infinite mismatches; `normalizedError` is additionally capped at one.

Mobile MLP and GNN policies observe local output history, elapsed time, distance changes, neighbour churn,
message age, beta and requested tolerance. Ground truth and global source events are never policy inputs.
The first ten episodes train on diffusion values 0, 0.5 and 2 m²/s, cycling by episode, and tolerances 0, 5 and
15 m. Trajectories are relabelled for those training tolerances and all the existing beta values. The evaluation
tolerance is not added to the training set: 30 m is an unseen conditioning value. Learning and exploration stop
after training; subsequent episodes use fresh deployment/movement seeds and the requested evaluation diffusion
and tolerance. Replay is bounded to 25,000 transitions per regime for the mobile learners.

| `algoId` | Policy |
| --- | --- |
| 0 | Mobile GNN (default) |
| 1 | Mobile continuous MLP |
| 2 | Existing adaptive heuristic |
| 3–7 | Fixed periods 1–5 s |
| 8–9 | Existing MLP/GNN objective and encoders, trained on static nodes, tested on moving nodes |
| 10 | Fixed period 0.5 s |

The static-trained controls retrain with the existing objective in the new harness; they do not load previously
saved weights. They provide a transfer comparison under matching deployment and source-selection streams.
Training is on static nodes for these two controls, while their evaluation trajectories match the mobile methods.

```bash
# One full GNN run, default seed/beta, D=1 m²/s and tolerance=5 m:
./gradlew runSimulationMobility -Pvariables= -Pbatch=true
# Compare all methods at a chosen tolerance and diffusivity:
./gradlew runSimulationMobility -Pvariables=algoId -PmobilityTolerance=15 -PmobilityDiffusion=4 -Pbatch=true
# No source changes, to isolate tracking under mobility:
./gradlew runSimulationMobility -Pvariables=algoId -PmobilitySources=0 -Pbatch=true
# Quick training/evaluation smoke run (not a scientific result):
./gradlew runSimulationMobility -Pvariables= -PmobilityTrainingEpisodes=2 -PmobilityEpisodes=3 \
  -PmobilityEpisodeLength=20 -PmobilitySteps=2 -Pbatch=true
.venv/bin/python -m analysis.mobility --exports build/exports/mobility --out charts/archive/mobility
```

Other Gradle overrides are `mobilityAlgorithm` (0–10) and `mobilityInterval` (seconds).
`-Pvariables=random,algoId,diffusion,errorTolerance,sourceChanges,weight` sweeps the Cartesian product of the YAML
ranges; select only the dimensions needed for a comparison. Set `diffusion=0` through the override for a static
control; combine it with `mobilitySources=0` for a steady environment. Episode length and the midpoint source event
must be multiples of the movement interval. Check sensitivity to the movement interval before drawing conclusions.

Exports live exclusively in `build/exports/mobility`: per-episode metrics, cumulative time-series metrics sampled
every 2 s, and positions/source flags sampled at the same times. Filenames encode the run parameters. Recovery is
the time after the midpoint source change until at least 95% of nodes remain within tolerance for 5 s, measured at
the end of that dwell window. Unrecovered episodes have `recoverySeconds=NaN` and `recoveryCensored=1`; runs without
source changes have both fields `NaN`. Recovery fraction/dwell and training diffusion/tolerance lists are configurable
in `MobilityConfig`. The analysis excludes training, averages episodes within each seed, and uses independent seeds
for confidence intervals; one seed produces no confidence interval. Successful recoveries and censored observations
are exported together in `recovery.csv`. Report tracking, recovery, reachability and rounds together: rounds remain
an energy proxy, excluding locomotion and hardware-specific compute/communication costs.

### Mobile Pareto fronts over beta

The mobile beta campaign compares MLP, GNN, the heuristic and the two static-trained controls at
`beta = 0.5, 0.8, 0.95, 0.99, 1.0`, using seeds 1–4. Fixed periods 0.5–5 s are evaluated once per seed:
their schedules and error metrics are beta-independent, so they form one shared reference curve.
The campaign exports are in `build/exports/mobility-beta-campaign`; regenerate the paper-style figures with:

```bash
.venv/bin/python -m analysis.mobility --exports build/exports/mobility-beta-campaign \
  --out charts/archive/mobility-beta --pareto
```

`--pareto` validates the complete five-beta/four-seed grid before writing output. It produces Pareto plots for
time outside tolerance and integrated normalized error, an additional dead-zone objective view, metrics versus beta,
and gains over the fixed-period
reference. Units match the other figures: total rounds per 200 s episode and integrated error per node in seconds.
Intervals are computed over seed means, after averaging the ten evaluation episodes within each seed.
The gain tables compare each adaptive policy with the fixed-period curve of the same seed and report t intervals
of those paired gains. Values outside the measured fixed-period range remain undefined; no extrapolation is used.
`mobility_pareto_points.csv` includes empirical non-dominated flags, and `mobility_empirical_frontier.csv`
contains the corresponding subset of mean points. Lines connect increasing beta, without assuming that the
learned policies respond monotonically to beta. The raw error metric applies no tolerance to its measurement;
the additional objective panels apply the configured dead zone. Recovery censoring is retained in the CSV exports.

The mobile Pareto figures include a mean-curve overview and an expanded view of the two mobile policies,
with all five beta values labelled. Confidence intervals are shown in the beta plots, where mobile policies
and transfer/heuristic controls have separate rows and explicitly separate y ranges. The gain heatmap shows
mean percentages; hatching marks intervals crossing zero. `mobility_frontier_gains_intervals.pdf` provides
all paired 95% intervals in aligned forest plots. These presentation changes preserve every numerical CSV export.

## Independent static / zero-shot campaign

The versioned runner is `experiments/campaign.py` (no dependency on scripts in the ignored `build/` folders). The
previous figures are in a timestamped folder of `charts/archive/`, with checksums, the working diff and a source
snapshot; the raw exports were not moved. The new figures go to `charts/static`, `dynamic`, `generalization`,
`beta-study`, `attention`, `filmstrip` and `learning`. The paper's own figures (the original static experiment of
`plotter.py`, which the archive also holds) are regenerated, byte for byte, with
`.venv/bin/python plotter.py --exports build/exports --out charts/paper`.

**Protocol.** Every episode starts from a fresh, still network (positions, program state and episodic memory are
reset; only the weights and the replay buffer survive between training episodes). Training uses the gradient on 100
nodes: two static episodes with a source switch at 100 s for every episode in which half of the nodes move between
100 and 150 s (1 and 2 m/s alternately). After training the policy is frozen. Evaluation topologies come from
evaluation seeds that are independent of the training seed and of the training length; populations of 50 and 200
nodes use arenas scaled to the same nominal density. The actor still chooses one period in [0.5, 5] s at its own
round (no reactive wake-ups); ground truths, event identities and replay strata never reach its observations.
Messages follow the Alchemist 25 timing (`SendPreviousExport`, see *Message timing*), so the neighbours' data must
outlast two of the longest periods: the retention is 15 s, as in the paper. The campaigns in `validation*` and `test`
(before 2026-11-04) ran with the immediate send of Alchemist 45 and a 6 s retention; `test-previous-export` repeats the
test plan, identical but for these two settings (the variants were selected on the earlier validation).

**Grid.** Validation (selection, β = 0.95): training seeds 11–14 × evaluation seeds 101–102 (ten episodes each) on
the static gradient, the four other programs at 100 nodes, and motion at 1 m/s; eight variants × GNN/MLP:
`current` (mobile objective, full mobile features), `legacy` (the original static learners and objective),
`old-encoding`, `transferable` (without the gradient-specific `dependants` and `unexplainedChange` features), then
cumulatively `balanced` (replay balanced per training scenario), `log-exploration` (log-uniform exploration of the
same period range), `long-training` (30 episodes) and `matched-updates` (10 episodes with the optimizer updates of
30); a second round (`--name validation-agnostic --without-shared`, merged by the analysis with `--with`) adds
`agnostic`: the transferable features with the base ones that assume the gradient's semantics blanked (Bellman-Ford
residual, children ratio, relative order of the outputs; YAML variable `programAgnostic`). Test: new training seeds 21–26 × evaluation seeds 201–202 on every scenario (also 50/200 nodes and motion at
2 m/s) for `current` and the selected variant, `legacy` where the old experiments are comparable, and the β study
(0.5, 0.8, 0.95, 0.99, 1, separately trained policies) of the selected variant. The fixed periods and the heuristic
do not learn, so their evaluation does not depend on the training seed (checked bit for bit): they run once per
evaluation seed and are paired with every training seed.

```bash
./gradlew spotlessCheck test campaignRuntime   # the runtime task freezes the JDK and classpath for the runner
.venv/bin/python -m pytest
.venv/bin/python -m experiments.campaign run --smoke --variants transferable --workers 10
# 852 runs, about 2 h; then the second round (96 runs) and the merged analysis, which writes selection.json
.venv/bin/python -m experiments.campaign run --split validation --workers 10 --variants current legacy \
  old-encoding transferable balanced log-exploration long-training matched-updates
.venv/bin/python -m experiments.campaign run --split validation --variants agnostic --name validation-agnostic \
  --without-shared --workers 10
.venv/bin/python -m analysis.campaign build/learning-campaign/validation --with build/learning-campaign/validation-agnostic \
  --out charts
.venv/bin/python -m experiments.campaign run --split test --workers 14         # locked to the selection
# The same plan with the Alchemist 25 timing; --first static runs the static family (and its beta study) first
.venv/bin/python -m experiments.campaign run --split test --name test-previous-export --first static --workers 14
# Beta study of the original learners in the same protocol (their controls are those of test-previous-export)
.venv/bin/python -m experiments.campaign run --split test --name test-previous-export-legacy-beta \
  --selection build/learning-campaign/legacy-finals.json --beta-study-only --without-shared \
  --families static generalization --first static --workers 14
.venv/bin/python -m analysis.campaign build/learning-campaign/test --out charts
```

`prepare` (or `run`) copies the compiled classes, the YAML and the sources into the campaign folder, so later code
changes cannot alter a resumed campaign; completed runs are reused only after checking their terminal rows and
training flags. A run uses about 2 GB of memory and a bit more than one core.

**Analysis.** It rejects incomplete campaigns and averages the evaluation episodes of each (training seed,
evaluation seed) pair before any inference; intervals resample training and evaluation seeds independently while
keeping the pairing between methods. The gain is the error reduction w.r.t. the fixed-period frontier at the same
rounds (undefined outside the measured range: no extrapolation). Selection: the variant with the largest paired
improvement over `current` of the mean of the static gain and the zero-shot gain on the other programs at 100 nodes,
provided the lower 95% bound is positive; otherwise `current` is kept. It is computed for GNN and MLP together
(`variant`) and for each architecture (`byArchitecture`); the test uses the latter, since the two architectures are
separate methods (a choice made after the validation showed opposite effects on them), and sweeps β for the final
variant of each architecture. Test
confirmation is written to `confirmation.json`, never over the validation selection: the same rule applied to the
test seeds (`testSelection`) and, per architecture, whether it selects the validation variant again (`confirmed`);
the variants are never re-selected on test data. `python -m analysis.campaign_report <campaign> --out <file.md>`
turns `summary.csv`, `variability.csv` and the decision into markdown tables. `summary.csv` holds every
estimate with its interval and `variability.csv` the descriptive spread across episodes, topologies and trained
policies (not an additive decomposition). The filmstrips and attention plots come from one run chosen in advance
(GNN of the selected variant, first test training seed, evaluation seed 202, β = 0.95);
`attention/period_saturation.csv` gives the share of decisions at the bounds of the action space.
`analysis.campaign_paper` (also called by the test analysis) draws the paper's figures from the campaign with the
paper's code and look: `learning/learning_curves_reward` and `learning_curves_metrics` (training and evaluation
episodes; dotted lines mark the training episodes with motion; the reward of a fixed period at any β is
−β · normalized error − (1 − β) · rounds per node per second, as exported), `static/pareto_frontiers` and
`generalization/pareto_frontiers_programs` (β sweeps against the 0.5–5 s frontier, the arrow gives the GNN's change at
β = 0.95), `beta-study/beta_effect_intra_episode` (and `_motion`; the period is 1 / rounds per node per second over
10 s, since nodes wake nearly in step), `filmstrip/gnn_attention_filmstrip`, `gnn_program_attention_filmstrip`,
`gnn_response_filmstrip_*` and `attention/gnn_latent_dynamics` (the inspected runs laid out as the paper's exports),
`filmstrip/spatial_filmstrip`, `spatial_wave_animation.gif` (MLP, as the paper's TD3; `_gnn` for the GNN) and
`static/spatial_computation`, `static/source_switch_dynamics` (the 6 training seeds on topology 202),
`static/tradeoff_vs_beta`, `summary_beta095` and `frontier_gains_heatmap` (the paper's functions on the campaign laid
out as the paper's evaluation table: recovery time after the event in place of the unstable time, mean absolute
error in place of the integrated error; fixed periods per evaluation topology), and
`generalization/generalization_overview`. The policy-explanation figures need the actor inspection, which the mobile
experiment does not export.

**Transfer diagnosis.** `current` with one setting of the original experiment restored at a time (GNN, validation
seeds, static and zero-shot programs): `error-cost` (training cost on the error instead of the gradient's staleness),
`wide-loss-scale` (500 m), `long-retention` (15 s) and `paper-settings` (all three). Cost and scale shape only the
training costs, so they reuse the validation controls; the retention changes the network, so its campaign runs its own
controls (`--set` applies a setting to every run, the controls included).

```bash
.venv/bin/python -m experiments.campaign run --split validation --variants error-cost wide-loss-scale \
  --name validation-diagnosis --without-shared --algorithms 0 --families static generalization --workers 14
.venv/bin/python -m experiments.campaign run --split validation --variants long-retention paper-settings \
  --name validation-retention --algorithms 0 --families static generalization --set retention=15 --workers 14
.venv/bin/python -m analysis.campaign_diagnosis build/learning-campaign/validation+build/learning-campaign/validation-diagnosis \
  build/learning-campaign/validation-retention --out charts/generalization
```

**Simulation cost.** The mobile experiment observes every node at every simulator event. Reachability and source ids
are cached until the topology or the sources change, the topology is invalidated only when a node actually moves,
the per-tolerance integrals use arrays, molecules are looked up once with shared instances, and an event at an
instant already observed re-observes only the nodes whose output changed (and their neighbours for the staleness
cost). All of these preserve the exported results bit for bit (verified on static, dynamic, leader-removal and
snapshot runs) and make a run 2.4× (learned, 100 nodes) to 5.7× (0.5 s fixed period) faster.
