# CLAUDE.md

Guidance for working on this repository: experiments for the TETCI paper on
**learning when to execute the rounds of aggregate-computing programs** (ScaFi on Alchemist)
with tabular, offline-deep and graph-based reinforcement learning.

## Build, run, test

- Gradle runs on any JDK ≥ 17 (the system JDK 25 is fine); compilation, tests and simulations use a JDK 17
  toolchain, downloaded automatically through the Foojay resolver when missing.
- Compile: `./gradlew compileScala`
- Tests: `./gradlew test` (ScalaTest, specs live in `src/test/scala`, mirroring the main packages)
- Format: `./gradlew spotlessApply` (scalafmt, configuration in `.scalafmt.conf`); `spotlessCheck` must pass.
- Stack: Scala 3.10, Alchemist 45 with the ScaFi 1.8 incarnation, DJL 0.39 (PyTorch 2.7). The Scala snippets of the
  YAML files are evaluated by the Scala 3 REPL shipped with Alchemist.
- One Gradle task per simulation file: `src/main/yaml/simulationDeepOffline.yml` → `./gradlew runSimulationDeepOffline`.
  `gradle.properties` sets `batch=true` (headless, a `DefaultLauncher` sweeping the listed variables) or `false`
  (Swing GUI with the effects in `effects/`); exports land in `build/exports/`.
- Charts, tables and statistics: `plotter.py` over the `analysis` package (Python, only for the paper outputs).
  The system Python has no matplotlib, use `uv`:
  `uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt`, then
  `.venv/bin/python plotter.py --exports build/exports --out charts --strict` and `.venv/bin/python -m pytest`.
  Figures follow an IEEE two-column style with one colour/marker per method (`analysis/style.py`).

## Architecture

| Package | Role |
| --- | --- |
| `it.unibo.scafi` | `Main`, the aggregate program of every node (runs the active `BuildingBlock`, observes its neighbourhood, asks the policy for the next wake-up, logs rewards, errors and trajectories); `BuildingBlock`, `Topology`, `GroundTruth(s)` and `EvaluationProtocol` (training on the gradient, then evaluation on the target block and on each β). |
| `it.unibo.alchemist.model.reactions` | `CentralAgent`, the global reaction that runs the episode loop (online/offline training, `Redeployment` of the nodes, ground truths); `FinalEpisodeSnapshots` (spatial and attention exports of the inspected run); `SingleBlinker` (moves the source). |
| `it.unibo.alchemist.model.timedistributions.NextWakeUp` | Time distribution that reads `nextWakeUp` to schedule the node's next round. |
| `it.unibo.learning` | `TradeOff` (the convergence/energy objective, β feature and rewards), `Box`, `LearningInfo`. |
| `it.unibo.learning.abstractions` | `AgentState` (windows of neighbourhood snapshots), `Step`/`Trajectory`, `ReplayBuffer`, `DecayReference`. |
| `it.unibo.learning.encoding` | State features: `NeighbourhoodSignals`, `InstabilityIntensity`, `OwnTrends`, and the `SpatialTemporal` (MLP) and `StarGraph` (GNN) encoders. |
| `it.unibo.learning.agents` | `Learner` implementations: `QLearning`, `OfflineLearner` (discrete BCQ), `OfflineLearnerContinuous` (TD3), `OfflineLearnerGNN` (TD3 with an EdgeGAT actor and multi-objective critics), `HeuristicAdaptiveAgent`, `SameActionAgent` (fixed-period baselines). |
| `it.unibo.learning.agents.offline` | What the offline learners share: `EpisodicOfflineLearner` (end-of-episode training loop), `TrainingSchedule`, `Relabelling` (hindsight relabelling for every β), `StratifiedTransitions`, `LearnerSeeds`, `TargetNetworks`, `TwinDelayed` (the TD3 policy delay and target smoothing), `BatchTensors`, and the post-training `ActorInspection` (`Regime`, `PolicySurface`). |
| `it.unibo.learning.network` | DJL/PyTorch: `TorchNetwork`, `EdgeGATBlock`, `FastInference` (lock-free forward passes used by the nodes every round), `MinMaxActionScaler`, `NDArrayOps`. |
| `it.unibo.inspection` | Testable recorders of the snapshot CSV files (`NodeView`, `AttentionSummary`, ...). |
| `it.unibo.alchemist.model.actions` | `SendPreviousExport`: the message timing of Alchemist 25 (at wake-up a node sends the export of its previous round, then computes). Every YAML uses it before `RunScafiProgram` instead of `- program: send`, whose Alchemist 45 timing (send as soon as the round completes) changes the results. |
| `it.unibo.alchemist` | `Molecules` (typed molecule access), `KotlinInterop`, and the custom extractors used by the YAML `export` sections (`boundary.extractors`). The Alchemist glue must stay under `it.unibo.alchemist`: the loader resolves the unqualified `type:` names of the YAML files only there. |

**The YAML files are a public API.** They instantiate learners and value objects through embedded Scala snippets
(e.g. `new OfflineLearnerGNN(0.5, 5.0, random.as[Double], weight.as[Double])`) and refer to classes, molecules and
constructor parameters by name. These snippets are compiled only at simulation start-up, so the compiler will not
catch a broken signature: when changing a constructor, class name, package or molecule name, grep `src/main/yaml`
and update it in the same commit.

## How to change code here

- **Small, incremental, behaviour-preserving steps.** One concern per commit; the build and the tests are green after
  every commit.
- **TDD.** For new behaviour: failing test → minimal code → refactor. For refactoring untested code: first write a
  *characterization* test that pins the current behaviour (it must pass on the old code), then refactor while it
  stays green.
- **Reproducibility is behaviour.** Experiments are seeded: a refactoring must not change the sequence of calls on a
  `Random`/`RandomGenerator`, the order of floating-point operations in hot numerical code (fast MLP/GAT inference,
  rewards, encoders), nor the names/order of exported CSV columns that `plotter.py` reads. If a change intentionally
  alters results, say so in the commit message.
- Commit messages follow Conventional Commits (`refactor(agents): ...`, `test(network): ...`, `fix: ...`).
  **No AI attribution lines** (`Co-Authored-By`, "Generated with ...").

## Scala style (Scala 3)

### Scala 3 syntax and formatting
- Formatting is owned by scalafmt (`.scalafmt.conf`, applied by `./gradlew spotlessApply`, checked by
  `spotlessCheck` in CI): don't hand-format.
- Significant indentation: templates and bodies open with `:` or `=` instead of braces (`class X:`, `object Y:`,
  `def f(): Unit =`), control structures use `if … then … else`, `while … do`, `for … do`/`yield`, `match` with
  indented cases. Definitions longer than ~25 lines are closed by an end marker (`end CentralAgent`); scalafmt adds
  and removes them. Multi-line lambdas passed as arguments keep their braces (`xs.foreach { x => … }`).
- `*` in imports (`import scala.jdk.CollectionConverters.*`), `?` in wildcard types (`CentralAgent[?, ?]`),
  `using` to pass an implicit argument explicitly (`PriorityQueue.empty(using ordering)`).
- `extension` methods instead of implicit classes; `given`/`using` instead of implicit values and parameters;
  top-level definitions (types, extensions) instead of `package object`s.
- `enum` for fixed sets of alternatives, with parameters and methods when the cases share behaviour
  (`BuildingBlock`, `Regime`); `values` lists the cases.
- Instantiate the classes of this project (and case classes) without `new` (`ReplayBuffer(capacity, random)`); keep
  `new` for Java and library classes and for anonymous classes.

### Values and state
- `val` and immutable collections by default. `var` is acceptable only for the inherently stateful Alchemist glue
  (reactions, learners that accumulate data); keep it `private` and as narrow as possible.
- Prefer expressions to statements: no `return`, no "init then mutate" — use `if`/`match`/`fold` as expressions.
- Model fixed sets of alternatives as `enum`s (or `sealed trait`s when the cases carry different data), not as
  `String` or `Int` codes. Parse external values (YAML, molecules) once, at the boundary, into the ADT.
- Use a `final case class` instead of a tuple as soon as it has more than two fields or crosses a module boundary.
- **No magic numbers.** Every literal with a domain meaning (thresholds, gains, sizes, times, seeds, file names,
  column names) is a named constant: `val UpperCamelCase` inside an `object` (the companion or a dedicated
  `object ...Constants`), with a one-line doc comment saying what it means. Only neutral literals are allowed inline
  (`0`, `1`, `0.0`, `1.0`, `2` in `x * x`-like arithmetic, empty collections). If a value is derived from others
  (e.g. the start of the last episode = `(episodes - 1) * episodeLength`), compute it instead of hard-coding it.

### Absence and failure
- No `null` (wrap Java values with `Option(...)`), no `Option.get`, no `head` on possibly empty collections:
  use `fold`, `getOrElse`, `collectFirst`, `headOption`, `maxByOption`, `minOption`.
- Do not swallow errors: never `catch { case _: Throwable => }`. Use `scala.util.Try`/`Either`, or
  `case NonFatal(e)` with a meaningful fallback.
- `asInstanceOf` only at the Alchemist/Java boundary (molecule values, YAML scripts), wrapped in a single helper.
- Resources (files, DJL managers) are handled with `scala.util.Using`.

### Functions and collections
- **Short methods.** A method does one thing and fits in a screen: aim for ≤ 15 lines of body, never more than ~25.
  When a method grows, extract well-named sub-methods (private, or local `def`s when they are tiny) until the
  top-level method reads as a sequence of steps. Deep nesting (`if` inside `foreach` inside `match`) is a signal to
  extract.
- **Short files, one domain concept each.** A file that grows past ~200 lines, or that mixes concerns (e.g. training,
  ground truth, CSV export and episode management in one reaction), is split into types named after the domain
  concepts they implement (`GroundTruth`, `EpisodeManager`, `SpatialSnapshotExporter`, `StratifiedTransitions`, ...),
  each in its own file, with the original class delegating to them.
- Small pure functions, placed in the companion `object` or in a dedicated `object`; keep side effects
  (Alchemist node access, file I/O, logging) at the edges so that the logic is testable without a simulator.
- Use combinators (`map`, `collect`, `foldLeft`, `sliding`, `zip`, `groupMapReduce`, `sum`, `maxBy`) instead of
  index loops and accumulators. `while` loops are allowed only in measured hot paths (neural-network inference
  executed every round); encapsulate them in one small, documented function.
- Don't repeat yourself across learners: shared behaviour goes in a trait or in a helper object, not copy-paste.
- Prefer `Iterator`/`LazyList` over manual buffers when building sequences on demand.

### Types and API
- Explicit return types on every public member.
- `final` for classes not meant to be extended; `private` for everything that is not API.
- Named and default arguments for constructors used from YAML; keep positional order stable.
- Traits provide sensible default implementations instead of forcing every implementor to write no-op overrides.

### Naming and layout
- **No compact or abbreviated names**: `groundTruth`, not `gtFin`; `ticks`, not `tks`; `maybeParent`/`parent`, not
  `opt`/`parentOpt`; `previous`/`current`, not `prev`/`curr`; `manager`, not `m`; `neighbour`, not `n`/`nId`.
  Single-letter names only for conventional math in a short expression (`u` for the normalized period of a formula
  that documents it, `i`/`k` for indices in a hot loop) and `_` placeholders in one-line lambdas.
- `camelCase` for terms, `UpperCamelCase` for types and constants.
- Imports at the top of the file; no fully qualified names inline (`new java.io.File`, `scala.collection.mutable.X`).
  Use `scala.jdk.CollectionConverters.*` for Java interop.

### Tests
- `SimulationGoldenMasterSpec` runs every YAML for a few episodes (no exporters) and compares fingerprints of the
  node states with pinned values: it is the safety net for refactoring Main, CentralAgent and the learners.
  The pinned values depend on the PyTorch CPU kernels, so they may need re-pinning on a different machine.
  The tests run PyTorch single-threaded (see `build.gradle.kts`): multi-threaded BLAS is not reproducible.
- `SimulationFilesSpec` loads every YAML (and every algorithm of the generalization scenario), so that the embedded
  Scala snippets are compiled by the build.
- Code that only runs late in an experiment (offline training after episode 2, inspection after episode 9, snapshots
  at episode 19) is not covered by the golden masters: verify it by running the full simulation before and after the
  change in two scratch git worktrees and comparing the outputs (never write into the real `build/exports`). Full
  GNN runs need a larger test heap (`maxHeapSize`).
- ScalaTest `AnyFlatSpec with Matchers`; one spec per production class, same package, named `<Class>Spec`.
- Build `AgentState` fixtures with the helpers in the test sources rather than ad-hoc maps in every test.
- Use fixed seeds; when pinning numerical behaviour, compare with a tolerance only if the code legitimately
  changes the arithmetic, otherwise compare exactly.
