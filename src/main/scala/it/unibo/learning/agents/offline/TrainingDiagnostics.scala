package it.unibo.learning.agents.offline

import ai.djl.ndarray.NDManager

import scala.util.Random

/** What a training session leaves in an offline learner, for one trade-off β and one regime of a fixed sample of its
  * replay: the TD loss of its first critic, the mean period of its actor and the mean period that maximizes the
  * scalarized first critic on a grid of periods. An actor far from the greedy period does not optimize its critic (it
  * needs more training); a lazy greedy period means a lazy critic (the reward or the credit must change).
  */
final case class TrainingDiagnostic(
    episode: Int,
    beta: Double,
    unstable: Boolean,
    samples: Int,
    criticLoss: Double,
    actorPeriod: Double,
    greedyPeriod: Double
):
  /** The fields in the order of [[TrainingDiagnostic.Header]]. */
  def row: Seq[String] =
    Seq(episode, beta, unstable, samples, criticLoss, actorPeriod, greedyPeriod).map(_.toString)

object TrainingDiagnostic:
  /** The columns of the diagnostics export. */
  val Header: Seq[String] = Seq("episode", "beta", "unstable", "samples", "criticLoss", "actorPeriod", "greedyPeriod")

/** What the networks of a TD3 learner answer about a batch of its transitions: the TD loss of the first critic (against
  * the clipped double-Q target without smoothing noise), the periods of the actor and, for every period of the grid,
  * the scalarized value of the first critic in each transition.
  */
final case class ProbedBatch(criticLoss: Double, actorPeriods: Seq[Double], values: Seq[Seq[Double]])

/** How the diagnostics query the networks of a learner on its transitions of type `T`. */
trait DiagnosticProbe[T]:
  /** The periods whose values the critic is asked for. */
  def grid: Seq[Double]

  /** The trade-off weight a transition was relabelled for, if it is known. */
  def betaOf(transition: T): Option[Double]

  /** Queries the networks on `batch`, allocating the tensors in `manager`. */
  def probe(batch: Vector[T], manager: NDManager): ProbedBatch

object TrainingDiagnostics:
  /** Transitions of each regime in the fixed sample. */
  val SamplesPerRegime: Int = 512

  /** Periods in the grid on which the critic is maximized. */
  val GridSize: Int = 10

  /** Seed of the fixed sample: a stream of its own, which leaves the learner's randomness untouched. */
  val SampleSeed: Long = 20261110L

  /** `GridSize` periods evenly spaced from `min` to `max`. */
  def grid(min: Double, max: Double): Seq[Double] =
    (0 until GridSize).map(index => min + index * (max - min) / (GridSize - 1))

  /** A fixed sample of the transitions, unstable ones first, keyed by regime. */
  def sample[T](transitions: StratifiedTransitions[T]): Seq[(Boolean, Vector[T])] =
    val random = new Random(SampleSeed)
    Seq(true, false).map(unstable => unstable -> transitions.sampleRegime(SamplesPerRegime, random, unstable))

  /** One diagnostic for every regime of the `sample` and every trade-off weight in it, after the training of `episode`. */
  def of[T](
      episode: Int,
      sample: Seq[(Boolean, Vector[T])],
      probe: DiagnosticProbe[T],
      manager: NDManager
  ): Seq[TrainingDiagnostic] =
    for
      (unstable, transitions) <- sample
      (beta, batch) <- transitions
        .groupBy(probe.betaOf)
        .collect { case (Some(beta), batch) => beta -> batch }
        .toSeq
        .sortBy(_._1)
    yield diagnose(episode, beta, unstable, batch, probe, manager)

  private def diagnose[T](
      episode: Int,
      beta: Double,
      unstable: Boolean,
      batch: Vector[T],
      probe: DiagnosticProbe[T],
      manager: NDManager
  ): TrainingDiagnostic =
    val probed = probe.probe(batch, manager)
    val greedy = batch.indices.map(sample => probe.grid(probe.grid.indices.maxBy(probed.values(_)(sample))))
    TrainingDiagnostic(episode, beta, unstable, batch.size, probed.criticLoss, mean(probed.actorPeriods), mean(greedy))

  private def mean(values: Seq[Double]): Double = if values.isEmpty then Double.NaN else values.sum / values.size
end TrainingDiagnostics
