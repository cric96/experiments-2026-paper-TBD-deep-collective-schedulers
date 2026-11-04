package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.model.reactions.FinalEpisodeSnapshots.*
import it.unibo.alchemist.model.{Environment, Node, Position}
import it.unibo.inspection.*
import it.unibo.learning.LearningInfo
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.AgentState.Snapshot
import it.unibo.learning.agents.{Learner, OfflineLearnerContinuous, OfflineLearnerGNN}
import it.unibo.scafi.{BuildingBlock, EvaluationProtocol, Sensors}

import java.io.File
import scala.collection.immutable.Queue
import scala.jdk.CollectionConverters.*

/** Records the last episode of the inspected run for the spatial figures: the state of every node (continuous learner)
  * or where the graph-attention policy looks (GNN learner).
  */
final class FinalEpisodeSnapshots[T, P <: Position[P]](
    environment: Environment[T, P],
    learner: Learner,
    learningInfo: LearningInfo,
    groundTruth: Int => Double,
    directory: File = DefaultDirectory,
    episodeOverride: Option[Int] = None,
    inspectEverySeed: Boolean = false,
    observationTolerance: () => Double = () => 0.0
):
  private def targetBlock: Option[BuildingBlock] =
    nodes.headOption.flatMap(_.valueOf[Any](Sensors.program)).map(BuildingBlock.fromMolecule)

  private def recordedEpisode: Int =
    episodeOverride.getOrElse(if targetBlock.nonEmpty then GeneralizationEpisode else RecordedEpisode)

  private def episodeStart: Double = (recordedEpisode * learningInfo.episodeSize).toDouble

  private def recordingDirectory: File = targetBlock.fold(directory) { block =>
    new File(directory, s"generalization/${block.name}")
  }
  private lazy val spatialRecorder = SpatialSnapshotRecorder(recordingDirectory, weightOr(NoWeight))
  private lazy val attentionRecorder = AttentionSnapshotRecorder(recordingDirectory)

  /** Called by the central agent at every step. */
  def observe(time: Double): Unit =
    if inspectEverySeed || learningInfo.randomSeed == InspectedRun then
      learner match
        case _: OfflineLearnerContinuous => observeNodes(time - episodeStart)
        case graphLearner: OfflineLearnerGNN => observeAttention(graphLearner, time - episodeStart)
        case _ => ()

  private def observeNodes(timeInEpisode: Double): Unit =
    if SpatialWindow.contains(timeInEpisode) && SpatialWeights.exists(isWeight(weightOr(NoWeight), _)) then
      spatialRecorder.record(timeInEpisode, nodes.map(viewOf), links)
      if timeInEpisode >= SpatialWindow.flushFrom then spatialRecorder.flush()

  private def observeAttention(graphLearner: OfflineLearnerGNN, timeInEpisode: Double): Unit =
    val weight = weightOr(AttentionWeight)
    val window = if episodeOverride.nonEmpty || targetBlock.contains(BuildingBlock.LeaderElection) then
      ElectionAttentionWindow
    else AttentionWindow
    if isWeight(weight, AttentionWeight) && window.contains(timeInEpisode) then
      val positions = nodes.map(node => node.getId -> coordinatesOf(node)).toMap
      nodes.foreach { node =>
        stateOf(node).foreach { state =>
          val inspection = graphLearner.inspectEmbeddingAndAttention(state, weight)
          val neighbourPositions = inspection.neighbours.map(positions.get)
          val summary = AttentionSummary.of(positions(node.getId), inspection.attention.toSeq, neighbourPositions)
          attentionRecorder.record(timeInEpisode, viewOf(node), summary, inspection.embedding)
        }
      }
      if timeInEpisode >= window.flushFrom then attentionRecorder.flush()

  private def nodes: List[Node[T]] = environment.getNodes.getCurrent.asScala.toList

  /** The trade-off weight of the run, read from the nodes. */
  private def weightOr(default: Double): Double =
    nodes.headOption.flatMap(_.valueOf[Double](Sensors.weight)).getOrElse(default)

  private def coordinatesOf(node: Node[T]): (Double, Double) =
    val position = environment.getCurrentPosition(node)
    (position.get(0), position.get(1))

  private def viewOf(node: Node[T]): NodeView =
    val (x, y) = coordinatesOf(node)
    NodeView(
      node.getId,
      x,
      y,
      isSource = node.valueOf[Boolean](Sensors.source).getOrElse(false),
      output = node.valueOf[Double](Sensors.localComputation).getOrElse(Double.PositiveInfinity),
      groundTruth = groundTruth(node.getId),
      period = node.valueOf[Double](Sensors.nextWakeUp).getOrElse(UnknownPeriod),
      ticks = node.valueOf[Number](Sensors.ticks).fold(0)(_.intValue()),
      isLeader = node.valueOf[Boolean](Sensors.electedLeader).getOrElse(false),
      categoricalOutput = targetBlock.contains(BuildingBlock.LeaderElection)
    )

  private def links: Seq[Link] = nodes.flatMap { node =>
    environment.getNeighborhood(node).getCurrent.getNeighbors.asScala.map { neighbour =>
      Link(node.getId, neighbour.getId, coordinatesOf(node), coordinatesOf(neighbour))
    }
  }

  /** The state the node's policy observed in its last round, if it has run at least once. */
  private def stateOf(node: Node[T]): Option[AgentState] =
    val outputs = node.valueOf[Queue[Snapshot]](Sensors.localComputationWindow).getOrElse(Queue.empty)
    Option.when(outputs.nonEmpty)(
      AgentState(
        node.getId,
        outputs,
        observationTolerance(),
        outputs.lastOption.flatMap(_.get(node.getId)).fold(0.0)(_.speed)
      )
    )
end FinalEpisodeSnapshots

object FinalEpisodeSnapshots:

  /** Where the snapshots of the experiments are exported. */
  val DefaultDirectory: File = new File("build/exports/spatial")

  /** Seed of the run whose last episode is recorded. */
  val InspectedRun: Int = 2

  /** Index of the recorded episode: the last of the 20 episodes of the experiments. */
  val RecordedEpisode: Int = 19

  /** A time window within the recorded episode; the files are (re)written from `flushFrom` on. */
  final case class Window(from: Double, to: Double, flushFrom: Double):
    def contains(timeInEpisode: Double): Boolean = timeInEpisode >= from && timeInEpisode <= to

  /** Node snapshots: the whole episode, after the time the re-deployed nodes need to restart. */
  val SpatialWindow: Window = Window(from = 2.0, to = 200.5, flushFrom = 199.0)
  val SpatialWeights: Seq[Double] = Seq(0.8, 0.95)

  /** Attention snapshots: around the move of the source, in the middle of the episode. */
  val AttentionWindow: Window = Window(from = 85.0, to = 160.5, flushFrom = 159.5)
  val AttentionWeight: Double = 0.95

  /** Election recovery can outlast a source-change wave; retain attention until the episode ends. */
  val ElectionAttentionWindow: Window = Window(from = 85.0, to = 199.5, flushFrom = 198.5)

  /** Last evaluation episode at beta = 0.95 (zero-based), shared by all target programs. */
  val GeneralizationEpisode: Int = EvaluationProtocol.TrainingEpisodes +
    EvaluationProtocol.EvaluationWeights.indexOf(AttentionWeight) * EvaluationProtocol.EpisodesPerWeight +
    EvaluationProtocol.EpisodesPerWeight - 1

  /** Weight read when the nodes have none, matching none of the recorded weights. */
  val NoWeight: Double = 0.0
  val WeightTolerance: Double = 1e-6

  /** Period exported for the nodes that have not chosen one yet: the longest one. */
  val UnknownPeriod: Double = 5.0

  private def isWeight(weight: Double, expected: Double): Boolean = math.abs(weight - expected) < WeightTolerance
end FinalEpisodeSnapshots
