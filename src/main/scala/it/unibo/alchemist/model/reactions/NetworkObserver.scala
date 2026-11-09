package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.model.reactions.NetworkObserver.*
import it.unibo.alchemist.model.{Node, Position}
import it.unibo.mobility.{CostSignal, MobileEpisode, MobilityConfig, NeighbourhoodError, NodeObservation, Staleness}
import it.unibo.scafi.Sensors

/** Observes the output of every node after each simulator event and records it in the episode ([[MobileEpisode]]).
  *
  * A simulator event changes at most the output of the node that ran, so at an instant that was already observed on the
  * same network only the nodes whose output changed, and the neighbours whose cost it changes (staleness or
  * neighbourhood error), are observed again: the others would be recorded with an unchanged observation, which adds
  * nothing.
  */
final class NetworkObserver[T, P <: Position[P]](network: NetworkView[T, P], config: MobilityConfig):
  private var last: Option[Observation] = None

  /** Forgets the last observation, e.g. when a new episode starts. */
  def forget(): Unit = last = None

  def observe(now: Double, episode: MobileEpisode, nodes: List[Node[T]]): Unit =
    val outputs = nodes.map(node => node.getId -> node.valueOf[Double](Sensors.localComputation)).toMap
    last.filter(_.sees(now, network.version, outputs.keySet)) match
      case Some(previous) => observeChanges(now, episode, previous, outputs)
      case None => observeAll(now, episode, outputs)

  private def observeAll(now: Double, episode: MobileEpisode, outputs: Map[Int, Option[Double]]): Unit =
    val observations = outputs.map((id, output) => id -> observationOf(episode, id, outputs))
    episode.observe(now, observations)
    last = Some(Observation(now, network.version, outputs, observations))

  private def observeChanges(now: Double, episode: MobileEpisode, previous: Observation, outputs: Outputs): Unit =
    val changed = outputs.keySet.filterNot(id => sameOutput(previous.outputs(id), outputs(id)))
    if changed.nonEmpty then
      val affected =
        if chargedSignal(episode) == CostSignal.Error then changed else changed ++ changed.flatMap(neighbours)
      val updates = affected.map(id => id -> observationOf(episode, id, outputs)).toMap
      val observations = previous.observations ++ updates
      episode.observeChanges(now, updates, observations)
      last = Some(Observation(now, network.version, outputs, observations))

  private def observationOf(episode: MobileEpisode, id: Int, outputs: Outputs): NodeObservation =
    val error = errorOf(episode, id, outputs)
    val cost = chargedSignal(episode) match
      case CostSignal.Error => error
      case CostSignal.Staleness => stalenessOf(id, outputs)
      case CostSignal.Neighbourhood =>
        NeighbourhoodError.of(error, neighbours(id).map(errorOf(episode, _, outputs)), config.lossScale)
    NodeObservation(error, network.isReachable(id), cost)

  /** The error of the output held by `id` against the ground truth; unbounded before its first round. */
  private def errorOf(episode: MobileEpisode, id: Int, outputs: Outputs): Double =
    val expected = network.groundTruths.getOrElse(id, Double.PositiveInfinity)
    outputs.get(id).flatten.fold(Double.PositiveInfinity)(config.blockAt(episode.index).absoluteError(_, expected))

  /** What the decisions pay for ([[CostSignal]]): the configured signal in training, the error otherwise. */
  private def chargedSignal(episode: MobileEpisode): CostSignal =
    if config.isTraining(episode.index) then config.costSignal else CostSignal.Error

  private def neighbours(id: Int): Seq[Int] = network.neighboursOf(id).map(_._1)

  /** How far the output held by `id` is from the value a round would compute now ([[Staleness]]). */
  private def stalenessOf(id: Int, outputs: Outputs): Double =
    def held(node: Int): Double = outputs.get(node).flatten.getOrElse(Unknown)
    val neighbourhood = network.neighboursOf(id).map((neighbour, distance) => (held(neighbour), distance))
    Staleness.of(held(id), network.sources.contains(id), neighbourhood)
end NetworkObserver

object NetworkObserver:
  /** The output of every node: none before its first round. */
  type Outputs = Map[Int, Option[Double]]

  /** The value of a node that has not computed one yet. */
  val Unknown: Double = Double.PositiveInfinity

  /** What was observed at `time` on the network of `version`: the output and the observation of every node. */
  final case class Observation(
      time: Double,
      version: Long,
      outputs: Outputs,
      observations: Map[Int, NodeObservation]
  ):
    /** Whether this observation was made at `now`, on the same network and population. */
    def sees(now: Double, current: Long, population: Set[Int]): Boolean =
      time == now && version == current && outputs.keySet == population

  /** Whether a node holds the same output: unlike `==`, a zero that changes sign is a change and NaN is NaN. */
  def sameOutput(previous: Option[Double], current: Option[Double]): Boolean = (previous, current) match
    case (Some(before), Some(after)) => before.compare(after) == 0
    case (None, None) => true
    case _ => false
end NetworkObserver
