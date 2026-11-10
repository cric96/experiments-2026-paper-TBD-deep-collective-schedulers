package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.model.reactions.NetworkObserver.*
import it.unibo.alchemist.model.{Node, Position}
import it.unibo.learning.abstractions.CriticContext
import it.unibo.mobility.*
import it.unibo.scafi.Sensors

/** Observes the output of every node after each simulator event and records it in the episode ([[MobileEpisode]]).
  *
  * A simulator event changes at most the output of the node that ran, so at an instant that was already observed on the
  * same network only the nodes whose output changed, and the nodes whose cost it changes (the neighbours for the
  * staleness and the neighbourhood error, those it reaches for the network error), are observed again: the others would
  * be recorded with an unchanged observation, which adds nothing.
  */
final class NetworkObserver[T, P <: Position[P]](network: NetworkView[T, P], config: MobilityConfig):
  private var last: Option[Observation] = None
  private var networkSize: Option[Double] = None

  /** Forgets the last observation, e.g. when a new episode starts. */
  def forget(): Unit = last = None

  /** The factor the network costs are divided by ([[CostSignal.Network]]): the effective size of the first network they
    * were computed on, so that they are of order one and on one scale for the whole run; one for the other signals.
    */
  def costScale: Double = networkSize.getOrElse(1.0)

  /** What the privileged critics see of a decision of `id` ([[CriticContext]]), from the last observation, when the
    * last change of the network happened `sinceChange` seconds ago.
    */
  def contextOf(id: Int, sinceChange: Double): CriticContext =
    last
      .flatMap(previous =>
        NetworkObserver.contextOf(id, previous.observations, config.lossScale, eventAge(sinceChange))
      )
      .getOrElse(CriticContext.Unknown)

  def observe(now: Double, episode: MobileEpisode, nodes: List[Node[T]]): Unit =
    val outputs = nodes.map(node => node.getId -> node.valueOf[Double](Sensors.localComputation)).toMap
    last.filter(_.sees(now, network.version, outputs.keySet)) match
      case Some(previous) => observeChanges(now, episode, previous, outputs)
      case None => observeAll(now, episode, outputs)

  private def observeAll(now: Double, episode: MobileEpisode, outputs: Map[Int, Option[Double]]): Unit =
    val observations = observationsOf(episode, outputs.keySet, outputs, unchanged = Map.empty)
    episode.observe(now, observations)
    last = Some(Observation(now, network.version, outputs, observations))

  private def observeChanges(now: Double, episode: MobileEpisode, previous: Observation, outputs: Outputs): Unit =
    val changed = outputs.keySet.filterNot(id => sameOutput(previous.outputs(id), outputs(id)))
    if changed.nonEmpty then
      val updates = observationsOf(episode, affectedBy(episode, changed), outputs, previous.observations -- changed)
      val observations = previous.observations ++ updates
      episode.observeChanges(now, updates, observations)
      last = Some(Observation(now, network.version, outputs, observations))

  /** The nodes whose cost a change of the output of the `changed` ones changes. */
  private def affectedBy(episode: MobileEpisode, changed: Set[Int]): Set[Int] = chargedSignal(episode) match
    case CostSignal.Error => changed
    case CostSignal.Staleness | CostSignal.Neighbourhood => changed ++ changed.flatMap(neighbours)
    case CostSignal.Network => weights.reachedFrom(changed)

  /** The observations of `ids`, given the `unchanged` ones of the nodes whose output did not change. */
  private def observationsOf(
      episode: MobileEpisode,
      ids: Set[Int],
      outputs: Outputs,
      unchanged: Map[Int, NodeObservation]
  ): Map[Int, NodeObservation] = chargedSignal(episode) match
    case CostSignal.Network =>
      val errors = outputs.keySet.map { id =>
        id -> unchanged.get(id).fold(errorOf(episode, id, outputs))(_.error)
      }.toMap
      networkObservations(ids, errors)
    case local => ids.map(id => id -> observationOf(episode, id, outputs, local)).toMap

  /** The observations of `ids` charged with the network error ([[NetworkError]]) of the `errors` of every node. */
  private def networkObservations(ids: Set[Int], errors: Map[Int, Double]): Map[Int, NodeObservation] =
    val current = weights
    val scale = networkSize.getOrElse(current.effectiveSize)
    networkSize = Some(scale)
    val costs = current.costs(errors, config.lossScale)
    ids.map(id => id -> NodeObservation(errors(id), network.isReachable(id), costs(id) / scale)).toMap

  private def weights: SpatialWeights = network.spatialWeights(config.spatialDiscount)

  /** The observation of `id` charged with a `signal` that depends on its neighbourhood only. */
  private def observationOf(episode: MobileEpisode, id: Int, outputs: Outputs, signal: CostSignal): NodeObservation =
    val error = errorOf(episode, id, outputs)
    val cost = signal match
      case CostSignal.Error => error
      case CostSignal.Staleness => stalenessOf(id, outputs)
      case CostSignal.Neighbourhood =>
        NeighbourhoodError.of(error, neighbours(id).map(errorOf(episode, _, outputs)), config.lossScale)
      case CostSignal.Network =>
        throw new IllegalArgumentException("The network cost depends on every node: see networkObservations")
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

  /** The context of a decision of `id` ([[CriticContext]]) from the `observations` of every node, in units of `scale`,
    * with the `eventAge` of the last change: its own loss and the mean loss of the network count each error as at most
    * `scale`, an unbounded charged cost as `scale`.
    */
  def contextOf(
      id: Int,
      observations: Map[Int, NodeObservation],
      scale: Double,
      eventAge: Double
  ): Option[CriticContext] =
    def loss(error: Double): Double = (if error < scale then error else scale) / scale
    observations.get(id).map { own =>
      CriticContext(
        loss(own.error),
        (if own.cost.isFinite then own.cost else scale) / scale,
        observations.values.map(observation => loss(observation.error)).sum / observations.size,
        observations.values.count(!_.error.isFinite).toDouble / observations.size,
        eventAge
      )
    }

  /** The age of a change of the network `seconds` ago, in [0, 1]: one after [[CriticContext.EventAgeHorizon]]. */
  def eventAge(seconds: Double): Double = math.min(1.0, seconds / CriticContext.EventAgeHorizon)

  /** Whether a node holds the same output: unlike `==`, a zero that changes sign is a change and NaN is NaN. */
  def sameOutput(previous: Option[Double], current: Option[Double]): Boolean = (previous, current) match
    case (Some(before), Some(after)) => before.compare(after) == 0
    case (None, None) => true
    case _ => false
end NetworkObserver
