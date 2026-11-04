package it.unibo.learning.encoding

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.AgentState.{NeighborInfo, Snapshot}
import it.unibo.learning.encoding.NeighbourhoodSignals.*

/** Star-graph encoder for the EdgeGAT networks: the ego node features followed by one slot of features for each of the
  * `MaxNeighbors` nearest neighbours (padded with idle slots).
  *
  * Ego features: latest and previous own trend, magnitude of the last own change, children ratio, β feature. Slot
  * features: sign and magnitude of the neighbour's last change, drift of the edge difference, relative order of the
  * outputs, normalized distance, normalized wake-up period.
  */
object StarGraph:
  val SelfDim: Int = 5
  val MaxNeighbors: Int = 8
  val NeighDim: Int = 6
  val TotalDim: Int = SelfDim + MaxNeighbors * NeighDim

  /** Position of the β feature among the ego features. */
  val BetaFeature: Int = SelfDim - 1

  /** Position of the children ratio among the ego features. */
  val ChildrenFeature: Int = 3

  /** Position of the relative order of the outputs within a neighbour slot. */
  val RelativeOrderFeature: Int = 3

  /** Distance that normalizes the neighbours' distances to [0, 1]. */
  val MaxDistance: Double = 100.0

  /** Slot of a missing neighbour: nothing changes and it sleeps as long as possible. */
  val IdleSlot: Seq[Float] = Seq(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f)

  /** What a neighbour slot is computed against: the previous snapshot and the node's own last two outputs. */
  final private case class Reference(previous: Snapshot, ownNow: Double, ownBefore: Double, scale: Double)

  def encode(state: AgentState, weight: Double, maxAction: Double): Array[Float] =
    val scale = localScale(state)
    (egoFeatures(state, weight, scale) ++ neighbourSlots(state, scale, maxAction).flatten).toArray

  private def egoFeatures(state: AgentState, weight: Double, scale: Double): Seq[Float] =
    val trends = OwnTrends.recent(state, SpatialTemporal.TrendFeatures)
    Seq(
      trends(0).toFloat,
      trends(1).toFloat,
      selfChangeMagnitude(state, scale).toFloat,
      downstreamChildrenRatio(state).toFloat,
      TradeOff.betaFeature(weight)
    )

  private def neighbourSlots(state: AgentState, scale: Double, maxAction: Double): Seq[Seq[Float]] =
    val current = state.latestSnapshot.getOrElse(Map.empty[Int, NeighborInfo])
    val previous = state.lastTwoSnapshots.fold(current) { case (previous, _) => previous }
    val reference = Reference(
      previous,
      ownNow = current.get(state.me).fold(0.0)(_.data),
      ownBefore = previous.get(state.me).fold(0.0)(_.data),
      scale
    )
    nearestNeighbours(state, current)
      .map { case (id, neighbour) => slotOf(id, neighbour, reference, maxAction) }
      .padTo(MaxNeighbors, IdleSlot)

  /** The (at most `MaxNeighbors`) neighbours in `snapshot`, nearest first. */
  def nearestNeighbours(state: AgentState, snapshot: Snapshot): List[(Int, NeighborInfo)] =
    state.neighboursIn(snapshot).toList.sortBy { case (_, neighbour) => neighbour.distance }.take(MaxNeighbors)

  private def slotOf(id: Int, neighbour: NeighborInfo, reference: Reference, maxAction: Double): Seq[Float] =
    val now = bounded(neighbour.data, reference.scale)
    val before = reference.previous.get(id).fold(now)(info => bounded(info.data, reference.scale))
    val change = now - before
    Seq(
      math.signum(change).toFloat,
      math.tanh(math.abs(change) / reference.scale).toFloat,
      edgeDrift(now, before, reference).toFloat,
      relativeOrder(now, reference).toFloat,
      math.min(1.0, neighbour.distance / MaxDistance).toFloat,
      (neighbour.oldAction / maxAction).toFloat
    )

  private def edgeDrift(now: Double, before: Double, reference: Reference): Double =
    if reference.ownNow.isFinite && reference.ownBefore.isFinite then
      math.tanh(math.abs((reference.ownNow - now) - (reference.ownBefore - before)) / reference.scale)
    else 0.0

  private def relativeOrder(now: Double, reference: Reference): Double =
    if reference.ownNow.isFinite then math.tanh((now - reference.ownNow) / reference.scale) else 0.0
end StarGraph
