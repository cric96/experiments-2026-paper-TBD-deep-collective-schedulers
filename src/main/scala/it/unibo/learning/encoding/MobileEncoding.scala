package it.unibo.learning.encoding

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.AgentState.{NeighborInfo, Snapshot}

/** Optional local features of the mobile encoding, appended to the node's own features in this order. `programAgnostic`
  * adds none: it blanks the base features that assume the gradient's semantics (an output that is a distance from the
  * source), which describe states the other programs never reach.
  */
final case class MobileFeatures(
    ownSpeed: Boolean = false,
    unexplainedChange: Boolean = false,
    dependants: Boolean = false,
    neighbourSpeed: Boolean = false,
    programAgnostic: Boolean = false
):
  def count: Int = Seq(ownSpeed, unexplainedChange, dependants, neighbourSpeed).count(identity)

object MobileFeatures:
  val None: MobileFeatures = MobileFeatures()

/** Adds only local motion, message-age and tolerance features. No ground truth or source event enters the policy. */
object MobileEncoding:
  val ExtraSelfDim: Int = 6
  val ExtraNeighDim: Int = 4
  val SelfDim: Int = StarGraph.SelfDim + ExtraSelfDim
  val NeighDim: Int = StarGraph.NeighDim + ExtraNeighDim
  val FeatureDim: Int = SpatialTemporal.FeatureDim + ExtraSelfDim
  val ToleranceScale: Double = 500.0
  val DistanceRateScale: Double = 10.0

  /** A walking pace (m/s): the own-speed feature is tanh(speed / SpeedScale). */
  val SpeedScale: Double = 1.0

  /** How far (m) a neighbour's value may be from the node's value plus their distance to count as its child. */
  val DependencyTolerance: Double = 2.0

  def selfDim(features: MobileFeatures): Int = SelfDim + features.count
  def featureDim(features: MobileFeatures): Int = FeatureDim + features.count

  /** The features of a neighbour slot of the graph: the neighbour's speed is one more, when it is asked for. */
  def neighDim(features: MobileFeatures): Int = NeighDim + (if features.neighbourSpeed then 1 else 0)

  private def snapshots(state: AgentState): (Snapshot, Snapshot) =
    state.lastTwoSnapshots.getOrElse((Map.empty, state.latestSnapshot.getOrElse(Map.empty)))

  def duration(state: AgentState): Double =
    val (previous, current) = snapshots(state)
    math.max(0.0, current.get(state.me).fold(0.0)(_.observedAt) - previous.get(state.me).fold(0.0)(_.observedAt))

  private def extras(state: AgentState, maxPeriod: Double): Array[Float] =
    val (previous, current) = snapshots(state)
    val before = state.neighboursIn(previous).keySet
    val now = state.neighboursIn(current).keySet
    val union = before union now
    val churn = if union.isEmpty then 0.0 else (union.size - (before intersect now).size).toDouble / union.size
    val slots =
      state.neighboursIn(current).toSeq.map { case (id, neighbour) => edgeExtras(state, id, neighbour, maxPeriod) }
    val ownRate = for
      old <- previous.get(state.me)
      latest <- current.get(state.me)
    yield rate(latest.data, old.data, duration(state), NeighbourhoodSignals.localScale(state))
    Array(
      (state.tolerance / ToleranceScale).toFloat,
      math.tanh(duration(state) / maxPeriod).toFloat,
      churn.toFloat,
      ownRate.getOrElse(0.0f),
      slots.map(_(1)).maxOption.getOrElse(0.0f),
      slots.map(_(2)).maxOption.getOrElse(0.0f)
    )

  private def optionalExtras(state: AgentState, features: MobileFeatures): Array[Float] =
    Array(
      Option.when(features.ownSpeed)(speedFeature(state.speed).toFloat),
      Option.when(features.unexplainedChange)(unexplainedChange(state)),
      Option.when(features.dependants)(dependants(state)),
      Option.when(features.neighbourSpeed)(neighbourSpeed(state))
    ).flatten

  /** How fast the neighbours move, on average: each one's own speed, as the own-speed feature. */
  private def neighbourSpeed(state: AgentState): Float =
    val speeds = state.neighboursIn(state.latestSnapshot.getOrElse(Map.empty)).values.map(_.speed).toSeq
    if speeds.isEmpty then 0.0f else (speeds.map(speedFeature).sum / speeds.size).toFloat

  private def speedFeature(speed: Double): Double = math.tanh(speed / SpeedScale)

  /** How fast the own value changed beyond what the change of the distances to the common neighbours explains: a node
    * that only moves sees its value change by at most that much, an upstream event changes it further.
    */
  private def unexplainedChange(state: AgentState): Float =
    val (previous, current) = snapshots(state)
    val change = for
      old <- previous.get(state.me)
      latest <- current.get(state.me)
    yield
      val common = state.neighboursIn(current).keySet intersect state.neighboursIn(previous).keySet
      val motion = common.map(id => math.abs(current(id).distance - previous(id).distance)).maxOption.getOrElse(0.0)
      unexplainedRate(latest.data, old.data, motion, duration(state), NeighbourhoodSignals.localScale(state))
    change.getOrElse(0.0f)

  private def unexplainedRate(now: Double, before: Double, explained: Double, elapsed: Double, scale: Double): Float =
    if elapsed <= 0.0 then 0.0f
    else if !now.isFinite || !before.isFinite then if now == before then 0.0f else 1.0f
    else math.tanh(math.max(0.0, math.abs(now - before) - explained) / (elapsed * scale)).toFloat

  /** The share of the neighbour slots taken by children: neighbours whose value is the node's value plus their
    * distance, so that a stale value of the node propagates to them.
    */
  private def dependants(state: AgentState): Float =
    val current = state.latestSnapshot.getOrElse(Map.empty)
    current.get(state.me).filter(_.data.isFinite).fold(0.0f) { own =>
      val children = state.neighboursIn(current).values.count { neighbour =>
        math.abs(neighbour.data - own.data - neighbour.distance) <= DependencyTolerance
      }
      math.min(1.0, children.toDouble / StarGraph.MaxNeighbors).toFloat
    }

  private def rate(now: Double, before: Double, elapsed: Double, scale: Double): Float =
    if elapsed <= 0.0 then 0.0f
    else if !now.isFinite || !before.isFinite then if now == before then 0.0f else 1.0f
    else math.tanh(math.abs(now - before) / (elapsed * scale)).toFloat

  private def edgeExtras(state: AgentState, id: Int, neighbour: NeighborInfo, maxPeriod: Double): Array[Float] =
    val (previous, current) = snapshots(state)
    val now = current.get(state.me).fold(0.0)(_.observedAt)
    val old = previous.get(id)
    Array(
      old.fold(0.0f)(info =>
        rate(neighbour.data, info.data, neighbour.observedAt - info.observedAt, NeighbourhoodSignals.localScale(state))
      ),
      old.fold(0.0f)(info => rate(neighbour.distance, info.distance, duration(state), DistanceRateScale)),
      math.tanh(math.max(0.0, now - neighbour.observedAt) / maxPeriod).toFloat,
      1.0f // Explicit presence; padded slots have zero here.
    )

  def spatial(
      state: AgentState,
      beta: Double,
      maxPeriod: Double,
      features: MobileFeatures = MobileFeatures.None
  ): Array[Float] =
    val encoded =
      SpatialTemporal.encode(state, beta, maxPeriod) ++ extras(state, maxPeriod) ++ optionalExtras(state, features)
    encoded(SpatialTemporal.Feature.Beta) = beta.toFloat
    if features.programAgnostic then
      encoded(SpatialTemporal.Feature.Residual) = 0.0f
      encoded(SpatialTemporal.Feature.ChildrenRatio) = 0.0f
    encoded

  def graph(
      state: AgentState,
      beta: Double,
      maxPeriod: Double,
      features: MobileFeatures = MobileFeatures.None
  ): Array[Float] =
    val base = StarGraph.encode(state, beta, maxPeriod)
    base(StarGraph.BetaFeature) = beta.toFloat
    if features.programAgnostic then blankGradientSemantics(base)
    val neighbours = StarGraph.nearestNeighbours(state, state.latestSnapshot.getOrElse(Map.empty))
    val extraDim = neighDim(features) - StarGraph.NeighDim
    val slots = (0 until StarGraph.MaxNeighbors).flatMap { slot =>
      val start = StarGraph.SelfDim + slot * StarGraph.NeighDim
      val extra = neighbours.lift(slot).fold(Array.fill(extraDim)(0.0f)) { case (id, neighbour) =>
        edgeExtras(state, id, neighbour, maxPeriod) ++
          Option.when(features.neighbourSpeed)(speedFeature(neighbour.speed).toFloat)
      }
      base.slice(start, start + StarGraph.NeighDim) ++ extra
    }
    base.take(StarGraph.SelfDim) ++ extras(state, maxPeriod) ++ optionalExtras(state, features) ++ slots

  /** Blanks the children ratio of the node and the relative order of every neighbour slot of a star graph. */
  private def blankGradientSemantics(graph: Array[Float]): Unit =
    graph(StarGraph.ChildrenFeature) = 0.0f
    (0 until StarGraph.MaxNeighbors).foreach { slot =>
      graph(StarGraph.SelfDim + slot * StarGraph.NeighDim + StarGraph.RelativeOrderFeature) = 0.0f
    }
end MobileEncoding
