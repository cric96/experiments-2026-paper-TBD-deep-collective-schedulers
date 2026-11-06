package it.unibo.learning.encoding

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.encoding.NeighbourhoodSignals.*

/** Spatial-temporal encoder with topological role and trade-off conditioning. Features, all in [−1, 1]:
  *   - 0: latest trend of the node's output (−1, 0, +1)
  *   - 1: previous trend of the node's output (−1, 0, +1)
  *   - 2: magnitude of the last change of the node's output, tanh(|g_i(t) − g_i(t−1)| / scale)
  *   - 3: largest change of a neighbour's output, tanh(max_j |g_j(t) − g_j(t−1)| / scale)
  *   - 4: fraction of neighbours whose output changed in the last round
  *   - 5: local Bellman-Ford residual (see [[NeighbourhoodSignals.bellmanFordResidual]])
  *   - 6: fraction of neighbours routed through the node in the shortest-path tree (backbone vs. leaf)
  *   - 7: shortest wake-up period among the neighbours, divided by the longest admissible period
  *   - 8: the trade-off weight β (see [[TradeOff.betaFeature]])
  */
object SpatialTemporal:
  val FeatureDim: Int = 9

  /** Number of trends of the node's own output among the features (latest and previous). */
  val TrendFeatures: Int = 2

  /** Positions of the features in the encoded vector. */
  object Feature:
    val LatestTrend: Int = 0
    val PreviousTrend: Int = 1
    val SelfChange: Int = 2
    val NeighbourChange: Int = 3
    val ChangedRatio: Int = 4
    val Residual: Int = 5
    val ChildrenRatio: Int = 6
    val ShortestNeighbourPeriod: Int = 7
    val Beta: Int = 8

  /** The features above, followed by the [[MissingValue]] feature if `missingValueFeature`. */
  def encode(state: AgentState, weight: Double, maxAction: Double, missingValueFeature: Boolean = false): Array[Float] =
    val features = baseFeatures(state, weight, maxAction)
    if missingValueFeature then features :+ MissingValue.feature(state) else features

  private def baseFeatures(state: AgentState, weight: Double, maxAction: Double): Array[Float] =
    val scale = localScale(state)
    val trends = OwnTrends.recent(state, TrendFeatures)
    val change = neighbourhoodChange(state, scale)
    Array(
      trends(0).toFloat,
      trends(1).toFloat,
      selfChangeMagnitude(state, scale).toFloat,
      change.maxMagnitude.toFloat,
      change.changedRatio.toFloat,
      bellmanFordResidual(state).toFloat,
      downstreamChildrenRatio(state).toFloat,
      shortestNeighbourPeriod(state, maxAction),
      TradeOff.betaFeature(weight)
    )

  /** The shortest wake-up period among the neighbours over `maxAction` (1 if the node has no neighbours). */
  private def shortestNeighbourPeriod(state: AgentState, maxAction: Double): Float =
    state.latestSnapshot
      .flatMap(state.neighboursIn(_).values.map(_.oldAction).minOption)
      .fold(1.0f)(period => (period / maxAction).toFloat)
end SpatialTemporal
