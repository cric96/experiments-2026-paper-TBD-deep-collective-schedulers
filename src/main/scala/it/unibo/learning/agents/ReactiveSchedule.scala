package it.unibo.learning.agents

import it.unibo.learning.abstractions.AgentState
import it.unibo.util.TemporalInfo

/** What the reactive baselines that do not learn ([[TrickleAgent]], [[GradualAgent]]) read of a node's own history:
  * whether its last round changed its output, and how long it had slept before that round.
  */
object ReactiveSchedule:

  /** Own outputs needed to tell whether the last round changed the output. */
  private val LastTwoOutputs: Int = 2

  /** Whether the output of the last round differs from the previous one; two unknown (non-finite) outputs are equal. */
  def changedInLastRound(state: AgentState): Boolean =
    TemporalInfo.computeDeltaTrend(state.ownOutputs.takeRight(LastTwoOutputs)).lastOption.exists(_ != 0.0)

  /** The period the node chose in its previous round, i.e. how long it slept before its last one. */
  def lastPeriod(state: AgentState): Option[Double] =
    state.latestSnapshot.flatMap(_.get(state.me)).map(_.oldAction)

  /** `period` bounded to [`shortest`, `longest`]. */
  def bounded(period: Double, shortest: Double, longest: Double): Double = math.min(longest, math.max(shortest, period))
