package it.unibo.learning.agents

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.agents.ReactiveSchedule.*
import it.unibo.learning.agents.TrickleAgent.*

/** Trickle (Levis et al., NSDI 2004; RFC 6206) on the node's own output, a reactive baseline that does not learn: the
  * period doubles after every round that left the output unchanged, up to `longest`, and falls back to `shortest` as
  * soon as a round changes it. Every round both computes and sends, so Trickle's listen-only first half of the interval
  * and its suppression counter have no counterpart here. It ignores the trade-off weight β.
  */
final class TrickleAgent(shortest: Double, longest: Double) extends Learner:
  override def policy: AgentState => Double = state =>
    if changedInLastRound(state) then shortest
    else bounded(lastPeriod(state).getOrElse(shortest) * GrowthFactor, shortest, longest)

object TrickleAgent:

  /** How much the period grows after a round that left the output unchanged (RFC 6206 doubles the interval). */
  val GrowthFactor: Double = 2.0
