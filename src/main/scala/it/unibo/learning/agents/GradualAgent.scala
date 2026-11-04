package it.unibo.learning.agents

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.agents.GradualAgent.*
import it.unibo.learning.agents.ReactiveSchedule.*

/** Gradual controller of the period, a prudent baseline that does not learn: the additive-increase /
  * multiplicative-decrease law of TCP's congestion control (Chiu and Jain, 1989) applied to the period. It halves after
  * every round that changed the node's own output and grows by `step` after every round that left it unchanged, within
  * [`shortest`, `longest`]; unlike [[TrickleAgent]] it takes several changing rounds to reach the shortest period. It
  * ignores the trade-off weight β.
  */
final class GradualAgent(shortest: Double, longest: Double, step: Double = DefaultStep) extends Learner:
  override def policy: AgentState => Double = state =>
    val last = lastPeriod(state).getOrElse(shortest)
    bounded(if changedInLastRound(state) then last * DecreaseFactor else last + step, shortest, longest)

object GradualAgent:

  /** Seconds added to the period after a round that left the output unchanged. */
  val DefaultStep: Double = 0.5

  /** Share of the period kept after a round that changed the output. */
  val DecreaseFactor: Double = 0.5
