package it.unibo.learning.encoding

import it.unibo.learning.abstractions.AgentState

/** Whether a node has no value yet: its last output is not finite (e.g. a gradient that the source has not reached), or
  * it has none. Such a node is waiting for the field, and every encoding of the original learners would show it as a
  * converged one: no change of its own, none around it.
  */
object MissingValue:

  def of(state: AgentState): Boolean = !state.ownOutputs.lastOption.exists(_.isFinite)

  /** The feature that tells it to the networks: one for a node without a value, zero otherwise. */
  def feature(state: AgentState): Float = if of(state) then 1.0f else 0.0f
