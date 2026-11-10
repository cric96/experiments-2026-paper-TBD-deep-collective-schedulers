package it.unibo.learning.agents.offline

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.Step
import it.unibo.learning.encoding.{InstabilityIntensity, MissingValue}

/** The convergence cost of a learner trained on a measured cost, made hybrid: the measured cost plus the `weights` of
  * the trade-off β times the convergence surrogate of the node's instability ([[TradeOff.convergenceSurrogate]], with
  * the reward `shape`), which asks for a fast reaction to a change whatever the program, and for a period that β tunes
  * in steady state. Without a weight it is the measured cost.
  */
final case class HybridConvergence(weights: SurrogateWeights, shape: TradeOff.RewardShape, min: Double, max: Double):

  /** The convergence cost of `step`, which paid the `measured` cost, for the trade-off `beta`. */
  def cost(step: Step, measured: Double, beta: Double): Double =
    val weight = weights.at(beta)
    if weight == 0.0 then measured
    else
      val intensity = shape.intensity(InstabilityIntensity.of(step.state), !MissingValue.of(step.state))
      measured + weight * TradeOff.convergenceSurrogate(intensity, step.action, min, max, shape)
