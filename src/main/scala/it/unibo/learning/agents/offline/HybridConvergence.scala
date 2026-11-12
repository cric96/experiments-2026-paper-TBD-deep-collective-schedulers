package it.unibo.learning.agents.offline

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.Step
import it.unibo.learning.encoding.{InstabilityIntensity, MissingValue, NeighbourhoodMotion}

/** The convergence cost of a learner trained on a measured cost, made hybrid: the measured cost plus the `weights` of
  * the trade-off β times the convergence surrogate of the node's instability ([[TradeOff.convergenceSurrogate]], with
  * the reward `shape`), which asks for a fast reaction to a change whatever the program, and for a period that β tunes
  * in steady state. Without a weight it is the measured cost. The instability of a moving neighbourhood is continuous,
  * not a change to react to: the `motionAttenuation` (in [0, 1]) discounts it by that fraction of the share of the
  * neighbourhood that moves ([[NeighbourhoodMotion]]), so that the measured cost alone asks for the rounds in motion.
  */
final case class HybridConvergence(
    weights: SurrogateWeights,
    shape: TradeOff.RewardShape,
    min: Double,
    max: Double,
    motionAttenuation: Double = 0.0
):
  require(motionAttenuation >= 0.0 && motionAttenuation <= 1.0, "The motion attenuation is a fraction")

  /** The convergence cost of `step`, which paid the `measured` cost, for the trade-off `beta`. */
  def cost(step: Step, measured: Double, beta: Double): Double =
    val weight = weights.at(beta)
    if weight == 0.0 then measured
    else
      val intensity = shape.intensity(InstabilityIntensity.of(step.state), !MissingValue.of(step.state))
      measured + weight * TradeOff.convergenceSurrogate(intensity * stillness(step), step.action, min, max, shape)

  /** How much of the instability of `step` counts: all of it without the attenuation or without motion. */
  private def stillness(step: Step): Double =
    if motionAttenuation == 0.0 then 1.0 else 1.0 - motionAttenuation * NeighbourhoodMotion.share(step.state)
