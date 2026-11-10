package it.unibo.learning.agents.offline

import it.unibo.learning.TradeOff

/** How much of the convergence surrogate a hybrid objective adds for each trade-off β ([[HybridConvergence]]): from
  * `atLowest` at `lowestBeta` to `atHighest` at `highestBeta`, geometrically, and constant outside. A single weight
  * sets a band of rounds whatever β, so a weight that grows with β lets β choose the band too.
  */
final case class SurrogateWeights(atLowest: Double, atHighest: Double, lowestBeta: Double, highestBeta: Double):
  require(
    atLowest >= 0.0 && atLowest.isFinite && atHighest >= 0.0 && atHighest.isFinite,
    "Weights must be non-negative"
  )
  require(atLowest == atHighest || (atLowest > 0.0 && atHighest > 0.0), "A schedule grows between positive weights")
  require(lowestBeta < highestBeta, s"The betas of a schedule must be ordered: $lowestBeta, $highestBeta")

  /** The weight of the surrogate at `beta`. */
  def at(beta: Double): Double =
    if atLowest == atHighest then atLowest
    else
      val position = math.max(0.0, math.min(1.0, (beta - lowestBeta) / (highestBeta - lowestBeta)))
      atLowest * math.pow(atHighest / atLowest, position)

object SurrogateWeights:
  /** The same `weight` at every β. */
  def constant(weight: Double): SurrogateWeights = SurrogateWeights(weight, weight, 0.0, 1.0)

  /** The constant `weight`, or a schedule from it at the lowest β of the `scale` to `highest` at its highest β. */
  def of(weight: Double, highest: Option[Double], scale: TradeOff.BetaScale): SurrogateWeights =
    highest.fold(constant(weight)) { high =>
      val betas = scale.counterfactualWeights
      SurrogateWeights(weight, high, betas.min, betas.max)
    }
