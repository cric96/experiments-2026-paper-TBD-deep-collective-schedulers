package it.unibo.learning.agents.offline

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{AgentState, Step, Trajectory}
import it.unibo.learning.encoding.InstabilityIntensity

/** A logged step relabelled for a target trade-off weight `beta`: both states are conditioned on `beta`, so that a
  * single β-conditioned policy can be trained for every trade-off from the same experience.
  *
  * @param intensity
  *   the instability intensity of the step's state, which determines the counterfactual reward for `beta`
  * @param discount
  *   the semi-Markov discount γ^Δt of the step, where Δt is the wake-up period it chose
  */
final case class RelabelledStep(
    step: Step,
    state: Array[Float],
    nextState: Array[Float],
    beta: Double,
    intensity: Double,
    isTerminal: Boolean,
    discount: Float,
    errorIntegral: Option[Double] = None,
    energyCost: Double = 1.0,
    deferredBeta: Option[DeferredBeta] = None
):
  def isUnstable: Boolean = intensity > 0.0

  def terminalFlag: Float = if isTerminal then 1.0f else 0.0f

  def conditionedState: Array[Float] = DeferredBeta.conditioned(state, deferredBeta)

  def conditionedNextState: Array[Float] = DeferredBeta.conditioned(nextState, deferredBeta)

/** The β feature that still has to be written into features shared by the relabelled copies of a step: each copy would
  * otherwise hold its own conditioned arrays, and a mobile replay could keep only a fraction of an episode.
  */
final case class DeferredBeta(index: Int, value: Float):
  def applyTo(features: Array[Float]): Array[Float] =
    val conditioned = features.clone()
    conditioned(index) = value
    conditioned

object DeferredBeta:
  def conditioned(features: Array[Float], deferred: Option[DeferredBeta]): Array[Float] =
    deferred.fold(features)(_.applyTo(features))

/** Hindsight relabelling of the trajectories collected with one trade-off weight for many others. */
object Relabelling:

  /** The trade-off weights every step is relabelled for, besides the learner's own one. */
  val CounterfactualWeights: Seq[Double] = Seq(0.5, 0.8, 0.95, 0.99, 1.0)

  /** How many times the learner's own weight is repeated, to oversample the trade-off it was configured with. */
  val OwnWeightRepetitions: Int = 2

  /** Wake-up periods shorter than this are discounted as if they lasted this long. */
  val MinDiscountPeriod: Double = 0.5

  def targetWeights(ownWeight: Double): Seq[Double] = Seq.fill(OwnWeightRepetitions)(ownWeight) ++ CounterfactualWeights

  /** Relabels every step of `trajectory` (encoded with `encode`, whose β feature is at `betaFeatureIndex`) for every
    * target weight; the last step is terminal and points to itself.
    */
  def relabel(
      trajectory: Trajectory,
      ownWeight: Double,
      gamma: Double,
      encode: AgentState => Array[Float],
      betaFeatureIndex: Int,
      mobile: Boolean = false
  ): Seq[RelabelledStep] =
    if mobile then relabelMobile(trajectory, ownWeight, gamma, encode, betaFeatureIndex)
    else relabelLegacy(trajectory, ownWeight, gamma, encode, betaFeatureIndex)

  private def relabelLegacy(
      trajectory: Trajectory,
      ownWeight: Double,
      gamma: Double,
      encode: AgentState => Array[Float],
      betaFeatureIndex: Int
  ): Seq[RelabelledStep] =
    val states = trajectory.map(step => encode(step.state)).toIndexedSeq
    trajectory.zipWithIndex.flatMap { case (step, index) =>
      val isTerminal = index == trajectory.size - 1
      val nextState = if isTerminal then states(index) else states(index + 1)
      val intensity = InstabilityIntensity.of(step.state)
      val discount = math.pow(gamma, math.max(MinDiscountPeriod, step.action)).toFloat
      targetWeights(ownWeight).map { beta =>
        def conditioned(features: Array[Float]) = withBetaFeature(features, beta, betaFeatureIndex)
        RelabelledStep(step, conditioned(states(index)), conditioned(nextState), beta, intensity, isTerminal, discount)
      }
    }

  private def relabelMobile(
      trajectory: Trajectory,
      ownWeight: Double,
      gamma: Double,
      encode: AgentState => Array[Float],
      betaFeatureIndex: Int
  ): Seq[RelabelledStep] =
    trajectory.flatMap { step =>
      val outcome =
        step.mobileOutcome.getOrElse(throw new IllegalArgumentException("Mobile training requires measured outcomes"))
      outcome.errorIntegrals.toSeq.sortBy(_._1).flatMap { case (tolerance, cost) =>
        val current = encode(step.state.copy(tolerance = tolerance))
        val next = encode(outcome.nextState.copy(tolerance = tolerance))
        targetWeights(ownWeight).map { beta =>
          RelabelledStep(
            step,
            current,
            next,
            beta,
            if cost > 0.0 then 1.0 else 0.0,
            outcome.terminal,
            math.pow(gamma, outcome.duration).toFloat,
            Some(cost),
            outcome.energy,
            Some(DeferredBeta(betaFeatureIndex, beta.toFloat))
          )
        }
      }
    }

  private def withBetaFeature(features: Array[Float], beta: Double, betaFeatureIndex: Int): Array[Float] =
    val conditioned = features.clone()
    conditioned(betaFeatureIndex) = TradeOff.betaFeature(beta)
    conditioned
end Relabelling
