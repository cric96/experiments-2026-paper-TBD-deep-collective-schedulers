package it.unibo.learning.agents.offline

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{AgentState, Step, Stratification, Trajectory}
import it.unibo.learning.encoding.{InstabilityIntensity, MissingValue}

/** A logged step relabelled for a target trade-off weight `beta`: both states are conditioned on `beta`, so that a
  * single β-conditioned policy can be trained for every trade-off from the same experience.
  *
  * @param intensity
  *   the instability intensity of the step's state, which determines the counterfactual reward for `beta`
  * @param discount
  *   the semi-Markov discount γ^Δt of the step, where Δt is the wake-up period it chose
  * @param context
  *   the features that only the privileged critics see of the step ([[it.unibo.learning.abstractions.CriticContext]]),
  *   and `nextContext` those of the next step, shared by the relabelled copies of a step; empty if not measured
  * @param costScale
  *   the factor the measured cost was divided by ([[it.unibo.learning.abstractions.MobileOutcome.costScale]])
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
    deferredBeta: Option[DeferredBeta] = None,
    context: Array[Float] = Array.emptyFloatArray,
    nextContext: Array[Float] = Array.emptyFloatArray,
    costScale: Double = 1.0
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

  /** The trade-off weights every step is relabelled for, besides the learner's own one (original objective). */
  val CounterfactualWeights: Seq[Double] = TradeOff.BetaScale.Ratio.counterfactualWeights

  /** How many times the learner's own weight is repeated, to oversample the trade-off it was configured with. */
  val OwnWeightRepetitions: Int = 2

  /** Wake-up periods shorter than this are discounted as if they lasted this long. */
  val MinDiscountPeriod: Double = 0.5

  def targetWeights(ownWeight: Double, counterfactual: Seq[Double] = CounterfactualWeights): Seq[Double] =
    Seq.fill(OwnWeightRepetitions)(ownWeight) ++ counterfactual

  /** Relabels every step of `trajectory` (encoded with `encode`, whose β feature is at `betaFeatureIndex`) for every
    * target weight of `scale`; the last step is terminal and points to itself. The mobile steps are conditioned on
    * `betaFeature` (the raw β of the mobile encoding by default), the others on the feature of `scale`.
    */
  def relabel(
      trajectory: Trajectory,
      ownWeight: Double,
      gamma: Double,
      encode: AgentState => Array[Float],
      betaFeatureIndex: Int,
      mobile: Boolean = false,
      scale: TradeOff.BetaScale = TradeOff.BetaScale.Ratio,
      betaFeature: Double => Float = _.toFloat
  ): Seq[RelabelledStep] =
    val weights = targetWeights(ownWeight, scale.counterfactualWeights)
    if mobile then relabelMobile(trajectory, weights, gamma, encode, DeferredFeature(betaFeatureIndex, betaFeature))
    else relabelLegacy(trajectory, weights, gamma, encode, betaFeatureIndex, scale)

  private def relabelLegacy(
      trajectory: Trajectory,
      weights: Seq[Double],
      gamma: Double,
      encode: AgentState => Array[Float],
      betaFeatureIndex: Int,
      scale: TradeOff.BetaScale
  ): Seq[RelabelledStep] =
    val states = trajectory.map(step => encode(step.state)).toIndexedSeq
    trajectory.zipWithIndex.flatMap { case (step, index) =>
      val isTerminal = index == trajectory.size - 1
      val nextState = if isTerminal then states(index) else states(index + 1)
      val intensity = InstabilityIntensity.of(step.state)
      val discount = math.pow(gamma, math.max(MinDiscountPeriod, step.action)).toFloat
      weights.map { beta =>
        def conditioned(features: Array[Float]) = withBetaFeature(features, beta, betaFeatureIndex, scale)
        RelabelledStep(step, conditioned(states(index)), conditioned(nextState), beta, intensity, isTerminal, discount)
      }
    }

  private def relabelMobile(
      trajectory: Trajectory,
      weights: Seq[Double],
      gamma: Double,
      encode: AgentState => Array[Float],
      betaFeature: DeferredFeature
  ): Seq[RelabelledStep] =
    trajectory.flatMap { step =>
      val outcome =
        step.mobileOutcome.getOrElse(throw new IllegalArgumentException("Mobile training requires measured outcomes"))
      val context = outcome.context.features
      val nextContext = outcome.nextContext.features
      lazy val locallyUnstable = isLocallyUnstable(step.state)
      outcome.errorIntegrals.toSeq.sortBy(_._1).flatMap { case (tolerance, cost) =>
        val current = encode(step.state.copy(tolerance = tolerance))
        val next = encode(outcome.nextState.copy(tolerance = tolerance))
        val unstable = outcome.stratification match
          case Stratification.ByCost => cost > 0.0
          case Stratification.ByLocalRegime => locallyUnstable
        weights.map { beta =>
          RelabelledStep(
            step,
            current,
            next,
            beta,
            if unstable then 1.0 else 0.0,
            outcome.terminal,
            math.pow(gamma, outcome.duration).toFloat,
            Some(cost),
            outcome.energy,
            Some(betaFeature.of(beta)),
            context,
            nextContext,
            outcome.costScale
          )
        }
      }
    }

  /** Whether the node sees a change of its own or of its neighbourhood, or has no value yet. */
  private def isLocallyUnstable(state: AgentState): Boolean =
    InstabilityIntensity.of(state) > 0.0 || MissingValue.of(state)

  /** Where the β feature goes and how a β is encoded in it. */
  final private case class DeferredFeature(index: Int, feature: Double => Float):
    def of(beta: Double): DeferredBeta = DeferredBeta(index, feature(beta))

  private def withBetaFeature(
      features: Array[Float],
      beta: Double,
      betaFeatureIndex: Int,
      scale: TradeOff.BetaScale
  ): Array[Float] =
    val conditioned = features.clone()
    conditioned(betaFeatureIndex) = scale.feature(beta)
    conditioned
end Relabelling
