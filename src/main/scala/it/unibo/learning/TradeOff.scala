package it.unibo.learning

/** The convergence/energy trade-off that every policy optimizes, controlled by the weight β ∈ [0, 1]: β = 1 means that
  * only convergence matters, lower values make slow (energy-saving) rounds more and more attractive.
  *
  * The wake-up period is normalized as u ∈ [0, 1] (0 = fastest admissible period): convergence is penalized by u²
  * (sleeping while the field changes) and energy by (1 − u)² (waking up while nothing changes).
  */
object TradeOff:

  /** Lower bound of the energy weight 1 − β, so that energy is never completely ignored, even for β = 1. */
  val MinEnergyWeight: Double = 0.005

  /** Residual convergence pressure of a node in steady state (instability intensity equal to zero). */
  val SteadyStateIntensity: Double = 0.06

  /** Gain of the instability intensity in the online reward (intensity 1 gives a convergence pressure of 1). */
  val OnlineIntensityGain: Double = 1.0 - SteadyStateIntensity

  /** Gain of the instability intensity in the counterfactual rewards used to train the offline learners. */
  val OfflineIntensityGain: Double = 8.0

  /** Fastest wake-up period used to normalize the periods when the action space does not define it. */
  val FastestPeriod: Double = 0.5

  /** The trade-off weights the offline learners are trained for on the ratio scale (the original objective). */
  val RatioWeights: Seq[Double] = Seq(0.5, 0.8, 0.95, 0.99, 1.0)

  /** The trade-off weights the offline learners are trained for on the linear scale: evenly spaced best periods. */
  val LinearWeights: Seq[Double] = Seq(0.2, 0.4, 0.6, 0.8, 1.0)

  /** The trade-off weights the offline learners are trained for on the full-range scale: with a steady-state intensity
    * of 1, the best period of a stable node goes from 4.1 s to 0.7 s.
    */
  val FullRangeWeights: Seq[Double] = Seq(0.2, 0.4, 0.6, 0.8, 0.95)

  /** Normalized weights of the two objectives; they sum to one. */
  final case class ObjectiveWeights(convergence: Double, energy: Double)

  /** How the counterfactual rewards of the offline learners turn the instability intensity into convergence pressure.
    * The normalized period that maximizes a reward is its energy weight: `steadyStateIntensity` sets the best period of
    * a stable node for each β (with the default, 2.6 s at β 0.95; with 0, the longest period at every β), and
    * `intensityGain` the best period of an unstable one (with the default, about the fastest one for every β ≥ 0.8). A
    * node without a value yet is waiting for the field to reach it: `missingValueIntensity` is its least intensity.
    */
  final case class RewardShape(
      steadyStateIntensity: Double = SteadyStateIntensity,
      intensityGain: Double = OfflineIntensityGain,
      missingValueIntensity: Double = 0.0,
      scale: BetaScale = BetaScale.Ratio
  ):
    /** The intensity that shapes the reward of a step, from the node's `instability` and whether it `hasValue`. */
    def intensity(instability: Double, hasValue: Boolean): Double =
      if hasValue then instability else math.max(instability, missingValueIntensity)

    /** How much convergence matters for a node of instability `intensity` on the linear scale, in [0, 1]. */
    def urgency(intensity: Double): Double = math.min(1.0, steadyStateIntensity + intensityGain * intensity)

  /** How β sets the weights of the two objectives of the offline learners, and how the networks see it. */
  enum BetaScale:
    /** The original objective: convergence β(S + G·I) against energy 1 − β, normalized. Its best period saturates: with
      * a large gain an unstable node asks for about the fastest period at every β ≥ 0.8, and a stable one for about the
      * longest up to β 0.95 with a small pressure, so β tunes little more than the stable nodes near 1.
      */
    case Ratio

    /** β is a linear dial on the urgency c of the node ([[RewardShape.urgency]]): convergence β·c against energy 1 −
      * β·c, so the best normalized period is 1 − β·c: the longest one at β = 0, the fastest one for an urgent node (c =
      * 1) at β = 1, evenly in between.
      */
    case Linear

    /** The weights of the ratio scale on betas spread over the whole range, fed to the networks linearly. With a
      * steady-state intensity S = 1 the best normalized period of a stable node is 1 − β, so β spans the fixed periods
      * from the longest (β = 0, only energy matters) to the fastest (β = 1, only convergence matters), and an unstable
      * node divides its odds of sleeping, u / (1 − u), by 1 + G·I.
      */
    case FullRange

    /** β as fed to the networks. */
    def feature(beta: Double): Float = this match
      case Ratio => betaFeature(beta)
      case Linear | FullRange => (2.0 * beta - 1.0).toFloat

    /** The trade-off weights every step is relabelled for, besides the learner's own one. */
    def counterfactualWeights: Seq[Double] = this match
      case Ratio => RatioWeights
      case Linear => LinearWeights
      case FullRange => FullRangeWeights

    /** `features` encoded with the original β feature at `index`, conditioned on `beta` with this scale's feature. */
    def conditioned(features: Array[Float], index: Int, beta: Double): Array[Float] = this match
      case Ratio => features
      case Linear | FullRange => features.updated(index, feature(beta))
  end BetaScale

  def energyWeight(beta: Double): Double = math.max(MinEnergyWeight, 1.0 - beta)

  /** β as fed to the networks: the steady-state share of the energy objective, mapped from [0, 1] to [−1, 1]. */
  def betaFeature(beta: Double): Float =
    val energy = energyWeight(beta)
    (2.0 * (energy / (energy + SteadyStateIntensity * beta)) - 1.0).toFloat

  /** The wake-up `period` mapped to [0, 1] within [`min`, `max`], clamped outside. */
  def normalizedPeriod(period: Double, min: Double, max: Double): Double =
    math.max(0.0, math.min(1.0, (period - min) / (max - min)))

  /** Normalized objective weights used by the offline learners, given the node's instability `intensity` ∈ [0, 1]. */
  def objectiveWeights(intensity: Double, beta: Double, shape: RewardShape = RewardShape()): ObjectiveWeights =
    shape.scale match
      case BetaScale.Ratio | BetaScale.FullRange => ratioWeights(intensity, beta, shape)
      case BetaScale.Linear =>
        val convergence = beta * shape.urgency(intensity)
        ObjectiveWeights(convergence, 1.0 - convergence)

  private def ratioWeights(intensity: Double, beta: Double, shape: RewardShape): ObjectiveWeights =
    val energy = energyWeight(beta)
    val convergence = beta * (shape.steadyStateIntensity + shape.intensityGain * intensity)
    val total = convergence + energy
    ObjectiveWeights(convergence / total, energy / total)

  /** The best period of an urgent node on the linear scale, within [`min`, `max`]: the reactive baselines tuned by β
    * fall back to it after a change.
    */
  def urgentPeriod(beta: Double, min: Double, max: Double): Double = min + (1.0 - beta) * (max - min)

  /** The period within [`min`, `max`] that maximizes the counterfactual reward of a node of instability `intensity`:
    * its normalized period is the energy weight.
    */
  def bestPeriod(intensity: Double, beta: Double, shape: RewardShape, min: Double, max: Double): Double =
    min + objectiveWeights(intensity, beta, shape).energy * (max - min)

  /** Counterfactual reward used to train the offline learners for any β, from logged (intensity, period) pairs. */
  def offlineReward(
      intensity: Double,
      period: Double,
      beta: Double,
      min: Double,
      max: Double,
      shape: RewardShape = RewardShape()
  ): Double =
    val u = normalizedPeriod(period, min, max)
    val weights = objectiveWeights(intensity, beta, shape)
    -weights.convergence * (u * u) - weights.energy * ((1.0 - u) * (1.0 - u))

  /** Reward computed by the nodes during the simulation (used by Q-Learning and reported as `reward`). */
  def onlineReward(intensity: Double, period: Double, beta: Double, min: Double, max: Double): Double =
    val u = normalizedPeriod(period, min, max)
    -beta * (SteadyStateIntensity + OnlineIntensityGain * intensity) * (u * u) -
      energyWeight(beta) * ((1.0 - u) * (1.0 - u))
end TradeOff
