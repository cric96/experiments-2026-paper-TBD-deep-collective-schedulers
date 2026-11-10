package it.unibo.learning

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TradeOffSpec extends AnyFlatSpec with Matchers:
  private val tolerance = 1e-9
  private val fullRange =
    TradeOff.RewardShape(steadyStateIntensity = 1.0, scale = TradeOff.BetaScale.FullRange)

  "The energy weight" should "be 1 - beta, but never below its minimum" in {
    TradeOff.energyWeight(0.8) shouldBe 0.2 +- tolerance
    TradeOff.energyWeight(1.0) shouldBe TradeOff.MinEnergyWeight
  }

  "The beta feature" should "map the steady-state energy share to [-1, 1]" in {
    TradeOff.betaFeature(0.5).toDouble shouldBe (2 * 0.5 / (0.5 + 0.06 * 0.5) - 1) +- 1e-6
    TradeOff.betaFeature(1.0).toDouble shouldBe (2 * 0.005 / (0.005 + 0.06) - 1) +- 1e-6
  }

  it should "decrease as convergence becomes more important" in {
    val features = Seq(0.5, 0.8, 0.95, 0.99, 1.0).map(TradeOff.betaFeature)
    features shouldBe features.sorted.reverse
  }

  "The normalized period" should "map [min, max] to [0, 1] and clamp outside" in {
    TradeOff.normalizedPeriod(0.5, min = 0.5, max = 5.0) shouldBe 0.0
    TradeOff.normalizedPeriod(2.75, min = 0.5, max = 5.0) shouldBe 0.5 +- tolerance
    TradeOff.normalizedPeriod(9.0, min = 0.5, max = 5.0) shouldBe 1.0
    TradeOff.normalizedPeriod(0.1, min = 0.5, max = 5.0) shouldBe 0.0
  }

  "The objective weights" should "be normalized" in {
    val weights = TradeOff.objectiveWeights(intensity = 0.3, beta = 0.95)
    weights.convergence + weights.energy shouldBe 1.0 +- tolerance
  }

  it should "favour convergence when the node is unstable" in {
    val stable = TradeOff.objectiveWeights(intensity = 0.0, beta = 0.95)
    val unstable = TradeOff.objectiveWeights(intensity = 1.0, beta = 0.95)
    unstable.convergence should be > stable.convergence
  }

  "The offline reward" should "penalize slow rounds by the convergence weight and fast ones by the energy weight" in {
    val weights = TradeOff.objectiveWeights(intensity = 0.4, beta = 0.8)
    TradeOff.offlineReward(intensity = 0.4, period = 5.0, beta = 0.8, min = 0.5, max = 5.0) shouldBe
      -weights.convergence +- tolerance
    TradeOff.offlineReward(intensity = 0.4, period = 0.5, beta = 0.8, min = 0.5, max = 5.0) shouldBe
      -weights.energy +- tolerance
  }

  "The default reward shape" should "be the steady-state pressure and the offline gain of the original objective" in {
    val shape = TradeOff.RewardShape()
    shape.steadyStateIntensity shouldBe TradeOff.SteadyStateIntensity
    shape.intensityGain shouldBe TradeOff.OfflineIntensityGain
    TradeOff.objectiveWeights(intensity = 0.4, beta = 0.8, shape) shouldBe
      TradeOff.objectiveWeights(intensity = 0.4, beta = 0.8)
  }

  "The energy weight of the objectives" should "be the normalized period that maximizes the offline reward" in {
    val shape = TradeOff.RewardShape(steadyStateIntensity = 0.01, intensityGain = 1.0)
    for intensity <- Seq(0.0, 0.4, 1.0); beta <- Seq(0.5, 0.95, 1.0) do
      val best = 0.5 + 4.5 * TradeOff.objectiveWeights(intensity, beta, shape).energy
      def reward(period: Double): Double = TradeOff.offlineReward(intensity, period, beta, 0.5, 5.0, shape)
      reward(best) should be >= reward(math.max(0.5, best - 0.05))
      reward(best) should be >= reward(math.min(5.0, best + 0.05))
  }

  "A reward shape without steady-state pressure" should "make the longest period the best for a stable node only" in {
    val shape = TradeOff.RewardShape(steadyStateIntensity = 0.0)
    TradeOff.offlineReward(intensity = 0.0, period = 5.0, beta = 0.95, min = 0.5, max = 5.0, shape) shouldBe 0.0
    TradeOff.offlineReward(intensity = 1.0, period = 5.0, beta = 0.95, min = 0.5, max = 5.0, shape) should be < 0.0
  }

  "A smaller steady-state pressure" should "lengthen the best period of a stable node" in {
    def stableEnergy(steadyState: Double): Double =
      TradeOff
        .objectiveWeights(intensity = 0.0, beta = 0.95, TradeOff.RewardShape(steadyStateIntensity = steadyState))
        .energy
    stableEnergy(0.01) should be > stableEnergy(TradeOff.SteadyStateIntensity)
  }

  "A node without a value" should "be unstable at least by the missing-value intensity of the reward shape" in {
    val shape = TradeOff.RewardShape(missingValueIntensity = 0.8)
    shape.intensity(instability = 0.3, hasValue = false) shouldBe 0.8
    shape.intensity(instability = 1.0, hasValue = false) shouldBe 1.0
    shape.intensity(instability = 0.3, hasValue = true) shouldBe 0.3
    TradeOff.RewardShape().intensity(instability = 0.3, hasValue = false) shouldBe 0.3
  }

  "The default reward shape" should "use the ratio scale of the original objective" in {
    TradeOff.RewardShape().scale shouldBe TradeOff.BetaScale.Ratio
  }

  "The convergence surrogate" should "be the convergence pressure of the node times the squared normalized period" in {
    val shape = TradeOff.RewardShape(steadyStateIntensity = 1.0, intensityGain = 8.0)
    TradeOff.convergenceSurrogate(intensity = 0.5, period = 0.5, min = 0.5, max = 5.0, shape) shouldBe 0.0
    TradeOff.convergenceSurrogate(intensity = 0.5, period = 5.0, min = 0.5, max = 5.0, shape) shouldBe 5.0
    TradeOff.convergenceSurrogate(intensity = 0.0, period = 2.75, min = 0.5, max = 5.0, shape) shouldBe 0.25
  }

  "The urgency of a node" should "be the steady-state pressure plus the scaled instability, at most one" in {
    val shape = TradeOff.RewardShape(steadyStateIntensity = 0.2, intensityGain = 8.0)
    shape.urgency(0.0) shouldBe 0.2 +- tolerance
    shape.urgency(0.05) shouldBe 0.6 +- tolerance
    shape.urgency(0.5) shouldBe 1.0
  }

  "The linear beta scale" should "weight convergence by beta times the urgency, and energy by the rest" in {
    val shape = TradeOff.RewardShape(steadyStateIntensity = 0.2, scale = TradeOff.BetaScale.Linear)
    val weights = TradeOff.objectiveWeights(intensity = 0.0, beta = 0.6, shape)
    weights.convergence shouldBe 0.12 +- tolerance
    weights.energy shouldBe 0.88 +- tolerance
  }

  it should "make 1 - beta * urgency the normalized period that maximizes the offline reward" in {
    val shape = TradeOff.RewardShape(steadyStateIntensity = 0.2, scale = TradeOff.BetaScale.Linear)
    for intensity <- Seq(0.0, 0.05, 1.0); beta <- Seq(0.0, 0.2, 0.6, 1.0) do
      val best = 0.5 + 4.5 * (1.0 - beta * shape.urgency(intensity))
      def reward(period: Double): Double = TradeOff.offlineReward(intensity, period, beta, 0.5, 5.0, shape)
      reward(best) should be >= reward(math.max(0.5, best - 0.05))
      reward(best) should be >= reward(math.min(5.0, best + 0.05))
  }

  it should "spread the best period of an urgent node evenly between the longest and the fastest one" in {
    val shape = TradeOff.RewardShape(scale = TradeOff.BetaScale.Linear)
    val energies = TradeOff.BetaScale.Linear.counterfactualWeights.map(TradeOff.objectiveWeights(1.0, _, shape).energy)
    energies.zip(energies.tail).map((longer, shorter) => longer - shorter).foreach(_ shouldBe 0.2 +- tolerance)
    energies.last shouldBe 0.0 +- tolerance
  }

  it should "feed beta to the networks mapped linearly from [0, 1] to [-1, 1]" in {
    TradeOff.BetaScale.Linear.feature(0.0) shouldBe -1.0f
    TradeOff.BetaScale.Linear.feature(0.6).toDouble shouldBe 0.2 +- 1e-6
    TradeOff.BetaScale.Linear.feature(1.0) shouldBe 1.0f
  }

  "The ratio beta scale" should "feed the original beta feature to the networks" in
    Seq(0.5, 0.95, 1.0).foreach(beta => TradeOff.BetaScale.Ratio.feature(beta) shouldBe TradeOff.betaFeature(beta))

  it should "condition encoded features only on the linear scale" in {
    val features = Array(0.1f, TradeOff.betaFeature(0.6), 0.3f)
    TradeOff.BetaScale.Ratio.conditioned(features, 1, 0.6) should be theSameInstanceAs features
    TradeOff.BetaScale.Linear.conditioned(features, 1, 0.6).toSeq shouldBe
      Seq(0.1f, TradeOff.BetaScale.Linear.feature(0.6), 0.3f)
    features(1) shouldBe TradeOff.betaFeature(0.6)
  }

  "The full-range beta scale" should "make 1 - beta the best normalized period of a stable node" in {
    for beta <- Seq(0.0, 0.2, 0.6, 0.95) do
      TradeOff.bestPeriod(0.0, beta, fullRange, 0.5, 5.0) shouldBe (0.5 + 4.5 * (1.0 - beta)) +- tolerance
  }

  it should "divide the odds of sleeping of an unstable node by 1 + G * I" in {
    val energy = TradeOff.objectiveWeights(1.0, 0.6, fullRange).energy
    energy / (1.0 - energy) shouldBe (0.4 / 0.6 / (1.0 + TradeOff.OfflineIntensityGain)) +- tolerance
  }

  it should "be trained for betas whose stable nodes span the fixed periods from about 4 s to below 1 s" in {
    val stablePeriods = TradeOff.FullRangeWeights.map(TradeOff.bestPeriod(0.0, _, fullRange, 0.5, 5.0))
    TradeOff.BetaScale.FullRange.counterfactualWeights shouldBe TradeOff.FullRangeWeights
    stablePeriods.head should be > 4.0
    stablePeriods.last should be < 1.0
  }

  it should "feed beta to the networks linearly, as the linear scale" in {
    val features = Array(0.1f, TradeOff.betaFeature(0.6), 0.3f)
    TradeOff.BetaScale.FullRange.feature(0.6) shouldBe TradeOff.BetaScale.Linear.feature(0.6)
    TradeOff.BetaScale.FullRange.conditioned(features, 1, 0.6).toSeq shouldBe
      Seq(0.1f, TradeOff.BetaScale.Linear.feature(0.6), 0.3f)
  }

  "The best period" should "maximize the offline reward on every scale" in {
    for
      scale <- TradeOff.BetaScale.values
      intensity <- Seq(0.0, 0.3, 1.0)
      beta <- Seq(0.2, 0.8, 0.95)
    do
      val shape = TradeOff.RewardShape(steadyStateIntensity = 0.2, scale = scale)
      val best = TradeOff.bestPeriod(intensity, beta, shape, 0.5, 5.0)
      def reward(period: Double): Double = TradeOff.offlineReward(intensity, period, beta, 0.5, 5.0, shape)
      reward(best) should be >= reward(math.max(0.5, best - 0.05))
      reward(best) should be >= reward(math.min(5.0, best + 0.05))
  }

  "The urgent period" should "be the best period of an urgent node on the linear scale" in {
    val shape = TradeOff.RewardShape(scale = TradeOff.BetaScale.Linear)
    for beta <- Seq(0.0, 0.2, 0.8, 1.0) do
      TradeOff.urgentPeriod(beta, 0.5, 5.0) shouldBe
        (0.5 + 4.5 * TradeOff.objectiveWeights(1.0, beta, shape).energy) +- tolerance
    TradeOff.urgentPeriod(1.0, 0.5, 5.0) shouldBe 0.5 +- tolerance
    TradeOff.urgentPeriod(0.0, 0.5, 5.0) shouldBe 5.0 +- tolerance
  }

  "The online reward" should "weight the convergence penalty by beta and the instability intensity" in {
    val u = TradeOff.normalizedPeriod(2.0, min = 0.5, max = 5.0)
    val expected = -0.9 * (0.06 + 0.94 * 0.5) * u * u - 0.1 * (1 - u) * (1 - u)
    TradeOff.onlineReward(intensity = 0.5, period = 2.0, beta = 0.9, min = 0.5, max = 5.0) shouldBe
      expected +- tolerance
  }
end TradeOffSpec
