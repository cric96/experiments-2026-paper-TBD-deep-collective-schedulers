package it.unibo.learning.agents.offline

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{AgentState, MobileOutcome, Step}
import it.unibo.learning.agents.offline.SurrogateWeights.constant
import it.unibo.learning.encoding.InstabilityIntensity
import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class HybridConvergenceSpec extends AnyFlatSpec with Matchers:
  private val shape = TradeOff.RewardShape(steadyStateIntensity = 1.0, intensityGain = 8.0, missingValueIntensity = 1.0)

  /** The trade-off of the steps, when the weights do not depend on it. */
  private val beta = 0.8

  /** A mobile step from `state` that slept `period` seconds and paid a measured cost of 0.3. */
  private def step(state: AgentState, period: Double): Step =
    Step(state, -1, period, 0.0, Some(MobileOutcome(period, Map(0.0 -> 0.3), stableBackbone, terminal = false)))

  "The hybrid convergence cost" should "be the measured cost alone without a surrogate weight" in {
    HybridConvergence(constant(0.0), shape, 0.5, 5.0).cost(step(sourceMovingTowardsMe, 5.0), 0.3, beta) shouldBe 0.3
  }

  it should "add the weighted surrogate of the node's instability and period" in {
    val hybrid = HybridConvergence(constant(0.5), shape, 0.5, 5.0)
    val instability = InstabilityIntensity.of(neighbourBecameUnreachable)
    hybrid.cost(step(neighbourBecameUnreachable, 2.75), 0.3, beta) shouldBe
      0.3 + 0.5 * TradeOff.convergenceSurrogate(instability, 2.75, 0.5, 5.0, shape)
    hybrid.cost(step(stableBackbone, 5.0), 0.3, beta) shouldBe 0.3 + 0.5 * shape.steadyStateIntensity
  }

  it should "count a node without a value as unstable by the missing-value intensity" in {
    HybridConvergence(constant(1.0), shape, 0.5, 5.0).cost(step(isolated, 5.0), 0.0, beta) shouldBe
      shape.steadyStateIntensity + shape.intensityGain * shape.missingValueIntensity
  }

  it should "weigh the surrogate by the weight of the beta of the step" in {
    val scheduled = HybridConvergence(SurrogateWeights(0.2, 2.0, 0.2, 0.95), shape, 0.5, 5.0)
    scheduled.cost(step(stableBackbone, 5.0), 0.3, 0.2) shouldBe (0.3 + 0.2 * shape.steadyStateIntensity +- 1e-12)
    scheduled.cost(step(stableBackbone, 5.0), 0.3, 0.95) shouldBe (0.3 + 2.0 * shape.steadyStateIntensity +- 1e-12)
  }
end HybridConvergenceSpec
