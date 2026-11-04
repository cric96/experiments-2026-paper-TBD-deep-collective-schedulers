package it.unibo.learning.agents

import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TrickleAgentSpec extends AnyFlatSpec with Matchers:
  private val agent = TrickleAgent(shortest = 0.5, longest = 5.0)

  "Trickle" should "fall back to the shortest period as soon as its own output changes" in {
    agent.policy(sourceMovingTowardsMe) shouldBe 0.5
    agent.policy(alone(lastPeriod = 4.0, 10.0, 12.0)) shouldBe 0.5
  }

  it should "double the period it slept after a round that left its output unchanged" in {
    agent.policy(alone(lastPeriod = 1.0, 10.0, 10.0)) shouldBe 2.0
    agent.policy(alone(lastPeriod = 2.0, 12.0, 10.0, 10.0)) shouldBe 4.0
  }

  it should "never sleep longer than the longest period" in {
    agent.policy(alone(lastPeriod = 4.0, 10.0, 10.0)) shouldBe 5.0
    agent.policy(alone(lastPeriod = 5.0, 10.0, 10.0)) shouldBe 5.0
  }

  it should "count a value that is still unknown as unchanged" in {
    agent.policy(alone(lastPeriod = 1.0, Double.PositiveInfinity, Double.PositiveInfinity)) shouldBe 2.0
    agent.policy(alone(lastPeriod = 1.0, Double.PositiveInfinity, 30.0)) shouldBe 0.5
  }

  it should "ignore the trade-off weight" in {
    fixtures.map(agent.policyForWeight(0.5)) shouldBe fixtures.map(agent.policy)
  }
end TrickleAgentSpec
