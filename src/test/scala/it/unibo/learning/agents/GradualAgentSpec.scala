package it.unibo.learning.agents

import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GradualAgentSpec extends AnyFlatSpec with Matchers:
  private val agent = GradualAgent(shortest = 0.5, longest = 5.0)

  "The gradual controller" should "halve the period it slept after a round that changed its own output" in {
    agent.policy(alone(lastPeriod = 4.0, 10.0, 12.0)) shouldBe 2.0
    agent.policy(sourceMovingTowardsMe) shouldBe 0.5
  }

  it should "never wake up sooner than the shortest period" in {
    agent.policy(alone(lastPeriod = 0.75, 10.0, 12.0)) shouldBe 0.5
  }

  it should "lengthen the period by one step after a round that left its output unchanged" in {
    agent.policy(alone(lastPeriod = 2.0, 10.0, 10.0)) shouldBe 2.5
    agent.policy(alone(lastPeriod = 0.5, 12.0, 10.0, 10.0)) shouldBe 1.0
  }

  it should "never sleep longer than the longest period" in {
    agent.policy(alone(lastPeriod = 4.75, 10.0, 10.0)) shouldBe 5.0
  }

  it should "ignore the trade-off weight" in {
    fixtures.map(agent.policyForWeight(0.5)) shouldBe fixtures.map(agent.policy)
  }
end GradualAgentSpec
