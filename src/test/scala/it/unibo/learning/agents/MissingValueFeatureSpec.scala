package it.unibo.learning.agents

import it.unibo.learning.TradeOff
import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The original learners accept the missing-value feature: their networks take the longer encoding. */
class MissingValueFeatureSpec extends AnyFlatSpec with Matchers:
  private def periods(learner: Learner): Seq[Double] = fixtures.map(learner.policy)

  "The original learners" should "choose admissible periods with the missing-value feature" in {
    val learners = Seq(
      OfflineLearnerGNN(0.5, 5.0, 1.0, 0.95, explorationEpisodes = 0, missingValueFeature = true),
      OfflineLearnerContinuous(0.5, 5.0, 1.0, 0.95, explorationEpisodes = 0, inspectionEnabled = false,
        missingValueFeature = true)
    )
    learners.flatMap(periods).foreach { period =>
      period should be >= 0.5
      period should be <= 5.0
    }
  }

  they should "choose admissible periods on the linear beta scale" in {
    val shape = TradeOff.RewardShape(steadyStateIntensity = 0.2, scale = TradeOff.BetaScale.Linear)
    val learners = Seq(
      OfflineLearnerGNN(0.5, 5.0, 1.0, 0.6, explorationEpisodes = 0, rewardShape = shape, missingValueFeature = true),
      OfflineLearnerContinuous(0.5, 5.0, 1.0, 0.6, explorationEpisodes = 0, inspectionEnabled = false,
        rewardShape = shape, missingValueFeature = true)
    )
    learners.flatMap(periods).foreach { period =>
      period should be >= 0.5
      period should be <= 5.0
    }
  }
end MissingValueFeatureSpec
