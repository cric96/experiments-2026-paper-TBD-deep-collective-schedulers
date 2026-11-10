package it.unibo.learning.agents

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{CriticContext, MobileOutcome, Step, Stratification, Trajectory}
import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** The mobile learners with critics that also see the context of each decision measured on the ground truth. */
class PrivilegedCriticSpec extends AnyFlatSpec with Matchers:
  private val shape = TradeOff.RewardShape(scale = TradeOff.BetaScale.FullRange)
  private val costScale = 4.0

  /** A trajectory through the fixtures, every decision with its own context and a network cost. */
  private val trajectory: Trajectory =
    fixtures.zip(fixtures.drop(1) :+ fixtures.last).zipWithIndex.map { case ((state, next), index) =>
      val context = CriticContext(0.1 * index, 0.2 * index, 0.3, 0.05 * index, 0.5)
      val outcome = MobileOutcome(
        1.0 + index,
        Map(0.0 -> 0.1 * index),
        next,
        terminal = index == fixtures.size - 1,
        context = context,
        nextContext = CriticContext(0.1 * (index + 1), 0.2 * (index + 1), 0.3),
        costScale = costScale,
        stratification = Stratification.ByLocalRegime
      )
      Step(state, actionIndex = -1, action = 0.5 + index, reward = 0.0, Some(outcome))
    }

  /** The two learners, which explore during the first `exploration` episodes. */
  private def learners(privileged: Boolean, exploration: Int = 1, diagnosed: Boolean = false): Seq[Learner] = Seq(
    OfflineLearnerGNN(0.5, 5.0, 1.0, 0.8, explorationEpisodes = exploration, stepsPerEpisode = 4, mobile = true,
      mobileEncoding = false, rewardShape = shape, missingValueFeature = true, privilegedCritic = privileged,
      diagnosed = diagnosed),
    OfflineLearnerContinuous(0.5, 5.0, 1.0, 0.8, explorationEpisodes = exploration, stepsPerEpisode = 4, mobile = true,
      mobileEncoding = false, rewardShape = shape, missingValueFeature = true, privilegedCritic = privileged,
      diagnosed = diagnosed)
  )

  "A privileged critic" should "train on the contexts of the decisions and leave an admissible policy" in
    learners(privileged = true).foreach { learner =>
      learner.endEpisode(Seq(trajectory, trajectory))
      fixtures.map(learner.policy).foreach(period => period should (be >= 0.5 and be <= 5.0))
    }

  it should "leave the actor as it is: the same policy before training as without privileges" in
    learners(privileged = true, exploration = 0).zip(learners(privileged = false, exploration = 0)).foreach {
      (privileged, plain) =>
        fixtures.map(privileged.policy) shouldBe fixtures.map(plain.policy)
    }

  it should "train on the rich context too" in {
    val learners = Seq(
      OfflineLearnerGNN(0.5, 5.0, 1.0, 0.8, explorationEpisodes = 1, stepsPerEpisode = 4, mobile = true,
        mobileEncoding = false, rewardShape = shape, missingValueFeature = true, privilegedCritic = true,
        richContext = true),
      OfflineLearnerContinuous(0.5, 5.0, 1.0, 0.8, explorationEpisodes = 1, stepsPerEpisode = 4, mobile = true,
        mobileEncoding = false, rewardShape = shape, missingValueFeature = true, privilegedCritic = true,
        richContext = true)
    )
    learners.foreach { learner =>
      learner.endEpisode(Seq(trajectory))
      fixtures.map(learner.policy).foreach(period => period should (be >= 0.5 and be <= 5.0))
    }
    an[IllegalArgumentException] should be thrownBy
      OfflineLearnerGNN(0.5, 5.0, 1.0, 0.8, mobile = true, richContext = true)
  }

  it should "need the measured contexts of the mobile experiment" in {
    an[IllegalArgumentException] should be thrownBy OfflineLearnerGNN(0.5, 5.0, 1.0, 0.8, privilegedCritic = true)
    an[IllegalArgumentException] should be thrownBy
      OfflineLearnerContinuous(0.5, 5.0, 1.0, 0.8, privilegedCritic = true)
  }

  "A diagnosed learner" should "diagnose every trade-off it was relabelled for, in both regimes" in {
    val weights = (0.8 +: shape.scale.counterfactualWeights).distinct.sorted
    learners(privileged = true, diagnosed = true).foreach { learner =>
      learner.endEpisode(Seq(trajectory))
      val diagnostics = learner.trainingDiagnostics
      diagnostics.filter(_.unstable).map(_.beta) shouldBe weights
      diagnostics.filterNot(_.unstable).map(_.beta) shouldBe weights
      diagnostics.foreach { diagnostic =>
        diagnostic.criticLoss.isFinite shouldBe true
        diagnostic.actorPeriod should (be >= 0.5 and be <= 5.0)
        diagnostic.greedyPeriod should (be >= 0.5 and be <= 5.0)
      }
    }
    learners(privileged = true).flatMap(_.trainingDiagnostics) shouldBe empty
  }
end PrivilegedCriticSpec
