package it.unibo.learning.agents

import it.unibo.learning.TradeOff
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BaselineTuningSpec extends AnyFlatSpec with Matchers:
  private val fullRange =
    TradeOff.RewardShape(steadyStateIntensity = 1.0, scale = TradeOff.BetaScale.FullRange)

  "Untuned baselines" should "keep the whole range of periods whatever beta" in {
    BaselineTuning.Untuned.periods(0.2, fullRange, 0.5, 5.0) shouldBe ReactivePeriods(0.5, 5.0)
  }

  "Baselines tuned by the urgent period" should "fall back to it after a change and sleep up to the longest one" in {
    BaselineTuning.UrgentPeriod.periods(0.2, fullRange, 0.5, 5.0) shouldBe
      ReactivePeriods(TradeOff.urgentPeriod(0.2, 0.5, 5.0), 5.0)
  }

  "Baselines tuned by the best periods" should "move between those of an unstable and of a stable node" in {
    val periods = BaselineTuning.BestPeriods.periods(0.2, fullRange, 0.5, 5.0)
    periods shouldBe ReactivePeriods(
      TradeOff.bestPeriod(1.0, 0.2, fullRange, 0.5, 5.0),
      TradeOff.bestPeriod(0.0, 0.2, fullRange, 0.5, 5.0)
    )
    periods.longest shouldBe 4.1 +- 1e-9
    periods.shortest should be < periods.longest
  }

  it should "run about at the fastest period when only convergence matters" in {
    val periods = BaselineTuning.BestPeriods.periods(1.0, fullRange, 0.5, 5.0)
    periods.shortest shouldBe 0.5 +- 0.05
    periods.longest shouldBe 0.5 +- 0.05
  }

  "The tunings" should "keep the ordinals the simulation files select them by" in {
    BaselineTuning.values.toSeq shouldBe
      Seq(BaselineTuning.Untuned, BaselineTuning.UrgentPeriod, BaselineTuning.BestPeriods)
  }
end BaselineTuningSpec
