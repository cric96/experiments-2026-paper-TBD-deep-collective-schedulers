package it.unibo.learning.agents.offline

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PeriodExplorationSpec extends AnyFlatSpec with Matchers:
  "Log-uniform exploration" should "cover the existing interval with a geometric median" in {
    PeriodExploration.LogUniform.period(0.5, 5.0, 0.5) shouldBe math.sqrt(2.5) +- 1e-12
    for strategy <- PeriodExploration.values; index <- 0 to 100 do
      strategy.period(0.5, 5.0, index / 100.0) should (be >= (0.5 - 1e-12) and be <= (5.0 + 1e-12))
  }
