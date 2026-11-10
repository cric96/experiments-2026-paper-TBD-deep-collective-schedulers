package it.unibo.learning.agents.offline

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SurrogateWeightsSpec extends AnyFlatSpec with Matchers:
  private val scheduled = SurrogateWeights(atLowest = 0.2, atHighest = 2.0, lowestBeta = 0.2, highestBeta = 0.95)

  "Constant surrogate weights" should "be the same at every beta, zero included" in {
    Seq(0.0, 0.2, 0.6, 1.0).map(SurrogateWeights.constant(0.5).at) shouldBe Seq.fill(4)(0.5)
    Seq(0.0, 0.6, 1.0).map(SurrogateWeights.constant(0.0).at) shouldBe Seq.fill(3)(0.0)
  }

  "Scheduled surrogate weights" should "grow geometrically from the lowest beta to the highest one" in {
    scheduled.at(0.2) shouldBe 0.2 +- 1e-12
    scheduled.at(0.95) shouldBe 2.0 +- 1e-12
    scheduled.at((0.2 + 0.95) / 2) shouldBe math.sqrt(0.2 * 2.0) +- 1e-12
  }

  they should "stay at their ends outside the range of the betas" in {
    scheduled.at(0.0) shouldBe scheduled.at(0.2)
    scheduled.at(1.0) shouldBe scheduled.at(0.95)
  }

  they should "be positive and span an ordered range of betas" in {
    an[IllegalArgumentException] should be thrownBy SurrogateWeights(0.0, 2.0, 0.2, 0.95)
    an[IllegalArgumentException] should be thrownBy SurrogateWeights(0.2, 2.0, 0.95, 0.2)
    an[IllegalArgumentException] should be thrownBy SurrogateWeights.constant(-0.1)
  }
end SurrogateWeightsSpec
