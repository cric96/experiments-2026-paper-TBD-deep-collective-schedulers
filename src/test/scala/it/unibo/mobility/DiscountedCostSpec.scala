package it.unibo.mobility

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DiscountedCostSpec extends AnyFlatSpec with Matchers:
  private val gamma = 0.9
  private val tolerance = 1e-12

  /** ∫_0^duration γ^s ds. */
  private def discounted(duration: Double): Double = (1.0 - math.pow(gamma, duration)) / math.log(1.0 / gamma)

  "A discounted cost" should "integrate a constant rate as c (1 − γ^Δ) / ln(1/γ)" in {
    DiscountedCost(gamma, 10.0, 0.3).advance(14.0).integral shouldBe (0.3 * discounted(4.0) +- tolerance)
  }

  it should "discount a later rate from the decision, not from when it began" in {
    val cost = DiscountedCost(gamma, 10.0, 0.3).observe(11.0, 2.0).advance(13.0)
    val expected = 0.3 * discounted(1.0) + 2.0 * (discounted(3.0) - discounted(1.0))
    cost.integral shouldBe (expected +- tolerance)
  }

  it should "not depend on how often the node is observed while the rate holds" in {
    val observed = (11 to 14).foldLeft(DiscountedCost(gamma, 10.0, 0.3))((cost, time) => cost.observe(time, 0.3))
    observed.integral shouldBe (DiscountedCost(gamma, 10.0, 0.3).advance(14.0).integral +- tolerance)
  }

  it should "be the plain integral without a discount" in {
    DiscountedCost(1.0, 0.0, 0.5).observe(2.0, 0.25).advance(6.0).integral shouldBe 2.0
  }

  it should "start a new decision from zero, holding the current rate" in {
    val restarted = DiscountedCost(gamma, 0.0, 0.3).observe(2.0, 0.7).restart(5.0)
    restarted shouldBe DiscountedCost(gamma, 5.0, 0.7)
    restarted.advance(6.0).integral shouldBe (0.7 * discounted(1.0) +- tolerance)
  }
end DiscountedCostSpec
