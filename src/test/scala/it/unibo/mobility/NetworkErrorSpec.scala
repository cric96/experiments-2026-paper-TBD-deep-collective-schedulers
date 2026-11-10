package it.unibo.mobility

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NetworkErrorSpec extends AnyFlatSpec with Matchers:
  private val cap = 500.0

  "The weight of a node" should "fade by the spatial discount at every hop" in {
    NetworkError.weight(0.5, 2) shouldBe 0.25
    NetworkError.weight(1.0, 7) shouldBe 1.0
  }

  it should "be one for the node itself, also when the discount is zero" in {
    NetworkError.weight(0.0, 0) shouldBe 1.0
    NetworkError.weight(0.0, 1) shouldBe 0.0
  }

  "The error of a network" should "sum the weighted errors, without averaging them" in {
    NetworkError.of(Array(10.0, 20.0, 40.0), Array(1.0, 0.5, 0.25), cap) shouldBe 30.0
  }

  it should "count every error as at most the cap, an unbounded or undefined one as the cap" in {
    NetworkError.of(Array(2 * cap, Double.PositiveInfinity, Double.NaN), Array(1.0, 0.5, 0.25), cap) shouldBe
      1.75 * cap
  }

  it should "ignore the errors of the nodes with no weight" in {
    NetworkError.of(Array(0.0, Double.PositiveInfinity), Array(1.0, 0.0), cap) shouldBe 0.0
  }
end NetworkErrorSpec
