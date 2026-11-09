package it.unibo.mobility

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NeighbourhoodErrorSpec extends AnyFlatSpec with Matchers:
  private val cap = 500.0

  "The error of a neighbourhood" should "be the mean of the node's own error and of its neighbours' ones" in {
    NeighbourhoodError.of(0.0, Seq(30.0, 60.0), cap) shouldBe 30.0
  }

  it should "charge a node that holds a correct value while a neighbour still lacks one" in {
    NeighbourhoodError.of(0.0, Seq(0.0, Double.PositiveInfinity), cap) shouldBe cap / 3.0
  }

  it should "count every error as at most the cap, so that it is the mean of the losses without tolerance" in {
    NeighbourhoodError.of(2 * cap, Seq(Double.PositiveInfinity), cap) shouldBe cap
  }

  it should "be the node's own error when it has no neighbours" in {
    NeighbourhoodError.of(42.0, Seq.empty, cap) shouldBe 42.0
  }
end NeighbourhoodErrorSpec
