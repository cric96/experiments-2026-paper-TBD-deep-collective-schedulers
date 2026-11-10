package it.unibo.mobility

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SpatialWeightsSpec extends AnyFlatSpec with Matchers:
  private val cap = 500.0

  /** A chain 1 - 2 - 3 - 4 and an isolated node 5. */
  private val chain: Map[Int, Seq[Int]] = Map(1 -> Seq(2), 2 -> Seq(1, 3), 3 -> Seq(2, 4), 4 -> Seq(3), 5 -> Seq.empty)

  /** Only the last node of the chain holds a wrong value. */
  private val errors = Map(1 -> 0.0, 2 -> 0.0, 3 -> 0.0, 4 -> 80.0, 5 -> 0.0)

  "The hop distances" should "count the hops of the shortest paths from a node, and skip the nodes no path reaches" in {
    SpatialWeights.hops(chain, 1) shouldBe Map(1 -> 0, 2 -> 1, 3 -> 2, 4 -> 3)
    SpatialWeights.hops(chain, 5) shouldBe Map(5 -> 0)
  }

  "The cost of a node" should "charge the errors of the network, discounted at every hop" in {
    val costs = SpatialWeights(chain, 0.5).costs(errors, cap)
    costs(1) shouldBe 10.0
    costs(3) shouldBe 40.0
    costs(4) shouldBe 80.0
    costs(5) shouldBe 0.0
  }

  it should "be the node's own error without a spatial discount" in {
    SpatialWeights(chain, 0.0).costs(errors, cap) shouldBe errors
  }

  it should "charge every node of its component alike with a unit discount" in {
    val costs = SpatialWeights(chain, 1.0).costs(errors + (5 -> Double.PositiveInfinity), cap)
    (1 to 4).map(costs) shouldBe Seq.fill(4)(80.0)
    costs(5) shouldBe cap
  }

  "The effective size of a network" should "be the mean total weight of a node" in {
    val nodeWeights = Seq(1.875, 2.25, 2.25, 1.875, 1.0)
    SpatialWeights(chain, 0.5).effectiveSize shouldBe nodeWeights.sum / nodeWeights.size
    SpatialWeights(chain, 0.0).effectiveSize shouldBe 1.0
  }

  "The nodes reached by a change" should "be those whose cost weighs a changed node" in {
    SpatialWeights(chain, 0.5).reachedFrom(Set(4)) shouldBe Set(1, 2, 3, 4)
    SpatialWeights(chain, 0.0).reachedFrom(Set(4)) shouldBe Set(4)
    SpatialWeights(chain, 1.0).reachedFrom(Set(5)) shouldBe Set(5)
  }
end SpatialWeightsSpec
