package it.unibo.learning.encoding

import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class StarGraphSpec extends AnyFlatSpec with Matchers:
  private def padding(slots: Int): Seq[Float] = Seq.fill(slots)(StarGraph.IdleSlot).flatten

  "The star-graph encoding" should "keep the ego features and one slot per neighbour, nearest first" in {
    val beta = 0.61290324f
    fixtures.map(StarGraph.encode(_, 0.8, 5.0).toSeq) shouldBe Seq(
      Seq(-1.0f, 0.0f, 1.0f, 1.0f, beta) ++ Seq(0.0f, 0.0f, 1.0f, 0.9999092f, 0.1f, 0.6f) ++ padding(7),
      Seq(0.0f, 0.0f, 0.0f, 0.0f, beta) ++ Seq(0.0f, 0.0f, 0.0f, -0.9999092f, 0.1f, 0.2f) ++
        Seq(-1.0f, 0.99999976f, 0.99999976f, -0.9950548f, 0.1f, 0.8f) ++ padding(6),
      Seq(0.0f, 0.0f, 0.0f, 0.5f, beta) ++ Seq(0.0f, 0.0f, 0.0f, -0.9866143f, 0.1f, 0.2f) ++
        Seq(0.0f, 0.0f, 0.0f, 0.9866143f, 0.1f, 0.2f) ++ padding(6),
      Seq(0.0f, 0.0f, 0.0f, 0.0f, beta) ++ padding(8)
    )
  }

  it should "have the declared size" in {
    fixtures.map(StarGraph.encode(_, 0.8, 5.0).length).distinct shouldBe Seq(StarGraph.TotalDim)
  }

  it should "insert the missing-value feature after the ego features when asked" in
    fixtures.foreach { state =>
      val plain = StarGraph.encode(state, 0.8, 5.0)
      val flagged = StarGraph.encode(state, 0.8, 5.0, missingValueFeature = true)
      flagged.toSeq shouldBe
        ((plain.take(StarGraph.SelfDim) :+ MissingValue.feature(state)) ++ plain.drop(StarGraph.SelfDim)).toSeq
    }

  "The nearest neighbours" should "exclude the node itself and be sorted by distance" in {
    stableBackbone.latestSnapshot.map(StarGraph.nearestNeighbours(stableBackbone, _).map(_._1)) shouldBe
      Some(List(0, 2))
  }
end StarGraphSpec
