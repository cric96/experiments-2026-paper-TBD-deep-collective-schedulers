package it.unibo.learning.encoding

import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SpatialTemporalSpec extends AnyFlatSpec with Matchers:
  "The spatial-temporal encoding" should "keep its features" in {
    val beta = -0.06542056f
    fixtures.map(SpatialTemporal.encode(_, 0.95, 5.0).toSeq) shouldBe Seq(
      Seq(-1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.6f, beta),
      Seq(0.0f, 0.0f, 0.0f, 0.9640276f, 0.5f, 0.0f, 0.0f, 0.2f, beta),
      Seq(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.2f, beta),
      Seq(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, beta)
    )
  }

  it should "have the declared size" in {
    fixtures.map(SpatialTemporal.encode(_, 0.95, 5.0).length).distinct shouldBe Seq(SpatialTemporal.FeatureDim)
  }

  it should "append the missing-value feature when asked" in
    fixtures.foreach { state =>
      SpatialTemporal.encode(state, 0.95, 5.0, missingValueFeature = true).toSeq shouldBe
        (SpatialTemporal.encode(state, 0.95, 5.0) :+ MissingValue.feature(state)).toSeq
    }
