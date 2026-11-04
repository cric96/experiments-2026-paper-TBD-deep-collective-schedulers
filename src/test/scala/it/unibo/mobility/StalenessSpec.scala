package it.unibo.mobility

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class StalenessSpec extends AnyFlatSpec with Matchers:
  private val Unknown = Double.PositiveInfinity

  "Staleness" should "be zero for a node consistent with its neighbours" in {
    Staleness.of(held = 30.0, isSource = false, Seq((10.0, 20.0), (5.0, 40.0))) shouldBe 0.0
  }

  it should "measure the gap from the value a round would compute now" in {
    Staleness.of(held = 30.0, isSource = false, Seq((10.0, 26.0), (5.0, 40.0))) shouldBe 6.0
    Staleness.of(held = 30.0, isSource = false, Seq((10.0, 12.0))) shouldBe 8.0
  }

  it should "pin a source to zero, whatever its neighbours hold" in {
    Staleness.of(held = 0.0, isSource = true, Seq((10.0, 12.0))) shouldBe 0.0
    Staleness.of(held = 250.0, isSource = true, Seq.empty) shouldBe 250.0
  }

  it should "be unbounded when only one of the two values is defined" in {
    Staleness.of(held = Unknown, isSource = false, Seq((10.0, 12.0))) shouldBe Unknown
    Staleness.of(held = 30.0, isSource = false, Seq((Unknown, 12.0))) shouldBe Unknown
    Staleness.of(held = Unknown, isSource = true, Seq.empty) shouldBe Unknown
  }

  it should "be zero when neither the node nor its neighbours have a value" in {
    Staleness.of(held = Unknown, isSource = false, Seq((Unknown, 12.0))) shouldBe 0.0
    Staleness.of(held = Unknown, isSource = false, Seq.empty) shouldBe 0.0
  }
end StalenessSpec
