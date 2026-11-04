package it.unibo.mobility

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MovingRegionSpec extends AnyFlatSpec with Matchers:
  private val line = (0 until 6).map(id => id -> (id * 10.0, 0.0)).toMap

  "A moving region" should "hold the share of the nodes closest to its centre" in {
    MovingRegion.select(line, (0.0, 0.0), 0.5) shouldBe Set(0, 1, 2)
    MovingRegion.select(line, (50.0, 0.0), 0.5) shouldBe Set(3, 4, 5)
  }

  it should "round the share up and break distance ties by node id" in {
    MovingRegion.select(line, (25.0, 0.0), 0.25) shouldBe Set(2, 3)
    MovingRegion.select(line, (25.0, 0.0), 1.0 / 6.0) shouldBe Set(2)
  }

  it should "hold every node with a share of one" in {
    MovingRegion.select(line, (25.0, 0.0), 1.0) shouldBe line.keySet
  }
end MovingRegionSpec
