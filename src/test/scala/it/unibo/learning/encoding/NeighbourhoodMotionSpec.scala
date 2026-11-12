package it.unibo.learning.encoding

import it.unibo.learning.abstractions.AgentState
import it.unibo.testing.AgentStates
import it.unibo.testing.AgentStates.{fixtures, isolated, Reading}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NeighbourhoodMotionSpec extends AnyFlatSpec with Matchers:
  /** Node 1 between neighbours 0 and 2, of which only those in `moving` move (node 1 at `ownSpeed`). */
  private def line(ownSpeed: Double, moving: Int*): AgentState =
    AgentStates
      .of(
        me = 1,
        Map(
          0 -> Reading(0.0, distance = 10.0, speed = if moving.contains(0) then 1.0 else 0.0),
          1 -> Reading(10.0),
          2 -> Reading(20.0, distance = 10.0, speed = if moving.contains(2) then 1.0 else 0.0)
        )
      )
      .copy(speed = ownSpeed)

  "The motion of a neighbourhood" should "be the share of the node and of its neighbours that move" in {
    NeighbourhoodMotion.share(line(0.0)) shouldBe 0.0
    NeighbourhoodMotion.share(line(1.0)) shouldBe (1.0 / 3.0 +- 1e-12)
    NeighbourhoodMotion.share(line(0.0, 0, 2)) shouldBe (2.0 / 3.0 +- 1e-12)
    NeighbourhoodMotion.share(line(2.0, 0, 2)) shouldBe 1.0
  }

  it should "be the node's own motion when it has seen no neighbour" in {
    NeighbourhoodMotion.share(isolated) shouldBe 0.0
    NeighbourhoodMotion.share(isolated.copy(speed = 1.0)) shouldBe 1.0
  }

  it should "be zero outside the mobile experiment" in {
    fixtures.map(NeighbourhoodMotion.share) shouldBe fixtures.map(_ => 0.0)
  }
end NeighbourhoodMotionSpec
