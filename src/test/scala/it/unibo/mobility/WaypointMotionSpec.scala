package it.unibo.mobility

import org.apache.commons.math3.random.Well19937c
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class WaypointMotionSpec extends AnyFlatSpec with Matchers:
  private val Width = 500.0
  private val Interval = 0.25
  private val Nodes = 0 until 50

  /** The positions of `Nodes` (all starting from the centre) after each of `ticks` movement intervals. */
  private def trace(seed: Int, speed: Double, ticks: Int): Vector[Map[Int, (Double, Double)]] =
    val motion = WaypointMotion(new Well19937c(seed), Width, speed, Interval)
    val start = Nodes.map(_ -> (Width / 2, Width / 2)).toMap
    Iterator
      .iterate(start)(positions => positions.map { case (id, (x, y)) => id -> motion.move(id, x, y) })
      .slice(1, ticks + 1)
      .toVector

  "Waypoint motion" should "be reproducible for a seed" in {
    trace(7, 1.5, 400) shouldBe trace(7, 1.5, 400)
    trace(7, 1.5, 400) should not be trace(8, 1.5, 400)
  }

  it should "never exceed the speed nor leave the arena" in {
    val positions = trace(3, 2.0, 2000)
    positions.flatMap(_.values).foreach { case (x, y) =>
      x should (be >= 0.0 and be <= Width)
      y should (be >= 0.0 and be <= Width)
    }
    positions.sliding(2).foreach { case Seq(before, after) =>
      Nodes.foreach { id =>
        val ((x0, y0), (x1, y1)) = (before(id), after(id))
        math.hypot(x1 - x0, y1 - y0) should be <= 2.0 * Interval + 1e-9
      }
    }
  }

  it should "let some nodes rest while others move" in {
    val positions = trace(5, 1.0, 400)
    val moving = positions.sliding(2).map { case Seq(before, after) => Nodes.count(id => before(id) != after(id)) }
    moving.foreach { count =>
      count should be > 0
      count should be < Nodes.size
    }
  }

  it should "keep every node still at zero speed" in
    trace(5, 0.0, 100).foreach(_.values.toSet shouldBe Set((Width / 2, Width / 2)))
end WaypointMotionSpec
