package it.unibo.mobility

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Random

class ErrorIntegralSpec extends AnyFlatSpec with Matchers:
  private val Tolerances = Seq(0.0, 5.0, 15.0)
  private val Scale = 50.0

  /** The original map-based accumulation: the integral must reproduce it bit for bit. */
  final private case class Reference(
      time: Double = 0.0,
      error: Double = Double.PositiveInfinity,
      reachable: Boolean = false,
      costs: Map[Double, Double] = Tolerances.map(_ -> 0.0).toMap,
      within: Map[Double, Double] = Tolerances.map(_ -> 0.0).toMap,
      reachableDuration: Double = 0.0,
      raw: Double = 0.0,
      absolute: Double = 0.0
  ):
    def advance(now: Double): Reference =
      val duration = now - time
      copy(
        time = now,
        costs = costs.map((tolerance, cost) =>
          tolerance -> (cost + MobileObjective.loss(error, tolerance, Scale) * duration)
        ),
        within =
          within.map((tolerance, elapsed) => tolerance -> (elapsed + (if error <= tolerance then duration else 0.0))),
        absolute = absolute + (if error.isFinite then error else Scale) * duration,
        raw = raw + MobileObjective.loss(error, 0.0, Scale) * duration,
        reachableDuration = if reachable then reachableDuration + duration else reachableDuration
      )

    def observe(now: Double, currentError: Double, isReachable: Boolean): Reference =
      advance(now).copy(error = currentError, reachable = isReachable)
  end Reference

  /** Simulator-like events: repeated instants, unbounded errors and unreachable nodes. */
  private def events(random: Random): Seq[(Double, Double, Boolean)] =
    Iterator
      .iterate(0.0)(time => if random.nextDouble() < 0.4 then time else time + random.nextDouble() * 2.0)
      .take(400)
      .map { time =>
        val error = if random.nextDouble() < 0.1 then Double.PositiveInfinity else random.nextDouble() * 80.0
        (time, error, random.nextDouble() < 0.8)
      }
      .toSeq

  "An error integral" should "accumulate exactly like the original per-tolerance accumulation" in {
    val sequence = events(new Random(7))
    val integral = ErrorIntegral(Tolerances, Scale)
    sequence.foreach((time, error, reachable) => integral.observe(time, error, reachable))
    val end = sequence.last._1 + 1.5
    integral.advance(end)
    val reference = sequence
      .foldLeft(Reference())((model, event) => model.observe(event._1, event._2, event._3))
      .advance(end)
    Tolerances.foreach { tolerance =>
      integral.cost(tolerance) shouldBe reference.costs(tolerance)
      integral.withinTime(tolerance) shouldBe reference.within(tolerance)
    }
    integral.reachableTime shouldBe reference.reachableDuration
    integral.rawIntegral shouldBe reference.raw
    integral.absoluteIntegral shouldBe reference.absolute
  }

  it should "reject a clock that moves backwards" in {
    val integral = ErrorIntegral(Tolerances, Scale)
    integral.advance(2.0)
    an[IllegalArgumentException] should be thrownBy integral.advance(1.0)
  }
end ErrorIntegralSpec
