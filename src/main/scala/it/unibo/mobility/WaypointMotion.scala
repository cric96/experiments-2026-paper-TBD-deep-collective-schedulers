package it.unibo.mobility

import it.unibo.mobility.WaypointMotion.*
import org.apache.commons.math3.random.RandomGenerator

/** Random waypoint with pauses, around the current position: every node alternates exponential pauses and straight
  * trips at `speed` (m/s) towards a uniform point within `Radius`, so at any time some nodes move while others rest.
  * Nodes start in the long-run mix of resting and moving ones.
  */
final class WaypointMotion(random: RandomGenerator, width: Double, speed: Double, interval: Double) extends NodeMotion:
  require(width > 0.0 && speed >= 0.0 && speed.isFinite && interval > 0.0)
  private var plans: Map[Int, Plan] = Map.empty

  override def move(id: Int, x: Double, y: Double): (Double, Double) =
    if speed == 0.0 then (x, y)
    else
      val (position, next) = advance(plans.getOrElse(id, initialPlan(x, y)), x, y)
      plans += id -> next
      position

  private def advance(plan: Plan, x: Double, y: Double): ((Double, Double), Plan) = plan match
    case Plan.Resting(remaining) =>
      ((x, y), if remaining > interval then Plan.Resting(remaining - interval) else trip(x, y))
    case Plan.Heading(targetX, targetY) =>
      val distance = math.hypot(targetX - x, targetY - y)
      val step = speed * interval
      if distance <= step then ((targetX, targetY), pause())
      else ((x + (targetX - x) * step / distance, y + (targetY - y) * step / distance), plan)

  private def initialPlan(x: Double, y: Double): Plan =
    val tripDuration = MeanTripLength / speed
    if random.nextDouble() < tripDuration / (tripDuration + MeanPause) then trip(x, y) else pause()

  /** A target uniformly distributed in the disc of radius `Radius`, reflected into the arena. */
  private def trip(x: Double, y: Double): Plan =
    val distance = Radius * math.sqrt(random.nextDouble())
    val angle = 2.0 * math.Pi * random.nextDouble()
    Plan.Heading(
      BrownianMotion.reflect(x + distance * math.cos(angle), width),
      BrownianMotion.reflect(y + distance * math.sin(angle), width)
    )

  private def pause(): Plan = Plan.Resting(-MeanPause * math.log(1.0 - random.nextDouble()))
end WaypointMotion

object WaypointMotion:
  /** Maximum distance of a waypoint from the position where the trip starts (m). */
  val Radius: Double = 100.0

  /** Mean of the exponential pauses between trips (s). */
  val MeanPause: Double = 60.0

  /** Mean distance of a uniform point of the disc from its centre (m). */
  val MeanTripLength: Double = 2.0 * Radius / 3.0

  private enum Plan:
    case Resting(remaining: Double)
    case Heading(targetX: Double, targetY: Double)
