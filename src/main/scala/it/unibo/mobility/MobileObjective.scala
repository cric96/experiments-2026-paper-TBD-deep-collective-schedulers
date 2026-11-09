package it.unibo.mobility

/** Time-integrated accuracy and per-round energy, using seconds and one round as reference units. */
object MobileObjective:
  /** Bound replay memory when relabelling over both tolerance and beta. */
  val ReplayCapacityPerRegime: Int = 25000

  /** Seconds for which holding the whole loss scale costs as much as a round at β 0.5, unless configured otherwise. */
  val ReferenceTime: Double = 1.0
  val RoundCost: Double = 1.0

  def loss(error: Double, tolerance: Double, scale: Double): Double =
    require(tolerance >= 0.0 && tolerance.isFinite && scale > 0.0)
    if !error.isFinite then 1.0 else math.min(1.0, math.max(0.0, error - tolerance) / scale)

  def reward(errorIntegral: Double, energy: Double, beta: Double): Double =
    -beta * errorIntegral / ReferenceTime - (1.0 - beta) * energy

/** The per-node signal integrated into the cost of a mobile decision. */
enum CostSignal:
  /** The error w.r.t. the ground truth that the node holds while it sleeps. */
  case Error

  /** The gap between the held value and the one a round would compute now ([[Staleness]]). */
  case Staleness

  /** The mean error of the node and of its neighbours ([[NeighbourhoodError]]). */
  case Neighbourhood
