package it.unibo.mobility

/** Integrates a piecewise-constant held error. Observe after each simulator event, not only node rounds.
  *
  * It is advanced for every node at every simulator event, so the per-tolerance sums live in arrays indexed like
  * `tolerances`; an event at the current instant adds nothing and is skipped.
  */
final class ErrorIntegral(tolerances: Seq[Double], scale: Double):
  private val levels: Array[Double] = tolerances.toArray
  private val indexOf: Map[Double, Int] = tolerances.zipWithIndex.toMap
  private val costs: Array[Double] = Array.fill(levels.length)(0.0)
  private val within: Array[Double] = Array.fill(levels.length)(0.0)
  private var time: Double = 0.0
  private var error: Double = Double.PositiveInfinity
  private var reachable: Boolean = false
  private var reachableDuration: Double = 0.0
  private var raw: Double = 0.0
  private var absolute: Double = 0.0

  def advance(now: Double): Unit =
    require(now >= time, s"Clock moved backwards: $now < $time")
    if now > time then accumulate(now - time)
    time = now

  /** Adds `duration` seconds of the held error; a hot path, hence the index loop. */
  private def accumulate(duration: Double): Unit =
    var level = 0
    while level < levels.length do
      costs(level) += MobileObjective.loss(error, levels(level), scale) * duration
      within(level) += (if error <= levels(level) then duration else 0.0)
      level += 1
    absolute += (if error.isFinite then error else scale) * duration
    raw += MobileObjective.loss(error, 0.0, scale) * duration
    if reachable then reachableDuration += duration

  def observe(now: Double, currentError: Double, reachable: Boolean): Unit =
    advance(now)
    error = currentError
    this.reachable = reachable

  def cost(tolerance: Double): Double = costs(indexOf(tolerance))
  def withinTime(tolerance: Double): Double = within(indexOf(tolerance))
  def reachableTime: Double = reachableDuration
  def rawIntegral: Double = raw
  def absoluteIntegral: Double = absolute
end ErrorIntegral
