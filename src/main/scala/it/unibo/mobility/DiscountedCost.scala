package it.unibo.mobility

/** The cost a decision taken at `start` pays for its sleep, discounted like the rewards that follow it: ∫_start^time
  * γ^(τ−start) c(τ) dτ of a piecewise-constant cost rate c. The learners bootstrap the next decision with γ^Δt, so an
  * undiscounted cost would charge a long sleep for the costs of the network that no decision can avoid (with a cost
  * summed over the network, a large part of it). It is advanced at every observation of the node, between which the
  * rate is constant.
  */
final case class DiscountedCost(gamma: Double, start: Double, time: Double, rate: Double, integral: Double):
  require(gamma > 0.0 && gamma <= 1.0 && time >= start)

  /** Integrates the held rate up to `now`. */
  def advance(now: Double): DiscountedCost =
    require(now >= time, s"Clock moved backwards: $now < $time")
    if now == time || rate == 0.0 then copy(time = now)
    else copy(time = now, integral = integral + rate * DiscountedCost.duration(gamma, time - start, now - start))

  /** Integrates the held rate up to `now`, then holds `value`. */
  def observe(now: Double, value: Double): DiscountedCost = advance(now).copy(rate = value)

  /** The cost of a new decision at `now`, which holds the current rate. */
  def restart(now: Double): DiscountedCost =
    require(now >= time, s"Clock moved backwards: $now < $time")
    DiscountedCost(gamma, now, rate)
end DiscountedCost

object DiscountedCost:
  /** The cost of a decision at `now`, holding `rate`, before it accrues anything. */
  def apply(gamma: Double, now: Double, rate: Double): DiscountedCost = DiscountedCost(gamma, now, now, rate, 0.0)

  /** ∫_from^to γ^s ds = (γ^from − γ^to) / ln(1/γ): the discounted length of the interval, its length when γ is one. */
  def duration(gamma: Double, from: Double, to: Double): Double =
    if gamma == 1.0 then to - from else (math.pow(gamma, from) - math.pow(gamma, to)) / math.log(1.0 / gamma)
