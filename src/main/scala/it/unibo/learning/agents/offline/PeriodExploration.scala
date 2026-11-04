package it.unibo.learning.agents.offline

/** Exploration changes coverage, never the action bounds or wake-up semantics. */
enum PeriodExploration:
  case Uniform, LogUniform

  def period(min: Double, max: Double, quantile: Double): Double =
    require(min > 0.0 && max >= min && quantile >= 0.0 && quantile <= 1.0)
    this match
      case Uniform => min + quantile * (max - min)
      case LogUniform => math.max(min, math.min(max, math.exp(math.log(min) + quantile * math.log(max / min))))
