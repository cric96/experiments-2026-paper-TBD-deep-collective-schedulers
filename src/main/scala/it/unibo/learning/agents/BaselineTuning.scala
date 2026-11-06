package it.unibo.learning.agents

import it.unibo.learning.TradeOff
import it.unibo.learning.TradeOff.RewardShape

/** The periods of a reactive baseline: the one it falls back to after a change, and the longest one it grows to. */
final case class ReactivePeriods(shortest: Double, longest: Double)

/** How β tunes the periods of the reactive baselines ([[TrickleAgent]], [[GradualAgent]]). The simulation files select
  * a tuning by its ordinal (`tunedBaselines`), so new cases go last.
  */
enum BaselineTuning:
  /** The whole admissible range, whatever β. */
  case Untuned

  /** After a change, the best period of an urgent node on the linear scale ([[TradeOff.urgentPeriod]]); up to the
    * longest admissible period otherwise.
    */
  case UrgentPeriod

  /** Between the best periods of a fully unstable node and of a stable one under the reward shape: on the full-range
    * scale both shrink to the fastest period as β grows to 1.
    */
  case BestPeriods

  /** The periods of a baseline tuned for `beta` under `shape`, within [`min`, `max`]. */
  def periods(beta: Double, shape: RewardShape, min: Double, max: Double): ReactivePeriods = this match
    case Untuned => ReactivePeriods(min, max)
    case UrgentPeriod => ReactivePeriods(TradeOff.urgentPeriod(beta, min, max), max)
    case BestPeriods =>
      ReactivePeriods(
        TradeOff.bestPeriod(BaselineTuning.Unstable, beta, shape, min, max),
        TradeOff.bestPeriod(BaselineTuning.Stable, beta, shape, min, max)
      )
end BaselineTuning

object BaselineTuning:
  /** Instability intensity of a node whose output just changed, when a baseline falls back to its shortest period. */
  val Unstable: Double = 1.0

  /** Instability intensity of a node whose output stays the same, when a baseline grows to its longest period. */
  val Stable: Double = 0.0
