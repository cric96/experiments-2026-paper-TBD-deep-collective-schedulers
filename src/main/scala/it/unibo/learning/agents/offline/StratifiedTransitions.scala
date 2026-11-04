package it.unibo.learning.agents.offline

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** Transitions collected by an offline learner, split by regime: instability is rare, so batches are balanced to
  * contain as many transitions from unstable states as from stable ones.
  */
final class StratifiedTransitions[T](capacityPerRegime: Int = Int.MaxValue):
  require(capacityPerRegime > 0)
  private var activeScenario: Option[Int] = None
  private val scenarios = scala.collection.mutable.LinkedHashMap.empty[Int, StratifiedTransitions[T]]

  /** Each scenario has its own bounded stable/unstable reservoirs. */
  def scenario(value: Int): Unit = activeScenario = Some(value)

  private val unstable: ArrayBuffer[T] = ArrayBuffer.empty
  private val stable: ArrayBuffer[T] = ArrayBuffer.empty

  private var unstableCursor: Int = 0
  private var stableCursor: Int = 0

  def add(transition: T, isUnstable: Boolean): Unit =
    if activeScenario.nonEmpty then
      scenarios
        .getOrElseUpdate(activeScenario.getOrElse(0), StratifiedTransitions[T](capacityPerRegime))
        .add(transition, isUnstable)
    else addLocal(transition, isUnstable)

  private def addLocal(transition: T, isUnstable: Boolean): Unit =
    val regime = if isUnstable then unstable else stable
    val cursor = if isUnstable then unstableCursor else stableCursor
    if regime.size < capacityPerRegime then regime += transition else regime(cursor) = transition
    val next = (cursor + 1) % capacityPerRegime
    if isUnstable then unstableCursor = next else stableCursor = next

  def size: Int = unstable.size + stable.size + scenarios.values.map(_.size).sum

  def isEmpty: Boolean = size == 0

  /** Samples with replacement half of the batch from the unstable regime and the rest from the stable one (or the whole
    * batch from the only non-empty regime).
    */
  def sample(batchSize: Int, random: Random): Vector[T] =
    if scenarios.nonEmpty then
      val reservoirs = scenarios.values.toVector
      val offset = random.nextInt(reservoirs.size)
      reservoirs.indices.toVector.flatMap { index =>
        val count = batchSize / reservoirs.size + (if index < batchSize % reservoirs.size then 1 else 0)
        reservoirs((index + offset) % reservoirs.size).sample(count, random)
      }
    else if unstable.isEmpty then draw(stable, batchSize, random)
    else if stable.isEmpty then draw(unstable, batchSize, random)
    else
      val unstableHalf = draw(unstable, batchSize / 2, random)
      unstableHalf ++ draw(stable, batchSize - batchSize / 2, random)

  /** The first `perRegime` transitions of each regime, unstable ones first. */
  def oldest(perRegime: Int): Vector[T] =
    if scenarios.nonEmpty then scenarios.values.toVector.flatMap(_.oldest(perRegime))
    else (oldestIn(unstable, unstableCursor, perRegime) ++ oldestIn(stable, stableCursor, perRegime)).toVector

  private def oldestIn(regime: ArrayBuffer[T], cursor: Int, count: Int): Seq[T] =
    if regime.size < capacityPerRegime then regime.take(count).toVector
    else (regime.drop(cursor) ++ regime.take(cursor)).take(count).toVector

  private def draw(regime: ArrayBuffer[T], count: Int, random: Random): Vector[T] =
    Vector.fill(count)(regime(random.nextInt(regime.size)))
end StratifiedTransitions
