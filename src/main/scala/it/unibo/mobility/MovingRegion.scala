package it.unibo.mobility

import org.apache.commons.math3.random.RandomGenerator

/** The nodes that start moving in an [[EpisodeEvent.RegionStartsMoving]] event: the given share of the nodes closest to
  * a centre drawn uniformly in the arena, so that the moving and the still nodes form two contiguous regions.
  */
object MovingRegion:
  /** Share of the nodes that start moving. */
  val DefaultShare: Double = 0.5

  def centre(random: RandomGenerator, width: Double): (Double, Double) =
    (random.nextDouble() * width, random.nextDouble() * width)

  def select(positions: Map[Int, (Double, Double)], centre: (Double, Double), share: Double): Set[Int] =
    val (centreX, centreY) = centre
    positions.toSeq
      .sortBy { case (id, (x, y)) => (math.hypot(x - centreX, y - centreY), id) }
      .take(math.ceil(share * positions.size).toInt)
      .map(_._1)
      .toSet
