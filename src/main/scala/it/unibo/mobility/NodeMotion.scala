package it.unibo.mobility

import org.apache.commons.math3.random.RandomGenerator

/** How the nodes move during one episode: where a node is after one movement interval. */
trait NodeMotion:
  def move(id: Int, x: Double, y: Double): (Double, Double)

/** The node motions of the mobile experiment. */
enum MobilityModel:
  /** Every node diffuses with the same coefficient, all the time. */
  case Brownian

  /** Every node alternates pauses and straight trips ([[WaypointMotion]]). */
  case Waypoint

object NodeMotion:
  /** The motion of `episode`, drawing from `random` (one stream per episode). */
  def forEpisode(config: MobilityConfig, episode: Int, random: RandomGenerator, width: Double): NodeMotion =
    config.model match
      case MobilityModel.Brownian =>
        val brownian = BrownianMotion(random, width)
        (_, x, y) => brownian.move(x, y, config.levelAt(episode), config.movementInterval)
      case MobilityModel.Waypoint => WaypointMotion(random, width, config.levelAt(episode), config.movementInterval)
