package it.unibo.mobility

import org.apache.commons.math3.random.RandomGenerator

/** Seeded Gaussian increments: each coordinate has variance 2 D dt, reflected into the arena. */
final class BrownianMotion(random: RandomGenerator, width: Double):
  require(width > 0.0 && width.isFinite)

  def move(x: Double, y: Double, diffusion: Double, duration: Double): (Double, Double) =
    require(diffusion >= 0.0 && diffusion.isFinite && duration >= 0.0 && duration.isFinite)
    val deviation = math.sqrt(2.0 * diffusion * duration)
    val nextX = x + deviation * random.nextGaussian()
    val nextY = y + deviation * random.nextGaussian()
    if diffusion == 0.0 then (x, y)
    else (BrownianMotion.reflect(nextX, width), BrownianMotion.reflect(nextY, width))

object BrownianMotion:
  def reflect(coordinate: Double, width: Double): Double =
    val wrapped = ((coordinate % (2.0 * width)) + 2.0 * width) % (2.0 * width)
    if wrapped <= width then wrapped else 2.0 * width - wrapped
