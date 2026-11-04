package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.model.deployments.Grid
import it.unibo.alchemist.model.reactions.Redeployment.*
import it.unibo.alchemist.model.{Environment, Node, Position}
import it.unibo.learning.Box
import org.apache.commons.math3.random.Well19937c

import scala.jdk.CollectionConverters.*

/** Starts every episode on a fresh, randomly perturbed grid of nodes that are clones of the initial ones. */
final class Redeployment[T, P <: Position[P]](environment: Environment[T, P], box: Box, runSeed: Int):
  private var prototypes: List[Node[T]] = List.empty

  /** Remembers the nodes as they are at the beginning of the simulation. */
  def capturePrototypes(): Unit = prototypes = nodes.map(_.cloneNode())

  /** The deployment of `episode`, reproducible for a given run. */
  def positionsFor(episode: Int): List[P] =
    val random = new Well19937c(runSeed.toLong * SeedMultiplier + episode + 1L)
    new Grid[P](environment, random, 0, 0, box.width, box.width, box.step, box.step, box.randomness, box.randomness)
      .stream()
      .iterator()
      .asScala
      .toList

  /** Replaces the current nodes with fresh clones of the prototypes in `positions`. */
  def redeploy(positions: List[P]): Unit =
    nodes.foreach(environment.removeNode)
    positions.zip(Iterator.continually(prototypes).flatten).foreach { case (position, prototype) =>
      environment.addNode(prototype.cloneNode(), position)
    }

  private def nodes: List[Node[T]] = environment.getNodes.getCurrent.asScala.toList

object Redeployment:

  /** Spreads the run seed, so that the deployments of different runs do not overlap. */
  val SeedMultiplier: Long = 1000L
