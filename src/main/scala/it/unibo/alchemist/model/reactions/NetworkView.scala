package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.model.{Environment, Node, Position}
import it.unibo.scafi.{BuildingBlock, GroundTruth, GroundTruths, Sensors, Topology}

import scala.jdk.CollectionConverters.*

/** What the mobile experiment knows of the network: the neighbours of every node, the sources, the nodes they reach and
  * the ground truth of the running `block`. They are read at every simulator event, so they are cached until the
  * experiment reports that the topology (positions, population) or the sources changed; `version` counts the changes.
  */
final class NetworkView[T, P <: Position[P]](
    environment: Environment[T, P],
    nodes: () => List[Node[T]],
    block: () => BuildingBlock
):
  private var neighbourhoods: Option[Map[Int, Seq[(Int, Double)]]] = None
  private var sourceIds: Option[Set[Int]] = None
  private var reachability: Option[Map[Int, Double]] = None
  private var truths: Option[Map[Int, Double]] = None
  private var changes: Long = 0L

  /** How many times the topology or the sources changed: equal versions see the same network. */
  def version: Long = changes

  /** Forgets what depends on the positions and on the population, and on the sources. */
  def topologyChanged(): Unit =
    neighbourhoods = None
    sourcesChanged()

  /** Forgets what depends on the sources: their ids, the nodes they reach and the ground truths. */
  def sourcesChanged(): Unit =
    sourceIds = None
    reachability = None
    truths = None
    changes += 1

  /** The current neighbours of every node, with their distances. */
  def links: Map[Int, Seq[(Int, Double)]] = neighbourhoods.getOrElse {
    val calculated = nodes().map { node =>
      val position = environment.getCurrentPosition(node)
      node.getId -> environment.getNeighborhood(node).getCurrent.getNeighbors.asScala.toSeq.map { neighbour =>
        neighbour.getId -> position.distanceTo(environment.getCurrentPosition(neighbour))
      }
    }.toMap
    neighbourhoods = Some(calculated)
    calculated
  }

  def neighboursOf(id: Int): Seq[(Int, Double)] = links.getOrElse(id, Seq.empty)

  /** The ids of the current sources. */
  def sources: Set[Int] = sourceIds.getOrElse {
    val calculated = nodes().filter(NetworkView.isSource).map(_.getId).toSet
    sourceIds = Some(calculated)
    calculated
  }

  /** Whether a source reaches `id` through the current links. */
  def isReachable(id: Int): Boolean =
    reachability
      .getOrElse {
        val calculated = GroundTruth.shortestPaths(topology, sources)
        reachability = Some(calculated)
        calculated
      }
      .getOrElse(id, Double.PositiveInfinity)
      .isFinite

  /** The value the running block should hold in every node. */
  def groundTruths: Map[Int, Double] = truths.getOrElse {
    val calculated = block() match
      case BuildingBlock.Gradient => GroundTruth.shortestPaths(topology, sources)
      case other =>
        val reference = GroundTruths.of(topology, sources)
        nodes().map(node => node.getId -> reference.of(node.getId, other)).toMap
    truths = Some(calculated)
    calculated
  }

  private def topology: Topology = Topology(nodes().map(_.getId), links)
end NetworkView

object NetworkView:
  def isSource[T](node: Node[T]): Boolean = node.valueOf[Boolean](Sensors.source).getOrElse(false)
