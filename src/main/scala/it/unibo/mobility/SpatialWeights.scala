package it.unibo.mobility

import scala.collection.immutable.Queue

/** The weights ρ^hop ([[NetworkError.weight]]) between every pair of nodes of a network, by breadth-first search over
  * its links; zero between nodes that no path connects. The costs of all the nodes are a product of the weights by the
  * errors, computed at every simulator event, so the weights are rows of arrays indexed like the ascending ids.
  */
final class SpatialWeights private (ids: IndexedSeq[Int], rows: IndexedSeq[Array[Double]]):
  private val indexOf: Map[Int, Int] = ids.zipWithIndex.toMap

  /** The network error ([[NetworkError.of]]) charged to every node; a node without an error counts as unbounded. */
  def costs(errors: Map[Int, Double], cap: Double): Map[Int, Double] =
    val held = ids.map(id => errors.getOrElse(id, Double.PositiveInfinity)).toArray
    ids.zip(rows).map((id, weights) => id -> NetworkError.of(held, weights, cap)).toMap

  /** The mean total weight of a node: how many nodes its cost counts, on average. */
  def effectiveSize: Double = if ids.isEmpty then 1.0 else rows.map(_.sum).sum / ids.size

  /** The nodes whose cost weighs at least one of the `changed` ones. */
  def reachedFrom(changed: Set[Int]): Set[Int] =
    val columns = changed.flatMap(indexOf.get)
    ids.zip(rows).collect { case (id, weights) if columns.exists(weights(_) > 0.0) => id }.toSet

object SpatialWeights:
  def apply(links: Map[Int, Seq[Int]], discount: Double): SpatialWeights =
    require(discount >= 0.0 && discount <= 1.0, s"The spatial discount must be within [0, 1]: $discount")
    val ids = links.keys.toIndexedSeq.sorted
    val rows = ids.map { origin =>
      val distances = hops(links, origin)
      ids.map(id => distances.get(id).fold(0.0)(NetworkError.weight(discount, _))).toArray
    }
    new SpatialWeights(ids, rows)

  /** The hops of the shortest paths from `origin` to every node it reaches through `links`. */
  def hops(links: Map[Int, Seq[Int]], origin: Int): Map[Int, Int] =
    @annotation.tailrec
    def visit(frontier: Queue[Int], reached: Map[Int, Int]): Map[Int, Int] = frontier.dequeueOption match
      case None => reached
      case Some((node, rest)) =>
        val next = links.getOrElse(node, Seq.empty).filterNot(reached.contains).distinct
        visit(rest.enqueueAll(next), reached ++ next.map(_ -> (reached(node) + 1)))
    visit(Queue(origin), Map(origin -> 0))
end SpatialWeights
